from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.action_features import GLOBAL_STATE_NAMES, LOCAL_FEATURE_NAMES, build_dynamic_action_features, summarize_global_state
from models.candidates import _candidate_score, build_dynamic_shortlist, build_topm_actions
from models.evaluate import GKDEvaluator, LiveEdgeWorldCache
from models.gkd_recruiter import StateAwareDuelingQNetwork
from models.marginal_q_network import CandidateAwareDuelingQNetwork, MarginalQNetwork
from models.runtime import maybe_compile


def load_graph(env_dir: Path) -> nx.DiGraph:
    edge_index = np.atleast_2d(np.loadtxt(env_dir / 'edge_index.txt', dtype=int))
    weights = np.loadtxt(env_dir / 'w_ij.txt', dtype=float)
    graph = nx.DiGraph()
    graph.add_weighted_edges_from((int(u), int(v), float(w)) for (u, v), w in zip(edge_index, weights))
    return graph


def make_state(current_step: int, budget_k: int, current_ets: float, task_ets: np.ndarray, worker_load: np.ndarray, u_max: int) -> torch.Tensor:
    remain = 1 - current_step / max(int(budget_k), 1)
    load = worker_load / max(int(u_max), 1)
    vec = np.concatenate([[remain, current_ets], task_ets, load]).astype(np.float32)
    return torch.tensor(vec, dtype=torch.float32)


def _worker_influence_proxy(graph: nx.DiGraph, worker_indices: np.ndarray) -> np.ndarray:
    worker_to_local = {int(worker): idx for idx, worker in enumerate(worker_indices.tolist())}
    influence = np.zeros(len(worker_indices), dtype=np.float32)
    for u, v in graph.edges():
        if int(u) in worker_to_local:
            influence[worker_to_local[int(u)]] += 1.0
        if int(v) in worker_to_local:
            influence[worker_to_local[int(v)]] += 1.0
    return influence


def _selected_worker_mask(selected_pairs: List[Tuple[int, int]], worker_to_local: Dict[int, int], num_tasks: int, num_workers: int) -> np.ndarray:
    mask = np.zeros((num_tasks, num_workers), dtype=np.float32)
    for worker_id, task_idx in selected_pairs:
        local = worker_to_local.get(int(worker_id))
        if local is not None:
            mask[int(task_idx), int(local)] = 1.0
    return mask


def _load_policy_network(q_network: str, checkpoint_path: Path, pair_dim: int, state_dim: int, local_dim: int, device: torch.device):
    if q_network == 'marginal':
        net = MarginalQNetwork(pair_dim, state_dim, local_dim).to(device)
    elif q_network == 'candidate_aware_dueling':
        net = CandidateAwareDuelingQNetwork(pair_dim, state_dim, local_dim).to(device)
    else:
        net = StateAwareDuelingQNetwork(pair_dim // 2, state_dim).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    net.eval()
    if hasattr(net, 'disable_noise'):
        net.disable_noise()
    return net


def rollout_gkd(
    env_dir: Path,
    pretrain_dir: Path,
    checkpoint_path: Path,
    budget_k: int,
    top_m: int,
    u_max: int,
    num_simulations: int,
    seed: int,
    step_simulations: int = 1,
    compile_model: bool = False,
    q_network: str = 'dueling',
    policy: str = 'q',
    dynamic_candidates: bool = False,
    wide_candidate_size: int = 2048,
    shortlist_size: int = 256,
    residual_threshold: float = 0.005,
    rollout_log_path: Path | None = None,
    decision_worlds: List[int] | None = None,
    worker_embeds_np: np.ndarray | None = None,
) -> Dict[str, object]:
    graph = load_graph(env_dir)
    q_workers = np.atleast_2d(np.loadtxt(env_dir / 'q_matrix.txt', dtype=float))
    a_workers = np.atleast_2d(np.loadtxt(env_dir / 'a_matrix.txt', dtype=float))
    demands = np.atleast_1d(np.loadtxt(env_dir / 'task_demands.txt', dtype=float))
    worker_indices = np.atleast_1d(np.loadtxt(env_dir / 'worker_indices.txt', dtype=int))
    full_q = np.atleast_2d(np.loadtxt(env_dir / 'full_q_matrix.txt', dtype=float))
    full_a = np.atleast_2d(np.loadtxt(env_dir / 'full_a_matrix.txt', dtype=float))

    evaluator = GKDEvaluator(graph, q_workers, a_workers, demands, worker_indices, num_simulations, full_q, full_a, seed)
    step_evaluator = GKDEvaluator(graph, q_workers, a_workers, demands, worker_indices, step_simulations, full_q, full_a, seed)
    decision_cache = LiveEdgeWorldCache() if decision_worlds is not None else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    worker_embeds = torch.load(pretrain_dir / 'distilled_worker_embeds.pt', map_location=device, weights_only=True)
    task_embeds = torch.load(pretrain_dir / 'distilled_task_embeds.pt', map_location=device, weights_only=True)
    if worker_embeds_np is None:
        worker_embeds_np = worker_embeds.detach().cpu().numpy()
    worker_capacity = np.full(len(worker_indices), u_max, dtype=np.float32)
    worker_to_local = {int(worker): idx for idx, worker in enumerate(worker_indices.tolist())}
    worker_influence_proxy = _worker_influence_proxy(graph, worker_indices)
    pair_dim = worker_embeds.shape[1] + task_embeds.shape[1]
    state_dim = len(GLOBAL_STATE_NAMES) if q_network in {'marginal', 'candidate_aware_dueling'} else 2 + len(demands) + len(worker_indices)
    local_dim = len(LOCAL_FEATURE_NAMES)
    net = None
    if policy == 'q':
        net = _load_policy_network(q_network, checkpoint_path, pair_dim, state_dim, local_dim, device)
        if compile_model:
            net = maybe_compile(net, True)

    selected_pairs: List[Tuple[int, int]] = []
    selected_set: set[int] = set()
    worker_load = np.zeros(len(worker_indices), dtype=int)
    task_ets = np.zeros(len(demands), dtype=float)
    current_ets = 0.0

    start = time.perf_counter()
    candidate_build_time_sec = 0.0
    feature_build_time_sec = 0.0
    q_scoring_time_sec = 0.0
    simulation_time_sec = 0.0
    state_update_time_sec = 0.0
    decision_logs: List[Dict[str, object]] = []

    for step in range(int(budget_k)):
        task_selected_count = np.bincount([task for _, task in selected_pairs], minlength=len(demands)).astype(np.float32)
        selected_mask = _selected_worker_mask(selected_pairs, worker_to_local, len(demands), len(worker_indices))
        candidate_start = time.perf_counter()
        if dynamic_candidates:
            actions_np, action_ids, local_features_np = build_dynamic_shortlist(
                q_workers,
                a_workers,
                demands,
                worker_load,
                worker_capacity,
                task_ets,
                task_selected_count,
                step,
                budget_k,
                top_m,
                wide_candidate_size,
                shortlist_size,
                residual_threshold,
                worker_influence_proxy,
                selected_mask,
                worker_embeds_np,
            )
        else:
            actions_np, action_ids = build_topm_actions(q_workers, a_workers, top_m, demands, False)
            local_features_np = build_dynamic_action_features(
                q_workers,
                a_workers,
                demands,
                worker_load,
                worker_capacity,
                task_ets,
                task_selected_count,
                step,
                budget_k,
                actions_np,
                worker_influence_proxy,
            )
        candidate_build_time_sec += time.perf_counter() - candidate_start
        if len(actions_np) == 0:
            break
        valid = np.ones(len(action_ids), dtype=bool)
        for idx, (worker_local, _) in enumerate(actions_np):
            if worker_load[int(worker_local)] >= int(u_max) or int(action_ids[idx]) in selected_set:
                valid[idx] = False
        valid_local = np.flatnonzero(valid)
        if len(valid_local) == 0:
            break

        feature_start = time.perf_counter()
        valid_mask = torch.zeros(len(actions_np), dtype=torch.bool, device=device)
        valid_local_tensor = torch.as_tensor(valid_local, dtype=torch.long, device=device)
        valid_mask[valid_local_tensor] = True
        action_worker_idx = torch.as_tensor(actions_np[:, 0], dtype=torch.long, device=device)
        action_task_idx = torch.as_tensor(actions_np[:, 1], dtype=torch.long, device=device)
        pair_features = torch.cat([
            worker_embeds[action_worker_idx],
            task_embeds[action_task_idx],
        ], dim=-1)
        local_features = torch.tensor(local_features_np, dtype=torch.float32, device=device)
        marginal_state = torch.tensor(
            summarize_global_state(current_ets, step, budget_k, task_ets, worker_load, worker_capacity),
            dtype=torch.float32,
            device=device,
        )
        state = make_state(step, budget_k, current_ets, task_ets, worker_load, u_max).to(device)
        feature_build_time_sec += time.perf_counter() - feature_start

        score_start = time.perf_counter()
        if policy == 'heuristic':
            q_values = torch.tensor(_candidate_score(local_features_np), dtype=torch.float32, device=device)
            q_values = q_values.masked_fill(~valid_mask, torch.finfo(q_values.dtype).min)
        elif q_network == 'marginal':
            q_values = net(marginal_state.unsqueeze(0), pair_features.unsqueeze(0), local_features.unsqueeze(0), valid_mask=valid_mask.unsqueeze(0)).squeeze(0)
        elif q_network == 'candidate_aware_dueling':
            q_values = net(marginal_state.unsqueeze(0), pair_features.unsqueeze(0), local_features.unsqueeze(0), valid_mask=valid_mask.unsqueeze(0)).squeeze(0)
        else:
            actions = torch.tensor(actions_np, dtype=torch.long, device=device)
            pair_encoded = net.encode_pairs(worker_embeds, task_embeds, actions)
            q_values = net(state[None], worker_embeds.unsqueeze(0), task_embeds.unsqueeze(0), actions, pair_features=pair_encoded, valid_mask=valid_mask.unsqueeze(0)).squeeze(0)
        q_scoring_time_sec += time.perf_counter() - score_start

        teacher_gains = None
        if decision_worlds is not None:
            teacher_gains = np.full(len(actions_np), np.nan, dtype=np.float32)
            for idx in valid_local.tolist():
                worker_local, task_idx = actions_np[int(idx)]
                worker_id = int(worker_indices[int(worker_local)])
                current_task_workers = {w for w, t in selected_pairs if t == int(task_idx)}
                gain = step_evaluator.evaluate_task_delta_with_worlds(
                    int(task_idx),
                    old_seeds=current_task_workers,
                    new_seeds=current_task_workers | {worker_id},
                    world_seeds=decision_worlds,
                    cache=decision_cache,
                )
                teacher_gains[int(idx)] = float(gain) / max(len(demands), 1)

        best_local = int(torch.argmax(q_values).item())
        worker_local, task_idx = actions_np[best_local]
        global_action = int(action_ids[best_local])
        selected_set.add(global_action)
        worker_load[int(worker_local)] += 1
        selected_pairs.append((int(worker_indices[int(worker_local)]), int(task_idx)))

        simulation_start = time.perf_counter()
        old_task_workers = {w for w, t in selected_pairs[:-1] if t == int(task_idx)}
        new_task_workers = old_task_workers | {int(worker_indices[int(worker_local)])}
        if decision_worlds is not None:
            old_task = step_evaluator.evaluate_task_with_worlds(int(task_idx), old_task_workers, decision_worlds, cache=decision_cache)
            new_task = step_evaluator.evaluate_task_with_worlds(int(task_idx), new_task_workers, decision_worlds, cache=decision_cache)
        else:
            old_task = step_evaluator.evaluate_task(int(task_idx), old_task_workers, seed=seed + step + 1)
            new_task = step_evaluator.evaluate_task(int(task_idx), new_task_workers, seed=seed + step + 1)
        simulation_time_sec += time.perf_counter() - simulation_start

        state_start = time.perf_counter()
        chosen_gain = float(new_task - old_task) / max(len(demands), 1)
        current_ets += chosen_gain
        task_ets[int(task_idx)] = float(new_task)
        oracle_gain = float(np.nanmax(teacher_gains)) if teacher_gains is not None and np.isfinite(teacher_gains).any() else chosen_gain
        decision_logs.append({
            'step': int(step),
            'worker_local': int(worker_local),
            'worker_id': int(worker_indices[int(worker_local)]),
            'task_idx': int(task_idx),
            'action_id': int(global_action),
            'q_value': float(q_values[best_local].item()),
            'teacher_marginal_gain': float(chosen_gain),
            'oracle_best_gain': float(oracle_gain),
            'oracle_regret': float(max(0.0, oracle_gain - chosen_gain)),
            'candidate_count': int(len(actions_np)),
            'valid_candidate_count': int(len(valid_local)),
        })
        state_update_time_sec += time.perf_counter() - state_start

    elapsed = time.perf_counter() - start
    final_start = time.perf_counter()
    final = evaluator.evaluate(selected_pairs, seed=seed)
    final_evaluation_time_sec = time.perf_counter() - final_start
    if rollout_log_path is not None:
        rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_log_path.write_text(json.dumps(decision_logs, indent=2, ensure_ascii=False), encoding='utf-8')
    online_decision_time_sec = candidate_build_time_sec + feature_build_time_sec + q_scoring_time_sec + state_update_time_sec
    return {
        'seed_pairs': selected_pairs,
        'selection_time_sec': float(online_decision_time_sec),
        'candidate_build_time_sec': float(candidate_build_time_sec),
        'feature_build_time_sec': float(feature_build_time_sec),
        'q_scoring_time_sec': float(q_scoring_time_sec),
        'policy_time_sec': float(feature_build_time_sec + q_scoring_time_sec),
        'online_decision_time_sec': float(online_decision_time_sec),
        'simulation_time_sec': float(simulation_time_sec),
        'state_update_time_sec': float(state_update_time_sec),
        'rollout_time_sec': float(elapsed),
        'final_evaluation_time_sec': float(final_evaluation_time_sec),
        'ets': float(final['Effective_Task_Satisfaction']),
        'spread': float(final['Expected_Influence_Spread']),
        'unique_seed_users': int(len({w for w, _ in selected_pairs})),
        'budget_used': int(len(selected_pairs)),
        'params': {
            'top_m': int(top_m),
            'u_max': int(u_max),
            'num_simulations': int(num_simulations),
            'checkpoint': str(checkpoint_path),
            'q_network': q_network,
            'policy': policy,
            'dynamic_candidates': bool(dynamic_candidates),
            'rollout_log_path': None if rollout_log_path is None else str(rollout_log_path),
        },
        'decision_log': decision_logs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-dir', required=True)
    parser.add_argument('--pretrain-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', default='gowalla_v3000_seed42')
    parser.add_argument('--budgets', nargs='+', type=int, default=[25, 50, 75, 100, 150])
    parser.add_argument('--top-m', type=int, default=5)
    parser.add_argument('--u-max', type=int, default=1)
    parser.add_argument('--num-simulations', type=int, default=10)
    parser.add_argument('--step-simulations', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--csv-out', default='reports/gkd_inference_results.csv')
    parser.add_argument('--json-out', default='reports/gkd_inference_results.json')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--q-network', choices=['dueling', 'candidate_aware_dueling', 'marginal'], default='marginal')
    parser.add_argument('--policy', choices=['q', 'heuristic'], default='q')
    parser.add_argument('--dynamic-candidates', action='store_true')
    parser.add_argument('--wide-candidate-size', type=int, default=2048)
    parser.add_argument('--shortlist-size', type=int, default=256)
    parser.add_argument('--residual-threshold', type=float, default=0.005)
    parser.add_argument('--rollout-log', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_dir = Path(args.env_dir)
    pretrain_dir = Path(args.pretrain_dir)
    checkpoint = Path(args.checkpoint)
    rows = []
    for budget in args.budgets:
        result = rollout_gkd(
            env_dir=env_dir,
            pretrain_dir=pretrain_dir,
            checkpoint_path=checkpoint,
            budget_k=budget,
            top_m=args.top_m,
            u_max=args.u_max,
            num_simulations=args.num_simulations,
            seed=args.seed,
            step_simulations=args.step_simulations,
            compile_model=args.compile,
            q_network=args.q_network,
            policy=args.policy,
            dynamic_candidates=args.dynamic_candidates,
            wide_candidate_size=args.wide_candidate_size,
            shortlist_size=args.shortlist_size,
            residual_threshold=args.residual_threshold,
            rollout_log_path=None if args.rollout_log is None else Path(args.rollout_log),
        )
        rows.append({'budget_k': int(budget), **result})

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open('w', encoding='utf-8', newline='') as handle:
        fieldnames = ['budget_k', 'ets', 'spread', 'selection_time_sec', 'candidate_build_time_sec', 'feature_build_time_sec', 'q_scoring_time_sec', 'state_update_time_sec', 'policy_time_sec', 'budget_used', 'unique_seed_users', 'seed_pairs', 'decision_log', 'params']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items() if k in fieldnames})

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved CSV to {csv_out}')
    print(f'Saved JSON to {json_out}')


if __name__ == '__main__':
    main()
