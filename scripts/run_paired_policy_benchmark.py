from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.extended_baselines import SelectionResult, run_extended_baseline
from baselines.rl_baselines import rollout_dqn_selector_baseline, rollout_maim_baseline
from models.evaluate import GKDEvaluator, LiveEdgeWorldCache
from scripts.run_gkd_inference import load_graph, rollout_gkd


SeedPair = Tuple[int, int]


@dataclass
class BenchmarkPolicyResult:
    name: str
    seed_pairs: List[SeedPair]
    selection_time_sec: float
    candidate_build_time_sec: float
    feature_build_time_sec: float
    q_scoring_time_sec: float
    state_update_time_sec: float
    policy_time_sec: float
    params: Dict[str, object]
    decision_log: List[Dict[str, object]]


def _expand_seed_range(bounds: Sequence[int]) -> list[int]:
    values = [int(v) for v in bounds]
    if len(values) == 2:
        start, end = values
        return list(range(start, end + 1))
    return values


def _bootstrap_ci(samples: Sequence[float], num_bootstrap: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(int(seed))
    means = np.empty(num_bootstrap, dtype=float)
    n = arr.size
    for i in range(num_bootstrap):
        means[i] = float(np.mean(arr[rng.integers(0, n, size=n)]))
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def _load_env_arrays(env_dir: Path):
    q_workers = np.atleast_2d(np.loadtxt(env_dir / 'q_matrix.txt', dtype=float))
    a_workers = np.atleast_2d(np.loadtxt(env_dir / 'a_matrix.txt', dtype=float))
    demands = np.atleast_1d(np.loadtxt(env_dir / 'task_demands.txt', dtype=float))
    worker_indices = np.atleast_1d(np.loadtxt(env_dir / 'worker_indices.txt', dtype=int))
    full_q = np.atleast_2d(np.loadtxt(env_dir / 'full_q_matrix.txt', dtype=float))
    full_a = np.atleast_2d(np.loadtxt(env_dir / 'full_a_matrix.txt', dtype=float))
    return q_workers, a_workers, demands, worker_indices, full_q, full_a


def _quality_greedy_candidates(q_workers, a_workers, worker_indices, budget_k, top_m, u_max) -> SelectionResult:
    start = time.perf_counter()
    ranked = []
    loads = {}
    for local_w in range(q_workers.shape[0]):
        scores = q_workers[local_w] * a_workers[local_w]
        top_tasks = np.argsort(scores)[-int(top_m):][::-1]
        for t in top_tasks:
            ranked.append((float(scores[int(t)]), int(worker_indices[local_w]), int(t)))
    ranked.sort(reverse=True)
    selected = []
    seen = set()
    for _, w, t in ranked:
        if (w, t) in seen or loads.get(w, 0) >= int(u_max):
            continue
        selected.append((w, t))
        seen.add((w, t))
        loads[w] = loads.get(w, 0) + 1
        if len(selected) >= int(budget_k):
            break
    return SelectionResult('QualityGreedy', selected, time.perf_counter() - start, {'top_m': int(top_m), 'u_max': int(u_max)})


def _random_candidates(q_workers, a_workers, worker_indices, budget_k, top_m, u_max, seed) -> SelectionResult:
    start = time.perf_counter()
    candidates = []
    for local_w in range(q_workers.shape[0]):
        scores = q_workers[local_w] * a_workers[local_w]
        top_tasks = np.argsort(scores)[-int(top_m):]
        for t in top_tasks:
            candidates.append((int(worker_indices[local_w]), int(t)))
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(candidates))
    selected = []
    seen = set()
    loads = {}
    for idx in order:
        w, t = candidates[int(idx)]
        if (w, t) in seen or loads.get(w, 0) >= int(u_max):
            continue
        selected.append((w, t))
        seen.add((w, t))
        loads[w] = loads.get(w, 0) + 1
        if len(selected) >= int(budget_k):
            break
    return SelectionResult('Random', selected, time.perf_counter() - start, {'top_m': int(top_m), 'u_max': int(u_max), 'seed': int(seed)})


def _topm_candidates(q_workers, a_workers, worker_indices, top_m):
    candidates = []
    for local_w in range(q_workers.shape[0]):
        scores = q_workers[local_w] * a_workers[local_w]
        top_tasks = np.argsort(scores)[-int(top_m):][::-1]
        for t in top_tasks:
            candidates.append((int(worker_indices[local_w]), int(t), float(scores[int(t)])))
    return candidates


def _prune_celf_candidates(q_workers, a_workers, worker_indices, budget_k, top_m, u_max, celf_limit=None):
    scored = _topm_candidates(q_workers, a_workers, worker_indices, top_m)
    scored.sort(key=lambda item: (item[2], item[0], item[1]), reverse=True)
    if celf_limit is None:
        limit = len(scored)
    else:
        limit = min(len(scored), int(celf_limit))
        min_needed = min(len(scored), max(int(budget_k), int(budget_k) * max(1, int(u_max))))
        limit = max(limit, min_needed)
    pruned = [(w, t) for w, t, _ in scored[:limit]]
    unique_workers = len({w for w, _ in pruned})
    if unique_workers < int(budget_k):
        raise ValueError(
            f'CELF candidate pool has only {unique_workers} unique workers for budget {budget_k}; increase celf_limit/top_m or relax u_max.'
        )
    return pruned, limit, unique_workers


def _full_celf_candidates(q_workers, a_workers, worker_indices, budget_k, top_m, u_max):
    scored = _topm_candidates(q_workers, a_workers, worker_indices, top_m)
    candidates = [(w, t) for w, t, _ in scored]
    unique_workers = len({w for w, _ in candidates})
    if unique_workers < int(budget_k):
        raise ValueError(
            f'CELF full candidate pool has only {unique_workers} unique workers for budget {budget_k}; increase top_m or relax u_max.'
        )
    return candidates, len(candidates), unique_workers


def _objective_value(evaluator: GKDEvaluator, seed_pairs: list[SeedPair], decision_worlds: Sequence[int], cache: LiveEdgeWorldCache, objective: str) -> float:
    metrics = evaluator.evaluate_with_worlds(seed_pairs, decision_worlds, cache=cache)
    if objective == 'Expected_Influence_Spread':
        return float(metrics['Expected_Influence_Spread'])
    if objective == 'Effective_Task_Satisfaction':
        return float(metrics['Effective_Task_Satisfaction'])
    raise ValueError(f'unsupported CELF objective {objective}')


def _celf_static(q_workers, a_workers, worker_indices, budget_k, top_m, u_max, evaluator, decision_worlds, cache, celf_limit=None, method_name: str = 'CELF-ETS', objective: str = 'Effective_Task_Satisfaction', use_full_candidate_pool: bool = False) -> SelectionResult:
    start = time.perf_counter()
    if use_full_candidate_pool:
        candidates, effective_limit, unique_workers = _full_celf_candidates(
            q_workers, a_workers, worker_indices, budget_k, top_m, u_max
        )
        candidate_pool = 'top_m_full'
    else:
        candidates, effective_limit, unique_workers = _prune_celf_candidates(
            q_workers, a_workers, worker_indices, budget_k, top_m, u_max, celf_limit
        )
        candidate_pool = 'qxa_pruned'
    heap = []
    for w, t in candidates:
        gain = _objective_value(evaluator, [(w, t)], decision_worlds, cache, objective)
        heap.append((-float(gain), int(w), int(t), 0))
    import heapq
    heapq.heapify(heap)
    selected = []
    current = 0.0
    seen = set()
    loads = {}
    while heap and len(selected) < int(budget_k):
        neg_gain, w, t, stamp = heapq.heappop(heap)
        if (w, t) in seen or loads.get(w, 0) >= int(u_max):
            continue
        if stamp == len(selected):
            selected.append((w, t))
            seen.add((w, t))
            loads[w] = loads.get(w, 0) + 1
            current += -neg_gain
        else:
            true_gain = _objective_value(evaluator, selected + [(w, t)], decision_worlds, cache, objective) - current
            heapq.heappush(heap, (-float(true_gain), int(w), int(t), len(selected)))
    if len(selected) != int(budget_k):
        raise ValueError(f'{method_name} selected {len(selected)} pairs for budget {budget_k}; benchmark requires full budget.')
    return SelectionResult(method_name, selected, time.perf_counter() - start, {
        'top_m': int(top_m),
        'u_max': int(u_max),
        'decision_worlds': list(decision_worlds),
        'decision_objective': str(objective),
        'candidate_pool': candidate_pool,
        'celf_limit': int(effective_limit),
        'candidate_unique_workers': int(unique_workers),
        'selected_count': int(len(selected)),
        'budget_filled': bool(len(selected) == int(budget_k)),
    })


def _evaluate_seed_pairs(evaluator: GKDEvaluator, seed_pairs: list[SeedPair], test_world: int, cache: LiveEdgeWorldCache) -> dict[str, float]:
    return evaluator.evaluate_with_worlds(seed_pairs, [int(test_world)], cache=cache)


def _extended_eval_fn(evaluator: GKDEvaluator, decision_worlds: Sequence[int], cache: LiveEdgeWorldCache):
    def evaluate(seed_pairs: List[SeedPair]) -> float:
        return float(evaluator.evaluate_with_worlds(seed_pairs, decision_worlds, cache=cache)["Effective_Task_Satisfaction"])
    return evaluate


def _pairwise_summary(anchor_rows, other_rows, anchor_result: BenchmarkPolicyResult, other_result: BenchmarkPolicyResult, seed: int):
    anchor = {int(r['test_world']): float(r['ets']) for r in anchor_rows}
    other = {int(r['test_world']): float(r['ets']) for r in other_rows}
    common = sorted(set(anchor) & set(other))
    diffs = np.asarray([anchor[w] - other[w] for w in common], dtype=float)
    lo, hi = _bootstrap_ci(diffs.tolist(), seed=seed)
    anchor_actions = [entry['action_id'] for entry in anchor_result.decision_log]
    other_actions = [entry['action_id'] for entry in other_result.decision_log]
    agree = float(np.mean([int(a == b) for a, b in zip(anchor_actions, other_actions)])) if anchor_actions and other_actions else float('nan')
    anchor_regret = float(np.mean([entry.get('oracle_regret', 0.0) for entry in anchor_result.decision_log])) if anchor_result.decision_log else float('nan')
    other_regret = float(np.mean([entry.get('oracle_regret', 0.0) for entry in other_result.decision_log])) if other_result.decision_log else float('nan')
    rel = diffs / np.maximum(np.asarray([other[w] for w in common], dtype=float), 1e-8)
    return {
        'mean_diff': float(np.mean(diffs)),
        'relative_uplift': float(np.mean(rel)),
        'win_rate': float(np.mean(diffs > 0.0)),
        'win_count': int(np.sum(diffs > 0.0)),
        'tie_rate': float(np.mean(np.isclose(diffs, 0.0))),
        'loss_rate': float(np.mean(diffs < 0.0)),
        'ci95': [lo, hi],
        'action_agreement_rate': agree,
        'anchor_mean_oracle_regret': anchor_regret,
        'baseline_mean_oracle_regret': other_regret,
        'anchor_runtime_ratio': float(anchor_result.selection_time_sec / max(other_result.selection_time_sec, 1e-8)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--budgets', nargs='+', type=int, default=None)
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--test-world-start', type=int, default=None)
    parser.add_argument('--test-world-end', type=int, default=None)
    parser.add_argument('--decision-world-start', type=int, default=None)
    parser.add_argument('--decision-world-end', type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    exp = cfg['experiment']
    worlds = cfg['worlds']
    candidates = cfg['candidates']
    evaluation = cfg['evaluation']
    q_network = str(exp.get('q_network', 'marginal'))
    policy = str(exp.get('policy', 'q'))
    baseline_checkpoints = cfg.get('baseline_checkpoints', {})
    methods = list(args.methods if args.methods is not None else evaluation['methods'])
    budgets = [int(v) for v in (args.budgets if args.budgets is not None else cfg['budget']['eval_values'])]
    decision_worlds = _expand_seed_range(worlds['decision'])
    test_worlds = _expand_seed_range(worlds['test'])
    if args.decision_world_start is not None or args.decision_world_end is not None:
        if args.decision_world_start is None or args.decision_world_end is None:
            raise ValueError('decision world override requires both --decision-world-start and --decision-world-end')
        decision_worlds = list(range(int(args.decision_world_start), int(args.decision_world_end) + 1))
    if args.test_world_start is not None or args.test_world_end is not None:
        if args.test_world_start is None or args.test_world_end is None:
            raise ValueError('test world override requires both --test-world-start and --test-world-end')
        test_worlds = list(range(int(args.test_world_start), int(args.test_world_end) + 1))

    env_dir = Path(exp['env_dir'])
    pretrain_dir = Path(exp['pretrain_dir'])
    checkpoint = Path(exp['checkpoint'])
    graph = load_graph(env_dir)
    q_workers, a_workers, demands, worker_indices, full_q, full_a = _load_env_arrays(env_dir)
    evaluator = GKDEvaluator(graph, q_workers, a_workers, demands, worker_indices, num_simulations=1, full_q_matrix=full_q, full_a_matrix=full_a, seed=int(exp['seed']))

    report_dir = Path(evaluation['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_out = Path(evaluation['csv_out'])
    json_out = Path(evaluation['json_out'])
    md_out = Path(evaluation['md_out'])
    rows = []
    decision_cache = LiveEdgeWorldCache()
    test_cache = LiveEdgeWorldCache()
    eval_fn = _extended_eval_fn(evaluator, decision_worlds, decision_cache)

    for budget_k in budgets:
        method_results = {}
        for method in methods:
            if method == 'A3_marginal_q':
                result = rollout_gkd(env_dir, pretrain_dir, checkpoint, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), 1, int(exp['seed']), 1, False, 'marginal', 'q', bool(candidates['dynamic_candidates']), int(candidates['wide_candidate_size']), int(candidates['shortlist_size']), float(candidates['residual_threshold']), None, decision_worlds)
                method_results[method] = BenchmarkPolicyResult(method, result['seed_pairs'], float(result['selection_time_sec']), float(result['candidate_build_time_sec']), float(result['feature_build_time_sec']), float(result['q_scoring_time_sec']), float(result['state_update_time_sec']), float(result['policy_time_sec']), result['params'], result['decision_log'])
            elif method == 'A4_RL':
                result = rollout_gkd(env_dir, pretrain_dir, checkpoint, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), 1, int(exp['seed']), 1, False, q_network, policy, bool(candidates['dynamic_candidates']), int(candidates['wide_candidate_size']), int(candidates['shortlist_size']), float(candidates['residual_threshold']), None, decision_worlds)
                method_results[method] = BenchmarkPolicyResult(method, result['seed_pairs'], float(result['selection_time_sec']), float(result['candidate_build_time_sec']), float(result['feature_build_time_sec']), float(result['q_scoring_time_sec']), float(result['state_update_time_sec']), float(result['policy_time_sec']), result['params'], result['decision_log'])
            elif method == 'A1_dynamic_heuristic':
                result = rollout_gkd(env_dir, pretrain_dir, checkpoint, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), 1, int(exp['seed']), 1, False, 'marginal', 'heuristic', bool(candidates['dynamic_candidates']), int(candidates['wide_candidate_size']), int(candidates['shortlist_size']), float(candidates['residual_threshold']), None, decision_worlds)
                method_results[method] = BenchmarkPolicyResult(method, result['seed_pairs'], float(result['selection_time_sec']), float(result['candidate_build_time_sec']), float(result['feature_build_time_sec']), float(result['q_scoring_time_sec']), float(result['state_update_time_sec']), float(result['policy_time_sec']), result['params'], result['decision_log'])
            elif method in {'DegGreedy', 'NDD', 'FastSelector', 'ComGreedy', 'TSIM'}:
                baseline_name = 'TSIM-lite' if method == 'TSIM' else method
                sel = run_extended_baseline(
                    baseline_name,
                    graph,
                    q_workers,
                    a_workers,
                    full_q,
                    full_a,
                    worker_indices,
                    int(budget_k),
                    int(candidates['top_m']),
                    int(candidates['u_max']),
                    int(exp['seed']),
                    evaluate_fn=eval_fn,
                    celf_limit=int(evaluation.get('celf_limit', 600)),
                    fast_alpha=float(evaluation.get('fast_alpha', 0.5)),
                    tsim_factor=int(evaluation.get('tsim_factor', 8)),
                )
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'CELF-ETS':
                sel = _celf_static(q_workers, a_workers, worker_indices, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), evaluator, decision_worlds, decision_cache, evaluation.get('celf_limit'), method_name='CELF-ETS', objective='Effective_Task_Satisfaction')
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'CELF-IS':
                sel = _celf_static(q_workers, a_workers, worker_indices, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), evaluator, decision_worlds, decision_cache, evaluation.get('celf_limit'), method_name='CELF-IS', objective='Expected_Influence_Spread')
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'CELF-IS-full':
                sel = _celf_static(q_workers, a_workers, worker_indices, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), evaluator, decision_worlds, decision_cache, evaluation.get('celf_limit'), method_name='CELF-IS-full', objective='Expected_Influence_Spread', use_full_candidate_pool=True)
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'DQNSelector':
                ckpt = baseline_checkpoints.get('dqn_selector')
                if not ckpt:
                    raise ValueError('config missing baseline_checkpoints.dqn_selector for DQNSelector benchmark')
                sel = rollout_dqn_selector_baseline(env_dir, ckpt, int(budget_k), int(candidates['u_max']), seed=int(exp['seed']))
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'MAIM':
                ckpt = baseline_checkpoints.get('maim')
                if not ckpt:
                    raise ValueError('config missing baseline_checkpoints.maim for MAIM benchmark')
                sel = rollout_maim_baseline(env_dir, ckpt, int(budget_k), int(candidates['u_max']), seed=int(exp['seed']))
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'QualityGreedy':
                sel = _quality_greedy_candidates(q_workers, a_workers, worker_indices, int(budget_k), int(candidates['top_m']), int(candidates['u_max']))
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            elif method == 'Random':
                sel = _random_candidates(q_workers, a_workers, worker_indices, int(budget_k), int(candidates['top_m']), int(candidates['u_max']), int(exp['seed']))
                method_results[method] = BenchmarkPolicyResult(method, sel.seed_pairs, sel.selection_time_sec, math.nan, math.nan, math.nan, math.nan, math.nan, sel.params, [])
            else:
                raise ValueError(f'unsupported method {method}')
        for test_world in test_worlds:
            for method, result in method_results.items():
                metrics = _evaluate_seed_pairs(evaluator, result.seed_pairs, int(test_world), test_cache)
                rows.append({'method': method, 'budget_k': int(budget_k), 'test_world': int(test_world), 'decision_worlds': json.dumps(decision_worlds), 'ets': float(metrics['Effective_Task_Satisfaction']), 'spread': float(metrics['Expected_Influence_Spread']), 'selection_time_sec': float(result.selection_time_sec), 'candidate_build_time_sec': float(result.candidate_build_time_sec), 'feature_build_time_sec': float(result.feature_build_time_sec), 'q_scoring_time_sec': float(result.q_scoring_time_sec), 'state_update_time_sec': float(result.state_update_time_sec), 'policy_time_sec': float(result.policy_time_sec), 'seed_pairs': json.dumps(result.seed_pairs), 'params': json.dumps(result.params, ensure_ascii=False, sort_keys=True)})

    with csv_out.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {'config': args.config, 'decision_worlds': decision_worlds, 'test_worlds': test_worlds, 'budgets': budgets, 'methods': methods, 'rows': rows, 'aggregates': {}, 'pairwise': {}}
    for budget_k in budgets:
        budget_rows = [r for r in rows if int(r['budget_k']) == int(budget_k)]
        summary['aggregates'][str(budget_k)] = {}
        for method in methods:
            vals = [float(r['ets']) for r in budget_rows if r['method'] == method]
            times = [float(r['selection_time_sec']) for r in budget_rows if r['method'] == method]
            lo, hi = _bootstrap_ci(vals, seed=int(exp['seed']) + int(budget_k))
            summary['aggregates'][str(budget_k)][method] = {'mean_ets': float(np.mean(vals)), 'std_ets': float(np.std(vals)), 'ci95': [lo, hi], 'mean_selection_time_sec': float(np.mean(times)), 'p95_selection_time_sec': float(np.quantile(times, 0.95))}
        if 'A3_marginal_q' in methods and 'A1_dynamic_heuristic' in methods:
            a3_rows = [r for r in budget_rows if r['method'] == 'A3_marginal_q']
            a1_rows = [r for r in budget_rows if r['method'] == 'A1_dynamic_heuristic']
            summary['pairwise'][str(budget_k)] = {'A1_dynamic_heuristic': _pairwise_summary(a3_rows, a1_rows, method_results['A3_marginal_q'], method_results['A1_dynamic_heuristic'], int(exp['seed']) + 1000 + int(budget_k))}

    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Paired A1 vs A3 benchmark', '']
    for budget_k in budgets:
        lines += [f'## K={budget_k}', '', '| Method | Mean ETS | 95% CI | Mean selection time (s) | P95 selection time (s) |', '|---|---:|---:|---:|---:|']
        for method in methods:
            agg = summary['aggregates'][str(budget_k)][method]
            lines.append(f"| {method} | {agg['mean_ets']:.6f} | [{agg['ci95'][0]:.6f}, {agg['ci95'][1]:.6f}] | {agg['mean_selection_time_sec']:.6f} | {agg['p95_selection_time_sec']:.6f} |")
        if str(budget_k) in summary['pairwise']:
            pair = summary['pairwise'][str(budget_k)]['A1_dynamic_heuristic']
            lines += ['', '### A3 vs A1 paired delta', '', '| Baseline | Mean ETS diff | Relative uplift | Wins | Win rate | 95% CI | Agreement | Runtime ratio |', '|---|---:|---:|---:|---:|---:|---:|---:|', f"| A1_dynamic_heuristic | {pair['mean_diff']:.6f} | {pair['relative_uplift']:.3%} | {pair['win_count']} | {pair['win_rate']:.3f} | [{pair['ci95'][0]:.6f}, {pair['ci95'][1]:.6f}] | {pair['action_agreement_rate']:.3f} | {pair['anchor_runtime_ratio']:.3f} |", '', f"A3 mean oracle regret: {pair['anchor_mean_oracle_regret']:.6f}", f"A1 mean oracle regret: {pair['baseline_mean_oracle_regret']:.6f}", '']
    md_out.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved CSV to {csv_out}')
    print(f'Saved JSON to {json_out}')
    print(f'Saved Markdown to {md_out}')


if __name__ == '__main__':
    main()
