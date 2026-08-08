"""Train the state-aware Double Dueling DQN selector.

Checkpoints are selected with independent fixed-seed validation rollouts,
not with the final state of a noisy training episode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.action_features import GLOBAL_STATE_NAMES, LOCAL_FEATURE_NAMES, build_dynamic_action_features, summarize_global_state
from models.candidates import build_dynamic_shortlist, build_topm_actions, compute_selected_worker_overlap
from models.gkd_env import GKDEnv
from models.evaluate import LiveEdgeWorldCache
from models.gkd_recruiter import StateAwareDuelingQNetwork
from models.marginal_q_network import CandidateAwareDuelingQNetwork, MarginalQNetwork
from models.runtime import configure_runtime, maybe_compile


@dataclass
class Transition:
    state: np.ndarray
    action_ids: np.ndarray
    pair_features: np.ndarray
    local_features: np.ndarray
    chosen_action_id: int
    reward: float
    next_state: np.ndarray
    next_action_ids: np.ndarray
    next_pair_features: np.ndarray
    next_local_features: np.ndarray
    done: bool


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = max(int(n_step), 1)
        self.gamma = float(gamma)
        self.buffer = deque()

    def append(self, transition: Transition) -> list[Transition]:
        self.buffer.append(transition)
        emitted: list[Transition] = []
        if transition.done:
            while self.buffer:
                emitted.append(self._aggregate_prefix(len(self.buffer)))
                self.buffer.popleft()
            return emitted
        if len(self.buffer) >= self.n_step:
            emitted.append(self._aggregate_prefix(self.n_step))
            self.buffer.popleft()
        return emitted

    def flush(self) -> list[Transition]:
        emitted: list[Transition] = []
        while self.buffer:
            emitted.append(self._aggregate_prefix(len(self.buffer)))
            self.buffer.popleft()
        return emitted

    def _aggregate_prefix(self, horizon: int) -> Transition:
        items = list(self.buffer)[: int(horizon)]
        total_reward = 0.0
        for step, item in enumerate(items):
            total_reward += (self.gamma ** step) * float(item.reward)
            if item.done:
                break
        first = items[0]
        last = items[-1]
        return Transition(
            state=first.state,
            action_ids=first.action_ids,
            pair_features=first.pair_features,
            local_features=first.local_features,
            chosen_action_id=first.chosen_action_id,
            reward=float(total_reward),
            next_state=last.next_state,
            next_action_ids=last.next_action_ids,
            next_pair_features=last.next_pair_features,
            next_local_features=last.next_local_features,
            done=bool(last.done),
        )


class PER:
    """Small proportional prioritized replay buffer."""

    def __init__(self, n: int = 100000, alpha: float = 0.6):
        self.n = int(n)
        self.a = float(alpha)
        self.data = []
        self.p = []
        self.i = 0

    def push(self, x):
        pr = max(self.p, default=1.0)
        if len(self.data) < self.n:
            self.data.append(x)
            self.p.append(pr)
        else:
            self.data[self.i] = x
            self.p[self.i] = pr
            self.i = (self.i + 1) % self.n

    def sample(self, b: int, beta: float = 0.4):
        if len(self.data) < int(b):
            raise ValueError("cannot sample more transitions than replay size")
        prob = np.asarray(self.p, dtype=float) ** self.a
        prob /= prob.sum()
        ids = np.random.choice(len(self.data), int(b), p=prob)
        weights = (len(self.data) * prob[ids]) ** (-float(beta))
        weights /= weights.max()
        return ids, [self.data[i] for i in ids], torch.tensor(weights, dtype=torch.float32)

    def update(self, ids, td):
        for i, value in zip(ids, td):
            self.p[int(i)] = float(abs(value)) + 1e-5

    @property
    def mean_priority(self) -> float:
        return float(np.mean(self.p)) if self.p else 0.0

    def __len__(self):
        return len(self.data)


def linear_epsilon(step: int, start: float = 1.0, end: float = 0.05,
                   decay_steps: int = 50000) -> float:
    if decay_steps <= 0:
        return float(end)
    ratio = min(max(float(step), 0.0) / float(decay_steps), 1.0)
    if ratio >= 1.0:
        return float(end)
    return float(start + ratio * (end - start))


def _expand_seed_range(bounds: Iterable[int] | None) -> list[int]:
    if bounds is None:
        return []
    values = [int(v) for v in bounds]
    if not values:
        return []
    if len(values) == 2:
        start, end = values
        if end < start:
            raise ValueError(f"invalid world range {values}")
        return list(range(start, end + 1))
    return values


def _candidate_mask(env: GKDEnv, action_ids: np.ndarray) -> np.ndarray:
    return np.asarray(env.valid_action_mask(action_ids), dtype=bool)[action_ids]


def _global_to_local_index(action_ids: np.ndarray, chosen_action_id: int) -> int:
    matches = np.flatnonzero(np.asarray(action_ids, dtype=np.int64) == int(chosen_action_id))
    if len(matches) != 1:
        raise ValueError(f"chosen action {chosen_action_id} not found exactly once in candidate set")
    return int(matches[0])


def _collate_variable_action_batch(batch: list[Transition], device: torch.device):
    if not batch:
        raise ValueError("batch must be non-empty")
    max_actions = max(len(item.action_ids) for item in batch)
    max_next_actions = max(len(item.next_action_ids) for item in batch)
    pair_dim = batch[0].pair_features.shape[1] if batch[0].pair_features.size else 0
    local_dim = batch[0].local_features.shape[1] if batch[0].local_features.size else 0
    next_pair_dim = batch[0].next_pair_features.shape[1] if batch[0].next_pair_features.size else pair_dim
    next_local_dim = batch[0].next_local_features.shape[1] if batch[0].next_local_features.size else local_dim

    states = torch.tensor(np.asarray([item.state for item in batch]), dtype=torch.float32, device=device)
    next_states = torch.tensor(np.asarray([item.next_state for item in batch]), dtype=torch.float32, device=device)
    rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=device)
    dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=device)

    action_ids = torch.full((len(batch), max_actions), -1, dtype=torch.long, device=device)
    next_action_ids = torch.full((len(batch), max_next_actions), -1, dtype=torch.long, device=device)
    pair_features = torch.zeros((len(batch), max_actions, pair_dim), dtype=torch.float32, device=device)
    next_pair_features = torch.zeros((len(batch), max_next_actions, next_pair_dim), dtype=torch.float32, device=device)
    local_features = torch.zeros((len(batch), max_actions, local_dim), dtype=torch.float32, device=device)
    next_local_features = torch.zeros((len(batch), max_next_actions, next_local_dim), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((len(batch), max_actions), dtype=torch.bool, device=device)
    next_valid_mask = torch.zeros((len(batch), max_next_actions), dtype=torch.bool, device=device)
    chosen_local = torch.zeros(len(batch), dtype=torch.long, device=device)

    for idx, item in enumerate(batch):
        cur_len = len(item.action_ids)
        next_len = len(item.next_action_ids)
        action_ids[idx, :cur_len] = torch.tensor(item.action_ids, dtype=torch.long, device=device)
        next_action_ids[idx, :next_len] = torch.tensor(item.next_action_ids, dtype=torch.long, device=device)
        pair_features[idx, :cur_len] = torch.tensor(item.pair_features, dtype=torch.float32, device=device)
        local_features[idx, :cur_len] = torch.tensor(item.local_features, dtype=torch.float32, device=device)
        next_pair_features[idx, :next_len] = torch.tensor(item.next_pair_features, dtype=torch.float32, device=device)
        next_local_features[idx, :next_len] = torch.tensor(item.next_local_features, dtype=torch.float32, device=device)
        valid_mask[idx, :cur_len] = True
        next_valid_mask[idx, :next_len] = True
        chosen_local[idx] = _global_to_local_index(item.action_ids, item.chosen_action_id)

    return {
        "states": states,
        "next_states": next_states,
        "rewards": rewards,
        "dones": dones,
        "action_ids": action_ids,
        "next_action_ids": next_action_ids,
        "pair_features": pair_features,
        "next_pair_features": next_pair_features,
        "local_features": local_features,
        "next_local_features": next_local_features,
        "valid_mask": valid_mask,
        "next_valid_mask": next_valid_mask,
        "chosen_local": chosen_local,
    }


def _worker_influence_proxy(env_dir: str, num_workers: int) -> np.ndarray:
    edge_index = np.atleast_2d(np.loadtxt(os.path.join(env_dir, "edge_index.txt"), dtype=int))
    worker_indices = np.atleast_1d(np.loadtxt(os.path.join(env_dir, "worker_indices.txt"), dtype=int))
    degree = np.zeros(len(worker_indices), dtype=np.float32)
    worker_to_local = {int(worker): idx for idx, worker in enumerate(worker_indices.tolist())}
    for src, dst in edge_index:
        if int(src) in worker_to_local:
            degree[worker_to_local[int(src)]] += 1.0
        if int(dst) in worker_to_local:
            degree[worker_to_local[int(dst)]] += 1.0
    if len(degree) != num_workers:
        raise ValueError("worker influence proxy size mismatch")
    return degree


def _candidate_pack(
    env: GKDEnv,
    q_workers: np.ndarray,
    a_workers: np.ndarray,
    demands: np.ndarray,
    worker_embeds: torch.Tensor,
    task_embeds: torch.Tensor,
    top_m: int,
    dynamic_candidates: bool,
    wide_candidate_size: int,
    shortlist_size: int,
    residual_threshold: float,
    worker_influence_proxy: np.ndarray,
):
    task_selected_count = np.bincount([task for _, task in env.selected_seeds], minlength=env.num_tasks).astype(np.float32)
    task_selected_workers: list[set[int]] = [set() for _ in range(env.num_tasks)]
    worker_to_local = {int(worker): idx for idx, worker in enumerate(env.worker_indices.tolist())}
    selected_mask = np.zeros((env.num_tasks, env.num_workers), dtype=np.float32)
    for worker_id, task_idx in env.selected_seeds:
        local = worker_to_local.get(int(worker_id))
        if local is not None:
            task_selected_workers[int(task_idx)].add(int(local))
            selected_mask[int(task_idx), int(local)] = 1.0

    if dynamic_candidates:
        base_actions, base_ids, _ = build_dynamic_shortlist(
            q_workers=q_workers,
            a_workers=a_workers,
            task_demands=demands,
            worker_load=env.worker_load,
            worker_capacity=np.full(env.num_workers, env.u_max, dtype=np.float32),
            task_ets=env.task_ets,
            task_selected_count=task_selected_count,
            current_step=env.current_step,
            budget_k=env.budget_K,
            top_m_static=top_m,
            wide_candidate_size=wide_candidate_size,
            shortlist_size=shortlist_size,
            residual_threshold=residual_threshold,
            worker_influence_proxy=worker_influence_proxy,
            pair_overlap_proxy=selected_mask,
            worker_embeddings=worker_embeds.detach().cpu().numpy(),
        )
        overlap = compute_selected_worker_overlap(worker_embeds.detach().cpu().numpy(), base_actions, selected_mask, reduction='max')
        local_features_np = build_dynamic_action_features(
            q_matrix=q_workers,
            a_matrix=a_workers,
            task_demands=demands,
            worker_load=env.worker_load,
            worker_capacity=np.full(env.num_workers, env.u_max, dtype=np.float32),
            task_ets=env.task_ets,
            task_selected_count=task_selected_count,
            current_step=env.current_step,
            budget_k=env.budget_K,
            pair_actions=base_actions,
            worker_influence_proxy=worker_influence_proxy,
            pair_overlap_proxy=overlap,
        )
        actions_np, action_ids = base_actions, base_ids
    else:
        actions_np, action_ids = build_topm_actions(q_workers, a_workers, top_m, demands, False)
        overlap = compute_selected_worker_overlap(worker_embeds.detach().cpu().numpy(), actions_np, selected_mask, reduction='max')
        local_features_np = build_dynamic_action_features(
            q_matrix=q_workers,
            a_matrix=a_workers,
            task_demands=demands,
            worker_load=env.worker_load,
            worker_capacity=np.full(env.num_workers, env.u_max, dtype=np.float32),
            task_ets=env.task_ets,
            task_selected_count=task_selected_count,
            current_step=env.current_step,
            budget_k=env.budget_K,
            pair_actions=actions_np,
            worker_influence_proxy=worker_influence_proxy,
            pair_overlap_proxy=overlap,
        )
    state_np = summarize_global_state(env.current_ets, env.current_step, env.budget_K, env.task_ets, env.worker_load, np.full(env.num_workers, env.u_max, dtype=np.float32))
    pair_features = torch.cat([worker_embeds[actions_np[:, 0]], task_embeds[actions_np[:, 1]]], dim=-1)
    return (
        actions_np,
        action_ids,
        torch.tensor(state_np, dtype=torch.float32),
        torch.tensor(local_features_np, dtype=torch.float32),
        pair_features,
    )


@torch.no_grad()
def select_greedy_action(net, state: torch.Tensor, valid_mask: np.ndarray,
                         actions: torch.Tensor, worker_embeds: torch.Tensor,
                         task_embeds: torch.Tensor, device: torch.device) -> int:
    """Return the local candidate index with the highest masked Q value."""
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=device).unsqueeze(0)
    worker_batch = worker_embeds.unsqueeze(0) if worker_embeds.dim() == 2 else worker_embeds
    task_batch = task_embeds.unsqueeze(0) if task_embeds.dim() == 2 else task_embeds
    q_values = net(
        state.unsqueeze(0).to(device), worker_batch, task_batch, actions,
        valid_mask=mask,
    ).squeeze(0)
    return int(torch.argmax(q_values).item())


def _select_action_values(
    net,
    q_network: str,
    state: torch.Tensor,
    marginal_state: torch.Tensor,
    pair_features: torch.Tensor,
    local_features: torch.Tensor,
    actions: torch.Tensor,
    worker_embeds: torch.Tensor,
    task_embeds: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if q_network in {"marginal", "candidate_aware_dueling"}:
        return net(
            marginal_state.unsqueeze(0).to(device),
            pair_features.unsqueeze(0).to(device),
            local_features.unsqueeze(0).to(device),
            valid_mask=valid_mask,
        ).squeeze(0)
    return net(
        state.unsqueeze(0).to(device),
        worker_embeds.unsqueeze(0),
        task_embeds.unsqueeze(0),
        actions,
        valid_mask=valid_mask,
    ).squeeze(0)


def _select_policy_action(
    policy: str,
    net,
    q_network: str,
    state: torch.Tensor,
    marginal_state: torch.Tensor,
    pair_features: torch.Tensor,
    local_features: torch.Tensor,
    actions: torch.Tensor,
    worker_embeds: torch.Tensor,
    task_embeds: torch.Tensor,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> int:
    if policy == "heuristic":
        heuristic_scores = local_features[:, 2] * local_features[:, 3] * torch.clamp(local_features[:, 5], min=0.0)
        heuristic_scores = heuristic_scores * (0.5 + 0.5 * local_features[:, 10]) * (1.0 - 0.5 * local_features[:, 11])
        heuristic_scores = heuristic_scores.masked_fill(~valid_mask.squeeze(0), torch.finfo(heuristic_scores.dtype).min)
        return int(torch.argmax(heuristic_scores).item())
    q_values = _select_action_values(
        net, q_network, state, marginal_state, pair_features, local_features,
        actions, worker_embeds, task_embeds, valid_mask, device,
    )
    return int(torch.argmax(q_values).item())


@torch.no_grad()
def evaluate_policy(
    net,
    env_factory: Callable[..., GKDEnv],
    validation_seeds: Iterable[int],
    top_m: int,
    q_workers: np.ndarray,
    a_workers: np.ndarray,
    demands: np.ndarray,
    worker_embeds: torch.Tensor,
    task_embeds: torch.Tensor,
    device: torch.device,
    q_network: str = "dueling",
    dynamic_candidates: bool = False,
    wide_candidate_size: int = 2048,
    shortlist_size: int = 256,
    residual_threshold: float = 0.005,
    worker_influence_proxy: np.ndarray | None = None,
    num_simulations: int = 30,
    deterministic: bool = True,
    policy: str = "q",
) -> dict[str, object]:
    """Run reset-to-terminal rollouts on independent fixed validation seeds."""
    was_training = net.training
    noisy_modules = [m for m in net.modules() if hasattr(m, "use_noise")]
    previous_noise = [bool(m.use_noise) for m in noisy_modules]
    net.eval()
    if deterministic and hasattr(net, "disable_noise"):
        net.disable_noise()
    scores: list[float] = []
    try:
        for seed in validation_seeds:
            env = env_factory(seed=int(seed), num_simulations=int(num_simulations))
            state = env.reset()
            done = False
            while not done:
                if hasattr(env, "num_workers") and hasattr(env, "selected_seeds") and hasattr(env, "worker_indices"):
                    actions_np, action_ids, marginal_state, local_features, pair_features = _candidate_pack(
                        env, q_workers, a_workers, demands, worker_embeds.cpu(), task_embeds.cpu(), top_m,
                        dynamic_candidates, wide_candidate_size, shortlist_size, residual_threshold,
                        np.ones(env.num_workers, dtype=np.float32) if worker_influence_proxy is None else worker_influence_proxy,
                    )
                    if len(action_ids) == 0:
                        break
                    valid_mask_np = _candidate_mask(env, action_ids)
                    if not valid_mask_np.any():
                        break
                    actions = torch.tensor(actions_np, dtype=torch.long, device=device)
                    valid_mask = torch.as_tensor(valid_mask_np, dtype=torch.bool, device=device).unsqueeze(0)
                    local_action = _select_policy_action(
                        policy, net, q_network, state, marginal_state, pair_features, local_features,
                        actions, worker_embeds, task_embeds, valid_mask, device,
                    )
                    chosen_action = int(action_ids[local_action])
                else:
                    actions_np, action_ids = build_topm_actions(q_workers, a_workers, top_m, demands, False)
                    valid_mask_np = np.asarray(env.valid_action_mask(action_ids), dtype=bool)
                    if not valid_mask_np.any():
                        break
                    actions = torch.tensor(actions_np, dtype=torch.long, device=device)
                    valid_mask = torch.as_tensor(valid_mask_np, dtype=torch.bool, device=device).unsqueeze(0)
                    q_values = net(
                        state.unsqueeze(0).to(device),
                        worker_embeds.unsqueeze(0),
                        task_embeds.unsqueeze(0),
                        actions,
                        valid_mask=valid_mask,
                    ).squeeze(0)
                    local_action = int(torch.argmax(q_values).item())
                    chosen_action = int(action_ids[local_action])
                state, _, done, _ = env.step(chosen_action)
            scores.append(float(env.current_ets))
    finally:
        for module, enabled in zip(noisy_modules, previous_noise):
            module.use_noise = enabled
        if was_training:
            net.train()
    return {
        "mean_ets": float(np.mean(scores)) if scores else float("nan"),
        "std_ets": float(np.std(scores)) if scores else float("nan"),
        "scores": scores,
    }


def _write_logs(checkpoint_dir: str, history: list[dict], summary: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    if history:
        with open(os.path.join(checkpoint_dir, "training_history.csv"), "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    with open(os.path.join(checkpoint_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2, ensure_ascii=False, allow_nan=False)


def main(
    episodes: int = 500, budget: int = 75, top_m: int = 5, u_max: int = 1,
    seed: int = 42,
    env_dir: str = "data/experiments/gowalla_v3000_seed42/env_params",
    pretrain_dir: str = "data/experiments/gowalla_v3000_seed42/pretrain",
    checkpoint_dir: str = "checkpoints", device_name: str = "auto",
    cpu_threads: int = 0, amp: bool = True, compile_model: bool = False,
    train_simulations: int = 1, incremental: bool = True,
    validation_interval: int = 20, validation_seeds: Iterable[int] = (41, 42, 43),
    validation_simulations: int = 30, deterministic_validation: bool = True,
    patience: int = 15, min_episodes: int = 100, min_delta: float = 0.001,
    replay_capacity: int = 100000, replay_warmup: int = 3000,
    batch_size: int = 128, update_frequency: int = 1,
    learning_rate: float = 3e-4, gamma: float = 1.0,
    n_step: int = 3,
    target_update_interval: int = 1000, reward_scale: float = 1.0,
    epsilon_start: float = 1.0, epsilon_end: float = 0.05,
    epsilon_decay_steps: int = 50000, noisy_training: bool = True,
    q_network: str = "dueling", dynamic_candidates: bool = False,
    wide_candidate_size: int = 2048, shortlist_size: int = 256,
    residual_threshold: float = 0.005,
    init_checkpoint: str | None = None,
    train_worlds: Iterable[int] | None = None,
    validation_worlds: Iterable[int] | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device, amp_enabled, _ = configure_runtime(device_name, cpu_threads, amp, compile_model)

    worker_embeds = torch.load(os.path.join(pretrain_dir, "distilled_worker_embeds.pt"), map_location=device, weights_only=True)
    task_embeds = torch.load(os.path.join(pretrain_dir, "distilled_task_embeds.pt"), map_location=device, weights_only=True)
    q_workers = np.atleast_2d(np.loadtxt(os.path.join(env_dir, "q_matrix.txt"), dtype=float))
    a_workers = np.atleast_2d(np.loadtxt(os.path.join(env_dir, "a_matrix.txt"), dtype=float))
    demands = np.atleast_1d(np.loadtxt(os.path.join(env_dir, "task_demands.txt"), dtype=float))
    worker_influence_proxy = _worker_influence_proxy(env_dir, q_workers.shape[0])
    train_world_bank = _expand_seed_range(train_worlds)
    validation_world_bank = _expand_seed_range(validation_worlds)
    shared_world_cache = LiveEdgeWorldCache()

    def make_env(seed: int, num_simulations: int, live_edge_worlds: Iterable[int] | None = None) -> GKDEnv:
        return GKDEnv(env_dir=env_dir, budget_K=budget, u_max=u_max,
                      num_simulations=num_simulations, seed=seed,
                      incremental=incremental, reward_scale=reward_scale,
                      live_edge_worlds=(tuple(int(world) for world in live_edge_worlds) if live_edge_worlds is not None else None),
                      live_edge_cache=shared_world_cache)

    bootstrap_seed = int(train_world_bank[0]) if train_world_bank else int(seed)
    env = make_env(bootstrap_seed, train_simulations, live_edge_worlds=train_world_bank if train_world_bank else None)
    state_dim = 2 + env.num_tasks + env.num_workers if q_network == "dueling" else len(GLOBAL_STATE_NAMES)
    pair_dim = worker_embeds.shape[1] * 2
    local_dim = len(LOCAL_FEATURE_NAMES)
    if q_network == "marginal":
        net = maybe_compile(MarginalQNetwork(pair_dim, state_dim, local_dim).to(device), compile_model)
        target = MarginalQNetwork(pair_dim, state_dim, local_dim).to(device)
    elif q_network == "candidate_aware_dueling":
        net = maybe_compile(CandidateAwareDuelingQNetwork(pair_dim, state_dim, local_dim).to(device), compile_model)
        target = CandidateAwareDuelingQNetwork(pair_dim, state_dim, local_dim).to(device)
    else:
        net = maybe_compile(StateAwareDuelingQNetwork(worker_embeds.shape[1], state_dim).to(device), compile_model)
        target = StateAwareDuelingQNetwork(worker_embeds.shape[1], state_dim).to(device)
    if init_checkpoint:
        net.load_state_dict(torch.load(init_checkpoint, map_location=device, weights_only=True))
        print(f"loaded_init_checkpoint={init_checkpoint}")
    target.load_state_dict(net.state_dict())
    target.eval()
    if hasattr(target, "disable_noise"):
        target.disable_noise()

    opt = torch.optim.Adam(net.parameters(), learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    mem = PER(replay_capacity)
    n_step_accumulator = NStepAccumulator(n_step=n_step, gamma=gamma)
    optimizer_steps = 0
    global_step = 0
    best_validation_ets = -float("inf")
    stale_validations = 0
    last_validation = {"mean_ets": 0.0, "std_ets": 0.0, "scores": []}
    stopped_early = False
    history: list[dict] = []
    best_path = os.path.join(checkpoint_dir, "seed_selector_best.pt")
    last_path = os.path.join(checkpoint_dir, "seed_selector_last.pt")
    compat_path = os.path.join(checkpoint_dir, "seed_selector.pt")
    training_start = time.perf_counter()
    training_worlds_used: list[int] = []

    for ep in range(int(episodes)):
        episode_start = time.perf_counter()
        net.train()
        if hasattr(net, "set_noise_enabled"):
            net.set_noise_enabled(noisy_training)
        if train_world_bank:
            episode_seed = int(random.choice(train_world_bank))
            training_worlds_used.append(episode_seed)
            env = make_env(episode_seed, train_simulations, live_edge_worlds=train_world_bank if train_world_bank else None)
        else:
            episode_seed = int(seed)
            training_worlds_used.append(episode_seed)
        state = env.reset()
        total_reward = 0.0
        episode_losses: list[float] = []
        q_samples: list[float] = []
        valid_counts: list[int] = []

        for _ in range(int(budget)):
            actions_np, action_ids, marginal_state, local_features, pair_features = _candidate_pack(
                env, q_workers, a_workers, demands, worker_embeds.cpu(), task_embeds.cpu(), top_m,
                dynamic_candidates, wide_candidate_size, shortlist_size, residual_threshold, worker_influence_proxy,
            )
            actions = torch.tensor(actions_np, dtype=torch.long, device=device)
            current_mask = _candidate_mask(env, action_ids)
            valid_local = np.flatnonzero(current_mask)
            if len(valid_local) == 0:
                break
            epsilon = linear_epsilon(global_step, epsilon_start, epsilon_end, epsilon_decay_steps)
            with torch.no_grad():
                mask_tensor = torch.as_tensor(current_mask, dtype=torch.bool, device=device).unsqueeze(0)
                if q_network in {"marginal", "candidate_aware_dueling"}:
                    q_values = net(
                        marginal_state.unsqueeze(0).to(device),
                        pair_features.unsqueeze(0).to(device),
                        local_features.unsqueeze(0).to(device),
                        valid_mask=mask_tensor,
                    ).squeeze(0)
                else:
                    q_values = net(state[None].to(device), worker_embeds.unsqueeze(0), task_embeds.unsqueeze(0), actions, valid_mask=mask_tensor).squeeze(0)
                q_numpy = q_values.detach().cpu().numpy()
            q_samples.append(float(np.mean(q_numpy[valid_local])))
            valid_counts.append(int(len(valid_local)))
            if random.random() < epsilon:
                action_local = int(random.choice(valid_local.tolist()))
            else:
                action_local = int(valid_local[np.argmax(q_numpy[valid_local])])
            chosen_action_id = int(action_ids[action_local])
            next_state, reward, done, _ = env.step(chosen_action_id)
            next_actions_np, next_action_ids, next_marginal_state, next_local_features, next_pair_features = _candidate_pack(
                env, q_workers, a_workers, demands, worker_embeds.cpu(), task_embeds.cpu(), top_m,
                dynamic_candidates, wide_candidate_size, shortlist_size, residual_threshold, worker_influence_proxy,
            )
            del next_actions_np
            transition = Transition(
                state=(marginal_state if q_network in {"marginal", "candidate_aware_dueling"} else state).numpy(),
                action_ids=np.asarray(action_ids, dtype=np.int64),
                pair_features=pair_features.detach().cpu().numpy().astype(np.float32),
                local_features=local_features.detach().cpu().numpy().astype(np.float32),
                chosen_action_id=chosen_action_id,
                reward=float(reward),
                next_state=(next_marginal_state if q_network in {"marginal", "candidate_aware_dueling"} else next_state).numpy(),
                next_action_ids=np.asarray(next_action_ids, dtype=np.int64),
                next_pair_features=next_pair_features.detach().cpu().numpy().astype(np.float32),
                next_local_features=next_local_features.detach().cpu().numpy().astype(np.float32),
                done=done,
            )
            for aggregated in n_step_accumulator.append(transition):
                mem.push(aggregated)
            state = next_state
            total_reward += float(reward)
            global_step += 1

            if len(mem) >= max(int(replay_warmup), int(batch_size)) and global_step % max(int(update_frequency), 1) == 0:
                ids, items, weights = mem.sample(batch_size, beta=min(1.0, 0.4 + ep / max(int(episodes), 1) * 0.6))
                batch = _collate_variable_action_batch(items, device)
                weights = weights.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    if q_network in {"marginal", "candidate_aware_dueling"}:
                        cur = net(
                            batch["states"],
                            batch["pair_features"],
                            batch["local_features"],
                            valid_mask=batch["valid_mask"],
                        ).gather(1, batch["chosen_local"][:, None]).squeeze(1)
                    else:
                        cur = net(batch["states"], None, None, batch["action_ids"], pair_features=batch["pair_features"], valid_mask=batch["valid_mask"]).gather(1, batch["chosen_local"][:, None]).squeeze(1)
                with torch.no_grad():
                    if q_network in {"marginal", "candidate_aware_dueling"}:
                        next_q = net(
                            batch["next_states"],
                            batch["next_pair_features"],
                            batch["next_local_features"],
                            valid_mask=batch["next_valid_mask"],
                        )
                    else:
                        next_q = net(batch["next_states"], None, None, batch["next_action_ids"], pair_features=batch["next_pair_features"], valid_mask=batch["next_valid_mask"])
                    next_actions = next_q.argmax(1)
                    if hasattr(target, "disable_noise"):
                        target.disable_noise()
                    if q_network in {"marginal", "candidate_aware_dueling"}:
                        target_q = target(
                            batch["next_states"],
                            batch["next_pair_features"],
                            batch["next_local_features"],
                            valid_mask=batch["next_valid_mask"],
                        ).gather(1, next_actions[:, None]).squeeze(1)
                    else:
                        target_q = target(batch["next_states"], None, None, batch["next_action_ids"], pair_features=batch["next_pair_features"], valid_mask=batch["next_valid_mask"]).gather(1, next_actions[:, None]).squeeze(1)
                    y = batch["rewards"] + gamma * target_q * (1 - batch["dones"])
                td = y - cur
                loss = (weights * F.smooth_l1_loss(cur, y, reduction="none")).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                mem.update(ids, td.detach().cpu().numpy())
                episode_losses.append(float(loss.item()))
                optimizer_steps += 1
                if optimizer_steps % max(int(target_update_interval), 1) == 0:
                    target.load_state_dict(net.state_dict())
                    target.eval()
                    if hasattr(target, "disable_noise"):
                        target.disable_noise()
            if done:
                break

        for aggregated in n_step_accumulator.flush():
            mem.push(aggregated)

        validation_improved = False
        validation_mean = float("nan")
        validation_std = float("nan")
        if (ep + 1) % max(int(validation_interval), 1) == 0:
            validation_env_factory = lambda seed, num_simulations: make_env(
                seed, num_simulations, live_edge_worlds=validation_world_bank if validation_world_bank else None,
            )
            last_validation = evaluate_policy(
                net, validation_env_factory, tuple(int(s) for s in validation_seeds), top_m,
                q_workers, a_workers, demands, worker_embeds, task_embeds, device,
                q_network=q_network, dynamic_candidates=dynamic_candidates,
                wide_candidate_size=wide_candidate_size, shortlist_size=shortlist_size,
                residual_threshold=residual_threshold, worker_influence_proxy=worker_influence_proxy,
                 num_simulations=validation_simulations, deterministic=deterministic_validation,
            )
            validation_mean = float(last_validation["mean_ets"])
            validation_std = float(last_validation["std_ets"])
            validation_improved = validation_mean > best_validation_ets + float(min_delta)
            if validation_improved:
                best_validation_ets = validation_mean
                stale_validations = 0
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(net.state_dict(), best_path)
            else:
                stale_validations += 1
            print(f"episode={ep + 1} train_ets={env.current_ets:.5f} reward={total_reward:.5f} "
                  f"validation={validation_mean:.5f}±{validation_std:.5f} "
                  f"best_val={best_validation_ets:.5f} stale={stale_validations}/{patience}")
            if ep + 1 >= int(min_episodes) and stale_validations >= int(patience):
                stopped_early = True

        history.append({
            "episode": ep + 1, "episode_reward": float(total_reward),
            "train_final_ets": float(env.current_ets),
            "train_world_seed": int(episode_seed),
            "validation_mean_ets": validation_mean, "validation_std_ets": validation_std,
            "best_validation_ets": float(best_validation_ets),
            "validation_improved": int(validation_improved),
            "stale_validations": int(stale_validations),
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else float("nan"),
            "q_mean": float(np.mean(q_samples)) if q_samples else float("nan"),
            "q_std": float(np.std(q_samples)) if q_samples else float("nan"),
            "mean_priority": mem.mean_priority,
            "mean_valid_action_count": float(np.mean(valid_counts)) if valid_counts else 0.0,
            "epsilon": linear_epsilon(global_step, epsilon_start, epsilon_end, epsilon_decay_steps),
            "global_step": int(global_step), "optimizer_steps": int(optimizer_steps),
            "replay_size": len(mem), "episode_time_sec": float(time.perf_counter() - episode_start),
            "stopped_early": int(stopped_early),
        })
        if stopped_early:
            break

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(net.state_dict(), last_path)
    torch.save(net.state_dict(), compat_path)
    if best_validation_ets == -float("inf"):
        best_validation_ets = float(env.current_ets)
        torch.save(net.state_dict(), best_path)
    summary = {
        "config": {"seed": seed, "episodes": episodes, "budget": budget, "top_m": top_m,
                   "u_max": u_max, "train_simulations": train_simulations,
                   "validation_seeds": [int(s) for s in validation_seeds],
                   "validation_simulations": validation_simulations, "reward_scale": reward_scale,
                   "n_step": n_step, "gamma": gamma,
                    "q_network": q_network, "dynamic_candidates": dynamic_candidates,
                     "train_worlds": train_world_bank,
                    "validation_worlds": validation_world_bank,
                   "wide_candidate_size": wide_candidate_size, "shortlist_size": shortlist_size,
                   "residual_threshold": residual_threshold,
                   "replay_capacity": replay_capacity, "replay_warmup": replay_warmup,
                   "batch_size": batch_size, "epsilon_start": epsilon_start,
                   "epsilon_end": epsilon_end, "epsilon_decay_steps": epsilon_decay_steps},
        "best_validation_ets": float(best_validation_ets), "last_validation": last_validation,
        "episodes_completed": len(history), "global_steps": global_step,
        "training_worlds_used": training_worlds_used,
        "optimizer_steps": optimizer_steps, "stopped_early": stopped_early,
        "training_time_sec": float(time.perf_counter() - training_start), "history": history,
    }
    _write_logs(checkpoint_dir, history, summary)
    print(f"training_complete episodes={len(history)} stopped_early={stopped_early} best_validation_ets={best_validation_ets:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500); parser.add_argument("--budget", type=int, default=75)
    parser.add_argument("--top-m", type=int, default=5); parser.add_argument("--u-max", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--env-dir", default="data/experiments/gowalla_v3000_seed42/env_params")
    parser.add_argument("--pretrain-dir", default="data/experiments/gowalla_v3000_seed42/pretrain"); parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--device", default="auto"); parser.add_argument("--cpu-threads", type=int, default=0); parser.add_argument("--no-amp", action="store_true"); parser.add_argument("--compile", action="store_true")
    parser.add_argument("--train-simulations", type=int, default=1); parser.add_argument("--exact-step-evaluation", action="store_true")
    parser.add_argument("--validation-interval", type=int, default=20); parser.add_argument("--validation-seeds", nargs="+", type=int, default=[41, 42, 43]); parser.add_argument("--validation-simulations", type=int, default=30); parser.add_argument("--non-deterministic-validation", action="store_true")
    parser.add_argument("--patience", type=int, default=15); parser.add_argument("--min-episodes", type=int, default=100); parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--replay-capacity", type=int, default=100000); parser.add_argument("--replay-warmup", type=int, default=3000); parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--update-frequency", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--gamma", type=float, default=1.0); parser.add_argument("--n-step", type=int, default=3); parser.add_argument("--target-update-interval", type=int, default=1000); parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0); parser.add_argument("--epsilon-end", type=float, default=0.05); parser.add_argument("--epsilon-decay-steps", type=int, default=50000)
    parser.add_argument("--noisy-training", dest="noisy_training", action="store_true", default=True); parser.add_argument("--no-noisy-training", dest="noisy_training", action="store_false"); parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--q-network", choices=["dueling", "marginal", "candidate_aware_dueling"], default="dueling")
    parser.add_argument("--dynamic-candidates", action="store_true")
    parser.add_argument("--wide-candidate-size", type=int, default=2048)
    parser.add_argument("--shortlist-size", type=int, default=256)
    parser.add_argument("--residual-threshold", type=float, default=0.005)
    parser.add_argument("--train-worlds", nargs="+", type=int, default=None)
    parser.add_argument("--validation-worlds", nargs="+", type=int, default=None)
    args = parser.parse_args()
    main(episodes=args.episodes, budget=args.budget, top_m=args.top_m, u_max=args.u_max, seed=args.seed,
         env_dir=args.env_dir, pretrain_dir=args.pretrain_dir, checkpoint_dir=args.checkpoint_dir,
         device_name=args.device, cpu_threads=args.cpu_threads, amp=not args.no_amp, compile_model=args.compile,
         train_simulations=args.train_simulations, incremental=not args.exact_step_evaluation,
         validation_interval=args.validation_interval, validation_seeds=args.validation_seeds,
         validation_simulations=args.validation_simulations, deterministic_validation=not args.non_deterministic_validation,
         patience=args.patience, min_episodes=args.min_episodes, min_delta=args.min_delta,
         replay_capacity=args.replay_capacity, replay_warmup=args.replay_warmup, batch_size=args.batch_size,
         update_frequency=args.update_frequency, learning_rate=args.learning_rate, gamma=args.gamma, n_step=args.n_step,
         target_update_interval=args.target_update_interval, reward_scale=args.reward_scale,
         epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay_steps=args.epsilon_decay_steps,
         noisy_training=args.noisy_training, q_network=args.q_network, dynamic_candidates=args.dynamic_candidates,
         wide_candidate_size=args.wide_candidate_size, shortlist_size=args.shortlist_size,
         residual_threshold=args.residual_threshold, init_checkpoint=args.init_checkpoint,
         train_worlds=args.train_worlds, validation_worlds=args.validation_worlds)