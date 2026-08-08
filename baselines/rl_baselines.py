from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.gkd_env import GKDEnv


SeedPair = Tuple[int, int]


class SimpleQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclass(frozen=True)
class RLBaselineRollout:
    name: str
    seed_pairs: List[SeedPair]
    selection_time_sec: float
    params: Dict[str, object]


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _masked_argmax(q_values: torch.Tensor, valid_mask: np.ndarray) -> int:
    masked = q_values.clone()
    invalid = torch.as_tensor(~valid_mask, dtype=torch.bool, device=q_values.device)
    masked[invalid] = torch.finfo(masked.dtype).min
    return int(torch.argmax(masked).item())


def _select_action(net: nn.Module, state: torch.Tensor, valid_mask: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    valid_ids = np.flatnonzero(valid_mask)
    if len(valid_ids) == 0:
        raise ValueError("no valid actions available")
    if float(epsilon) > 0.0 and float(rng.random()) < float(epsilon):
        return int(rng.choice(valid_ids))
    with torch.no_grad():
        q_values = net(state.unsqueeze(0)).squeeze(0)
    return _masked_argmax(q_values, valid_mask)


def _heuristic_action_scores(env: GKDEnv, valid_mask: np.ndarray) -> np.ndarray:
    scores = np.full(valid_mask.shape[0], -np.inf, dtype=np.float32)
    valid_ids = np.flatnonzero(valid_mask)
    if len(valid_ids) == 0:
        return scores
    task_ets = np.asarray(env.task_ets, dtype=np.float32)
    residual = np.clip(1.0 - task_ets, 0.0, 1.0)
    for action_id in valid_ids.tolist():
        worker_idx = int(action_id) // int(env.num_tasks)
        task_idx = int(action_id) % int(env.num_tasks)
        qa = float(env.q_matrix[worker_idx, task_idx] * env.a_matrix[worker_idx, task_idx])
        remaining = max(float(env.u_max - env.worker_load[worker_idx]), 0.0) / max(float(env.u_max), 1.0)
        scores[int(action_id)] = qa * (0.65 + 0.35 * float(residual[task_idx])) + 0.10 * remaining
    return scores


def train_dqn_selector_baseline(
    env_dir: str | Path,
    checkpoint_path: str | Path,
    budget_k: int,
    u_max: int,
    episodes: int = 20,
    seed: int = 42,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.90,
) -> Dict[str, object]:
    env = GKDEnv(env_dir=str(env_dir), budget_K=budget_k, u_max=u_max, seed=seed, num_simulations=1)
    device = _device()
    action_dim = env.num_workers * env.num_tasks
    state_dim = int(env.state_vector().numel())
    net = SimpleQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target = SimpleQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(int(seed))
    epsilon = float(epsilon_start)
    history: List[float] = []

    for episode in range(int(episodes)):
        state = env.reset().to(device)
        done = False
        total_reward = 0.0
        while not done:
            valid_mask = env.valid_action_mask()
            action = _select_action(net, state, valid_mask, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.to(device)
            with torch.no_grad():
                next_mask = env.valid_action_mask()
                next_q = target(next_state.unsqueeze(0)).squeeze(0)
                if next_mask.any() and not done:
                    next_value = torch.max(next_q[torch.as_tensor(next_mask, dtype=torch.bool, device=device)])
                else:
                    next_value = torch.tensor(0.0, device=device)
                target_value = torch.tensor(float(reward), dtype=torch.float32, device=device) + float(gamma) * next_value
            pred = net(state.unsqueeze(0)).squeeze(0)[int(action)]
            loss = loss_fn(pred, target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            total_reward += float(reward)
        if episode % 5 == 0:
            target.load_state_dict(net.state_dict())
        epsilon = max(float(epsilon_end), float(epsilon) * float(epsilon_decay))
        history.append(total_reward)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "meta": {
                "baseline": "DQNSelector",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "budget_k": int(budget_k),
                "u_max": int(u_max),
                "seed": int(seed),
            },
        },
        checkpoint,
    )
    checkpoint.with_suffix(checkpoint.suffix + ".json").write_text(
        json.dumps({"episodes": int(episodes), "reward_history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"checkpoint": str(checkpoint), "reward_history": history}


def rollout_dqn_selector_baseline(
    env_dir: str | Path,
    checkpoint_path: str | Path,
    budget_k: int,
    u_max: int,
    seed: int = 42,
    name: str = "DQNSelector",
    heuristic_blend: float = 0.25,
) -> RLBaselineRollout:
    env = GKDEnv(env_dir=str(env_dir), budget_K=budget_k, u_max=u_max, seed=seed, num_simulations=1)
    payload = torch.load(Path(checkpoint_path), map_location=_device(), weights_only=False)
    meta = dict(payload.get("meta", {}))
    device = _device()
    net = SimpleQNetwork(int(meta["state_dim"]), int(meta["action_dim"]), int(meta.get("hidden_dim", 128))).to(device)
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    start = time.perf_counter()
    state = env.reset().to(device)
    done = False
    rng = np.random.default_rng(int(seed))
    while not done:
        valid_mask = env.valid_action_mask()
        with torch.no_grad():
            q_values = net(state.unsqueeze(0)).squeeze(0)
        heuristic_scores = _heuristic_action_scores(env, valid_mask)
        combined = q_values.detach().cpu().numpy().astype(np.float32)
        valid_ids = np.flatnonzero(valid_mask)
        if len(valid_ids) == 0:
            break
        q_valid = combined[valid_ids]
        q_span = float(np.max(q_valid) - np.min(q_valid)) if len(q_valid) else 0.0
        if q_span < 1e-4:
            action = int(valid_ids[np.argmax(heuristic_scores[valid_ids])])
        else:
            q_norm = (combined - float(np.min(q_valid))) / max(q_span, 1e-6)
            h_valid = heuristic_scores[valid_ids]
            h_min = float(np.min(h_valid))
            h_span = float(np.max(h_valid) - h_min)
            h_norm = np.zeros_like(heuristic_scores, dtype=np.float32)
            if h_span > 1e-8:
                h_norm[valid_ids] = (heuristic_scores[valid_ids] - h_min) / h_span
            mixed = q_norm + float(heuristic_blend) * h_norm
            action = int(valid_ids[np.argmax(mixed[valid_ids])])
        next_state, _, done, _ = env.step(action)
        state = next_state.to(device)
    meta["heuristic_blend"] = float(heuristic_blend)
    return RLBaselineRollout(name=name, seed_pairs=list(env.selected_seeds), selection_time_sec=time.perf_counter() - start, params=meta)


def train_maim_baseline(
    env_dir: str | Path,
    checkpoint_path: str | Path,
    budget_k: int,
    u_max: int,
    num_agents: int = 2,
    episodes: int = 20,
    seed: int = 42,
    hidden_dim: int = 128,
) -> Dict[str, object]:
    # Lightweight proxy: independent DQN heads over task partitions.
    env = GKDEnv(env_dir=str(env_dir), budget_K=budget_k, u_max=u_max, seed=seed, num_simulations=1)
    device = _device()
    state_dim = int(env.state_vector().numel())
    action_dim = env.num_workers * env.num_tasks
    net = SimpleQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target = SimpleQNetwork(state_dim, action_dim, hidden_dim).to(device)
    target.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(int(seed) + 17)
    history: List[float] = []
    task_partitions = np.array_split(np.arange(env.num_tasks, dtype=int), max(int(num_agents), 1))

    for episode in range(int(episodes)):
        state = env.reset().to(device)
        done = False
        total_reward = 0.0
        agent_turn = 0
        while not done:
            valid_mask = env.valid_action_mask()
            allowed_tasks = set(int(t) for t in task_partitions[agent_turn % len(task_partitions)].tolist())
            restricted_mask = valid_mask.copy()
            for action_id in np.flatnonzero(restricted_mask):
                _, task_idx = divmod(int(action_id), env.num_tasks)
                if int(task_idx) not in allowed_tasks:
                    restricted_mask[int(action_id)] = False
            if not restricted_mask.any():
                restricted_mask = valid_mask
            action = _select_action(net, state, restricted_mask, 0.20, rng)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.to(device)
            with torch.no_grad():
                next_mask = env.valid_action_mask()
                next_q = target(next_state.unsqueeze(0)).squeeze(0)
                next_value = torch.max(next_q[torch.as_tensor(next_mask, dtype=torch.bool, device=device)]) if next_mask.any() and not done else torch.tensor(0.0, device=device)
                target_value = torch.tensor(float(reward), dtype=torch.float32, device=device) + 0.99 * next_value
            pred = net(state.unsqueeze(0)).squeeze(0)[int(action)]
            loss = loss_fn(pred, target_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            total_reward += float(reward)
            agent_turn += 1
        if episode % 5 == 0:
            target.load_state_dict(net.state_dict())
        history.append(total_reward)

    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "meta": {
                "baseline": "MAIM",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "hidden_dim": hidden_dim,
                "budget_k": int(budget_k),
                "u_max": int(u_max),
                "seed": int(seed),
                "num_agents": int(num_agents),
            },
        },
        checkpoint,
    )
    return {"checkpoint": str(checkpoint), "reward_history": history}


def rollout_maim_baseline(
    env_dir: str | Path,
    checkpoint_path: str | Path,
    budget_k: int,
    u_max: int,
    seed: int = 42,
) -> RLBaselineRollout:
    return rollout_dqn_selector_baseline(env_dir=env_dir, checkpoint_path=checkpoint_path, budget_k=budget_k, u_max=u_max, seed=seed, name="MAIM")