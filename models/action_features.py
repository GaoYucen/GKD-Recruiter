from __future__ import annotations

import numpy as np


LOCAL_FEATURE_NAMES = (
    "quality",
    "affinity",
    "quality_affinity",
    "task_residual",
    "worker_load_ratio",
    "worker_capacity_remaining_ratio",
    "task_selected_count_ratio",
    "task_demand_ratio",
    "budget_remaining_ratio",
    "step_ratio",
    "worker_influence_ratio",
    "overlap_proxy",
    "worker_selected_task_ratio",
    "task_current_ets",
    "saturation_risk",
    "worker_versatility_ratio",
    "worker_best_alt_score",
    "worker_second_alt_score",
    "worker_current_minus_best_alt",
    "task_eligible_worker_ratio",
    "task_best_worker_score",
    "task_second_worker_score",
    "task_best_second_gap",
    "task_scarcity",
)

GLOBAL_STATE_NAMES = (
    "current_ets",
    "budget_remaining_ratio",
    "task_residual_mean",
    "task_residual_max",
    "task_residual_min",
    "task_residual_std",
    "saturated_task_ratio",
    "unsatisfied_task_ratio",
    "worker_load_mean",
    "worker_load_max_ratio",
    "available_worker_ratio",
    "remaining_capacity_mean",
    "worker_utilization_mean",
    "worker_utilization_max",
    "step_ratio",
)


def build_dynamic_action_features(
    q_matrix: np.ndarray,
    a_matrix: np.ndarray,
    task_demands: np.ndarray,
    worker_load: np.ndarray,
    worker_capacity: np.ndarray,
    task_ets: np.ndarray,
    task_selected_count: np.ndarray,
    current_step: int,
    budget_k: int,
    pair_actions: np.ndarray,
    worker_influence_proxy: np.ndarray | None = None,
    pair_overlap_proxy: np.ndarray | None = None,
    worker_versatility_proxy: np.ndarray | None = None,
) -> np.ndarray:
    """Build normalized dynamic local features for candidate actions.

    Parameters are expected in local worker/task indexing space.
    """
    actions = np.asarray(pair_actions, dtype=np.int64)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("pair_actions must have shape [A, 2]")

    q_matrix = np.asarray(q_matrix, dtype=np.float32)
    a_matrix = np.asarray(a_matrix, dtype=np.float32)
    task_demands = np.asarray(task_demands, dtype=np.float32)
    worker_load = np.asarray(worker_load, dtype=np.float32)
    worker_capacity = np.asarray(worker_capacity, dtype=np.float32)
    task_ets = np.asarray(task_ets, dtype=np.float32)
    task_selected_count = np.asarray(task_selected_count, dtype=np.float32)

    worker_idx = actions[:, 0]
    task_idx = actions[:, 1]

    q = q_matrix[worker_idx, task_idx]
    a = a_matrix[worker_idx, task_idx]
    qa = q * a
    task_residual = np.clip(1.0 - task_ets[task_idx], 0.0, 1.0)
    task_current_ets = np.clip(task_ets[task_idx], 0.0, 1.0)

    max_capacity = max(float(np.max(worker_capacity)) if worker_capacity.size else 0.0, 1.0)
    max_demand = max(float(np.max(task_demands)) if task_demands.size else 0.0, 1.0)
    worker_load_norm = worker_load[worker_idx] / max_capacity
    capacity_remaining = np.clip(worker_capacity[worker_idx] - worker_load[worker_idx], 0.0, None) / max_capacity
    selected_count_norm = task_selected_count[task_idx] / max(float(budget_k), 1.0)
    demand_norm = task_demands[task_idx] / max_demand
    budget_remaining = np.clip(float(budget_k - current_step), 0.0, None) / max(float(budget_k), 1.0)
    step_norm = np.clip(float(current_step), 0.0, None) / max(float(budget_k), 1.0)
    worker_selected_task_ratio = worker_load[worker_idx] / max(float(budget_k), 1.0)
    saturation_risk = np.clip(task_current_ets + 0.5 * qa / np.maximum(demand_norm, 1e-6), 0.0, 1.5)

    if worker_influence_proxy is None:
        worker_influence = np.ones(len(worker_idx), dtype=np.float32)
    else:
        influence = np.asarray(worker_influence_proxy, dtype=np.float32)
        max_influence = max(float(np.max(influence)) if influence.size else 0.0, 1.0)
        worker_influence = influence[worker_idx] / max_influence

    if pair_overlap_proxy is None:
        overlap = np.zeros(len(worker_idx), dtype=np.float32)
    else:
        overlap = np.asarray(pair_overlap_proxy, dtype=np.float32)

    valid_worker_mask = worker_load < worker_capacity
    qa_matrix = q_matrix * a_matrix
    unsaturated_task_mask = task_ets < 0.999
    masked_scores = np.where(valid_worker_mask[:, None], qa_matrix, -np.inf)
    eligible_worker_count = np.sum(valid_worker_mask[:, None] & (qa_matrix > 0.0), axis=0).astype(np.float32)
    eligible_worker_ratio = eligible_worker_count[task_idx] / max(float(q_matrix.shape[0]), 1.0)

    if worker_versatility_proxy is None:
        versatility = np.sum((qa_matrix > 0.0) & valid_worker_mask[:, None], axis=1).astype(np.float32)
    else:
        versatility = np.asarray(worker_versatility_proxy, dtype=np.float32)
    versatility_norm = versatility[worker_idx] / max(float(np.max(versatility)) if versatility.size else 0.0, 1.0)

    worker_best_alt = np.full(len(worker_idx), 0.0, dtype=np.float32)
    worker_second_alt = np.full(len(worker_idx), 0.0, dtype=np.float32)
    task_best = np.full(len(task_demands), 0.0, dtype=np.float32)
    task_second = np.full(len(task_demands), 0.0, dtype=np.float32)

    for action_pos, (w, t) in enumerate(actions):
        row = masked_scores[int(w)].copy()
        unavailable = ~unsaturated_task_mask
        row[unavailable] = -np.inf
        row[int(t)] = -np.inf
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        top = np.sort(finite)[::-1]
        worker_best_alt[action_pos] = float(top[0])
        worker_second_alt[action_pos] = float(top[1]) if top.size > 1 else float(top[0])

    for t in np.unique(task_idx):
        col = masked_scores[:, int(t)]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        top = np.sort(finite)[::-1]
        task_best[int(t)] = float(top[0])
        task_second[int(t)] = float(top[1]) if top.size > 1 else float(top[0])

    worker_best_alt_score = worker_best_alt
    worker_second_alt_score = worker_second_alt
    worker_current_minus_best_alt = qa - worker_best_alt_score
    task_best_worker_score = task_best[task_idx]
    task_second_worker_score = task_second[task_idx]
    task_best_second_gap = task_best_worker_score - task_second_worker_score
    remaining_demand_ratio = np.clip(task_demands * np.clip(1.0 - task_ets, 0.0, 1.0), 0.0, None) / max_demand
    task_scarcity = remaining_demand_ratio[task_idx] / np.maximum(eligible_worker_count[task_idx], 1.0)

    return np.stack(
        [
            q,
            a,
            qa,
            task_residual,
            worker_load_norm,
            capacity_remaining,
            selected_count_norm,
            demand_norm,
            np.full(len(worker_idx), budget_remaining, dtype=np.float32),
            np.full(len(worker_idx), step_norm, dtype=np.float32),
            worker_influence,
            overlap,
            worker_selected_task_ratio,
            task_current_ets,
            saturation_risk,
            versatility_norm,
            worker_best_alt_score,
            worker_second_alt_score,
            worker_current_minus_best_alt,
            eligible_worker_ratio,
            task_best_worker_score,
            task_second_worker_score,
            task_best_second_gap,
            task_scarcity,
        ],
        axis=1,
    ).astype(np.float32)


def summarize_global_state(
    current_ets: float,
    current_step: int,
    budget_k: int,
    task_ets: np.ndarray,
    worker_load: np.ndarray,
    worker_capacity: np.ndarray,
) -> np.ndarray:
    task_ets = np.asarray(task_ets, dtype=np.float32)
    worker_load = np.asarray(worker_load, dtype=np.float32)
    worker_capacity = np.asarray(worker_capacity, dtype=np.float32)
    residual = np.clip(1.0 - task_ets, 0.0, 1.0)
    available = worker_load < worker_capacity
    remaining_capacity = np.clip(worker_capacity - worker_load, 0.0, None)
    utilization = np.divide(worker_load, np.maximum(worker_capacity, 1.0), out=np.zeros_like(worker_load), where=np.maximum(worker_capacity, 1.0) > 0)
    return np.asarray(
        [
            float(current_ets),
            np.clip(float(budget_k - current_step), 0.0, None) / max(float(budget_k), 1.0),
            float(residual.mean()) if residual.size else 0.0,
            float(residual.max()) if residual.size else 0.0,
            float(residual.min()) if residual.size else 0.0,
            float(residual.std()) if residual.size else 0.0,
            float(np.mean(task_ets >= 0.999)) if task_ets.size else 0.0,
            float(np.mean(task_ets < 0.999)) if task_ets.size else 0.0,
            float(worker_load.mean()) if worker_load.size else 0.0,
            float(worker_load.max()) / max(float(np.max(worker_capacity)) if worker_capacity.size else 0.0, 1.0),
            float(available.mean()) if available.size else 0.0,
            float(remaining_capacity.mean()) if remaining_capacity.size else 0.0,
            float(utilization.mean()) if utilization.size else 0.0,
            float(utilization.max()) if utilization.size else 0.0,
            np.clip(float(current_step), 0.0, None) / max(float(budget_k), 1.0),
        ],
        dtype=np.float32,
    )