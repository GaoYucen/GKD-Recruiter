from __future__ import annotations

import os
import numpy as np

from .action_features import LOCAL_FEATURE_NAMES, build_dynamic_action_features


SOURCE_NAMES = (
    'quality_suitability',
    'quality',
    'suitability',
    'quality_suitability_per_demand',
    'influence_quality_suitability',
    'residual_task',
    'task_minimum',
)


def _candidate_score(local_features: np.ndarray, variant: str | None = None) -> np.ndarray:
    variant_name = str(variant or os.environ.get('GKD_SHORTLIST_SCORE_VARIANT', 'default')).strip().lower()
    quality = local_features[:, 0]
    affinity = local_features[:, 1]
    quality_affinity = local_features[:, 2]
    task_residual = local_features[:, 3]
    capacity_remaining = np.maximum(local_features[:, 5], 0.0)
    influence = local_features[:, 10]
    overlap = local_features[:, 11]
    task_current_ets = local_features[:, 13]
    task_best_second_gap = local_features[:, 22]
    task_scarcity = local_features[:, 23]

    if variant_name == 'residual_focus':
        return (
            0.08 * quality
            + 0.12 * affinity
            + 0.24 * quality_affinity
            + 0.24 * task_residual
            + 0.10 * capacity_remaining
            + 0.08 * influence
            + 0.18 * task_scarcity
            + 0.08 * (1.0 - np.clip(task_current_ets, 0.0, 1.0))
            - 0.12 * overlap
        )
    if variant_name == 'less_overlap':
        return (
            0.10 * quality
            + 0.15 * affinity
            + 0.30 * quality_affinity
            + 0.16 * task_residual
            + 0.10 * capacity_remaining
            + 0.10 * influence
            + 0.05 * task_best_second_gap
            - 0.08 * overlap
        )
    return (
        0.10 * quality
        + 0.15 * affinity
        + 0.30 * quality_affinity
        + 0.15 * task_residual
        + 0.10 * capacity_remaining
        + 0.10 * influence
        - 0.20 * overlap
    )


def compute_selected_worker_overlap(
    worker_embeddings: np.ndarray,
    actions: np.ndarray,
    selected_worker_mask: np.ndarray | None,
    *,
    reduction: str = 'max',
) -> np.ndarray:
    """Compute embedding-based overlap between candidate workers and already selected workers.

    Args:
        worker_embeddings: [num_workers, dim] worker representations.
        actions: [num_actions, 2] local (worker, task) pairs.
        selected_worker_mask: [num_tasks, num_workers] mask of already selected workers per task.
        reduction: overlap aggregation across selected workers, supports "max" and "mean".
    """
    actions = np.asarray(actions, dtype=np.int64)
    if len(actions) == 0 or selected_worker_mask is None:
        return np.zeros(len(actions), dtype=np.float32)
    embeds = np.asarray(worker_embeddings, dtype=np.float32)
    if embeds.ndim != 2:
        raise ValueError('worker_embeddings must have shape [num_workers, dim]')
    selected_mask = np.asarray(selected_worker_mask, dtype=np.float32) > 0
    if selected_mask.ndim != 2:
        raise ValueError('selected_worker_mask must have shape [num_tasks, num_workers]')
    if selected_mask.shape[1] != embeds.shape[0]:
        raise ValueError('selected_worker_mask width must match num_workers in worker_embeddings')
    if not selected_mask.any():
        return np.zeros(len(actions), dtype=np.float32)

    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    normalized = embeds / np.clip(norms, 1e-8, None)
    overlap_by_task = np.zeros((selected_mask.shape[0], embeds.shape[0]), dtype=np.float32)
    for task_idx in range(selected_mask.shape[0]):
        chosen = np.flatnonzero(selected_mask[task_idx])
        if len(chosen) == 0:
            continue
        sims = normalized @ normalized[chosen].T
        if reduction == 'mean':
            overlap_by_task[task_idx] = sims.mean(axis=1)
        else:
            overlap_by_task[task_idx] = sims.max(axis=1)
    return np.clip(overlap_by_task[actions[:, 1], actions[:, 0]], 0.0, 1.0).astype(np.float32)


def _resolve_overlap_proxy(
    q_workers: np.ndarray,
    a_workers: np.ndarray,
    actions: np.ndarray,
    pair_overlap_proxy: np.ndarray | None,
) -> np.ndarray:
    if pair_overlap_proxy is None:
        return np.zeros(len(actions), dtype=np.float32)
    overlap = np.asarray(pair_overlap_proxy)
    if overlap.ndim == 1:
        if overlap.shape[0] != len(actions):
            raise ValueError('1D pair_overlap_proxy must match action count')
        return np.clip(overlap.astype(np.float32), 0.0, 1.0)
    if overlap.ndim != 2 or overlap.shape[0] != q_workers.shape[1] or overlap.shape[1] != q_workers.shape[0]:
        raise ValueError('pair_overlap_proxy must be either action-wise [A] or selected-worker mask [num_tasks, num_workers]')
    worker_repr = np.concatenate([q_workers, a_workers], axis=1).astype(np.float32)
    return compute_selected_worker_overlap(worker_repr, actions, overlap, reduction='max')


def build_topm_actions(
    q_workers: np.ndarray,
    a_workers: np.ndarray,
    top_m: int,
    task_demands: np.ndarray | None = None,
    demand_normalized: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    q_workers = np.asarray(q_workers, dtype=float)
    a_workers = np.asarray(a_workers, dtype=float)

    if q_workers.shape != a_workers.shape:
        raise ValueError(f'q/a shape mismatch: {q_workers.shape} vs {a_workers.shape}')
    if q_workers.ndim != 2:
        raise ValueError(f'q_workers and a_workers must be 2D, got ndim={q_workers.ndim}')

    num_workers, num_tasks = q_workers.shape
    m = max(1, min(int(top_m), num_tasks))

    scores = q_workers * a_workers
    if demand_normalized:
        if task_demands is None:
            raise ValueError('task_demands is required when demand_normalized=True')
        demands = np.maximum(np.asarray(task_demands, dtype=float), 1e-8)
        if demands.shape != (num_tasks,):
            raise ValueError(f'task_demands shape must be ({num_tasks},), got {demands.shape}')
        scores = scores / demands[None, :]

    top = np.argpartition(-scores, m - 1, axis=1)[:, :m]
    ordered_top = np.empty_like(top)
    for worker_idx in range(num_workers):
        candidate_tasks = top[worker_idx]
        order = np.lexsort((candidate_tasks, -q_workers[worker_idx, candidate_tasks], -scores[worker_idx, candidate_tasks]))
        ordered_top[worker_idx] = candidate_tasks[order]

    actions = np.stack([np.repeat(np.arange(num_workers), m), ordered_top.reshape(-1)], axis=1).astype(np.int64)
    action_ids = (actions[:, 0] * num_tasks + actions[:, 1]).astype(np.int64)
    return actions, action_ids


def build_dynamic_shortlist(
    q_workers: np.ndarray,
    a_workers: np.ndarray,
    task_demands: np.ndarray,
    worker_load: np.ndarray,
    worker_capacity: np.ndarray,
    task_ets: np.ndarray,
    task_selected_count: np.ndarray,
    current_step: int,
    budget_k: int,
    top_m_static: int = 10,
    wide_candidate_size: int = 2048,
    shortlist_size: int = 256,
    residual_threshold: float = 0.005,
    worker_influence_proxy: np.ndarray | None = None,
    pair_overlap_proxy: np.ndarray | None = None,
    worker_embeddings: np.ndarray | None = None,
    min_candidates_per_task: int = 1,
    return_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | dict[str, int] | int | float]]:
    q_workers = np.asarray(q_workers, dtype=float)
    a_workers = np.asarray(a_workers, dtype=float)
    task_demands = np.asarray(task_demands, dtype=float)
    num_workers, num_tasks = q_workers.shape
    if a_workers.shape != q_workers.shape:
        raise ValueError('q_workers and a_workers must have matching shapes')

    valid_task_mask = np.asarray(task_ets, dtype=float) < (1.0 - float(residual_threshold))
    if not valid_task_mask.any():
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)

    m = max(1, min(int(top_m_static), num_tasks))
    demand_safe = np.maximum(task_demands, 1e-8)
    influence = np.ones(num_workers, dtype=float) if worker_influence_proxy is None else np.asarray(worker_influence_proxy, dtype=float)
    score_sources = [
        q_workers * a_workers,
        q_workers,
        a_workers,
        (q_workers * a_workers) / demand_safe[None, :],
        influence[:, None] * q_workers * a_workers,
    ]
    candidate_source_masks: dict[int, int] = {}

    def _mark_candidate(worker_idx: int, task_idx: int, source_name: str) -> None:
        action_id = int(worker_idx * num_tasks + task_idx)
        candidate_ids.add(action_id)
        bit = 1 << SOURCE_NAMES.index(source_name)
        candidate_source_masks[action_id] = candidate_source_masks.get(action_id, 0) | bit

    candidate_ids: set[int] = set()
    for source_name, scores in zip(SOURCE_NAMES[:5], score_sources, strict=True):
        masked_scores = np.where(valid_task_mask[None, :], scores, -np.inf)
        top = np.argpartition(-masked_scores, min(m - 1, masked_scores.shape[1] - 1), axis=1)[:, :m]
        for worker_idx in range(num_workers):
            for task_idx in top[worker_idx]:
                _mark_candidate(worker_idx, int(task_idx), source_name)

    residual_scores = (q_workers * a_workers) * np.clip(1.0 - np.asarray(task_ets, dtype=float), 0.0, 1.0)[None, :]
    for task_idx in np.flatnonzero(valid_task_mask):
        worker_top = np.argpartition(-residual_scores[:, task_idx], min(m - 1, num_workers - 1))[:m]
        for worker_idx in worker_top:
            _mark_candidate(int(worker_idx), int(task_idx), 'residual_task')

    minimum_per_task = max(int(min_candidates_per_task), 0)
    if minimum_per_task > 0:
        for task_idx in np.flatnonzero(valid_task_mask):
            task_scores = q_workers[:, int(task_idx)] * a_workers[:, int(task_idx)]
            eligible = np.flatnonzero(np.asarray(worker_load, dtype=float) < np.asarray(worker_capacity, dtype=float))
            if len(eligible) == 0:
                continue
            ranked_workers = eligible[np.argsort(-task_scores[eligible], kind='stable')]
            for worker_idx in ranked_workers[:minimum_per_task]:
                _mark_candidate(int(worker_idx), int(task_idx), 'task_minimum')

    if not candidate_ids:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)

    ordered_ids = np.asarray(sorted(candidate_ids), dtype=np.int64)
    actions = np.stack([ordered_ids // num_tasks, ordered_ids % num_tasks], axis=1)
    load = np.asarray(worker_load, dtype=float)
    cap = np.asarray(worker_capacity, dtype=float)
    keep = (load[actions[:, 0]] < cap[actions[:, 0]]) & valid_task_mask[actions[:, 1]]
    actions = actions[keep]
    ordered_ids = ordered_ids[keep]
    if len(actions) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0, len(LOCAL_FEATURE_NAMES)), dtype=np.float32)

    candidate_source_bits = np.asarray([candidate_source_masks.get(int(action_id), 0) for action_id in ordered_ids], dtype=np.int64)

    if worker_embeddings is not None and pair_overlap_proxy is not None and np.asarray(pair_overlap_proxy).ndim == 2:
        full_overlap = compute_selected_worker_overlap(worker_embeddings, actions, pair_overlap_proxy, reduction='max')
    else:
        full_overlap = _resolve_overlap_proxy(q_workers, a_workers, actions, pair_overlap_proxy)
    full_local = build_dynamic_action_features(
        q_matrix=q_workers,
        a_matrix=a_workers,
        task_demands=task_demands,
        worker_load=worker_load,
        worker_capacity=worker_capacity,
        task_ets=task_ets,
        task_selected_count=task_selected_count,
        current_step=current_step,
        budget_k=budget_k,
        pair_actions=actions,
        worker_influence_proxy=worker_influence_proxy,
        pair_overlap_proxy=full_overlap,
    )
    wide_order = np.argsort(-_candidate_score(full_local), kind='stable')
    if wide_candidate_size > 0:
        wide_order = wide_order[: min(int(wide_candidate_size), len(wide_order))]
    wide_actions = actions[wide_order]
    wide_ids = ordered_ids[wide_order]

    if worker_embeddings is not None and pair_overlap_proxy is not None and np.asarray(pair_overlap_proxy).ndim == 2:
        wide_overlap = compute_selected_worker_overlap(worker_embeddings, wide_actions, pair_overlap_proxy, reduction='max')
    else:
        wide_overlap = _resolve_overlap_proxy(q_workers, a_workers, wide_actions, pair_overlap_proxy)
    wide_local = build_dynamic_action_features(
        q_matrix=q_workers,
        a_matrix=a_workers,
        task_demands=task_demands,
        worker_load=worker_load,
        worker_capacity=worker_capacity,
        task_ets=task_ets,
        task_selected_count=task_selected_count,
        current_step=current_step,
        budget_k=budget_k,
        pair_actions=wide_actions,
        worker_influence_proxy=worker_influence_proxy,
        pair_overlap_proxy=wide_overlap,
    )
    shortlist_scores = _candidate_score(wide_local)
    shortlist_order = np.argsort(-shortlist_scores, kind='stable')
    if shortlist_size > 0:
        shortlist_order = shortlist_order[: min(int(shortlist_size), len(shortlist_order))]
    selected_indices = shortlist_order.tolist()
    if minimum_per_task > 0 and len(wide_actions) > 0:
        covered_tasks = set(int(task_idx) for task_idx in wide_actions[selected_indices, 1].tolist()) if selected_indices else set()
        selected_set = set(selected_indices)
        for task_idx in np.flatnonzero(valid_task_mask):
            if int(task_idx) in covered_tasks:
                continue
            matches = np.flatnonzero(wide_actions[:, 1] == int(task_idx))
            if len(matches) == 0:
                continue
            best_idx = int(matches[np.argmax(shortlist_scores[matches])])
            if best_idx not in selected_set:
                selected_indices.append(best_idx)
                selected_set.add(best_idx)
                covered_tasks.add(int(task_idx))
        selected_indices = sorted(selected_indices, key=lambda idx: (-float(shortlist_scores[idx]), idx))
        if shortlist_size > 0:
            selected_indices = selected_indices[: min(int(shortlist_size), len(selected_indices))]

    selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
    shortlist_actions = wide_actions[selected_indices_np].astype(np.int64)
    shortlist_ids = wide_ids[selected_indices_np].astype(np.int64)
    shortlist_local = wide_local[selected_indices_np].astype(np.float32)
    if not return_metadata:
        return shortlist_actions, shortlist_ids, shortlist_local

    shortlist_source_bits = candidate_source_bits[wide_order][selected_indices_np]
    source_counts = {name: 0 for name in SOURCE_NAMES}
    for bits in shortlist_source_bits.tolist():
        for bit_idx, name in enumerate(SOURCE_NAMES):
            if bits & (1 << bit_idx):
                source_counts[name] += 1
    task_counts = np.bincount(shortlist_actions[:, 1], minlength=num_tasks) if len(shortlist_actions) else np.zeros(num_tasks, dtype=np.int64)
    metadata = {
        'source_bits': shortlist_source_bits.astype(np.int64),
        'source_counts': source_counts,
        'task_counts': task_counts.astype(np.int64),
        'raw_candidate_size': int(len(ordered_ids)),
        'wide_pool_size': int(len(wide_actions)),
        'shortlist_size': int(len(shortlist_actions)),
        'min_task_coverage': float(task_counts[valid_task_mask].min()) if valid_task_mask.any() and len(shortlist_actions) else 0.0,
    }
    return shortlist_actions, shortlist_ids, shortlist_local, metadata
