#!/usr/bin/env python3
"""Export processed SNAP benchmarks to the legacy GKD training layout.

The SNAP builder writes a rich processed dataset directory with full-node task
matrices in ``benchmark_arrays.npz``.  The current GKD training/evaluation code
expects text files split into ``env_params`` and ``model_inputs``.  This adapter
keeps the processed benchmark as the source of truth and materializes a
training-ready experiment directory without mean imputation or label
propagation.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

EPS = 1e-12


def row_topk_similarity(x: np.ndarray, k: int = 10) -> np.ndarray:
    z = x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)
    s = np.clip(z @ z.T, 0.0, 1.0)
    np.fill_diagonal(s, 0.0)
    if s.shape[0] <= 1:
        return s
    kk = min(k, s.shape[1] - 1)
    ids = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
    mask = np.zeros_like(s)
    rows = np.arange(s.shape[0])[:, None]
    mask[rows, ids] = s[rows, ids]
    return np.maximum(mask, mask.T)


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    return (x - lo) / max(hi - lo, EPS)


def build_worker_features(stats: pd.DataFrame, worker_node_ids: np.ndarray, dim: int) -> np.ndarray:
    stats_idx = stats.set_index("user_id")
    # candidate_workers.csv stores both original user_id and local node_index.
    cols = ["checkins", "active_days", "unique_pois", "degree", "mobility_entropy", "sender_influence", "receiver_susceptibility", "center_lat", "center_lon"]
    raw = stats_idx.loc[worker_node_ids, cols].to_numpy(np.float32)
    feat = np.zeros((len(worker_node_ids), dim), dtype=np.float32)
    usable = min(dim, raw.shape[1])
    for j in range(usable):
        feat[:, j] = minmax(raw[:, j])
    for j in range(usable, dim):
        feat[:, j] = 0.5 * np.sin((j + 1) * feat[:, 0]) + 0.5 * np.cos((j + 1) * feat[:, 1])
    return feat


def build_task_features(tasks: pd.DataFrame, demands: np.ndarray, rewards: np.ndarray, dim: int) -> np.ndarray:
    raw = np.column_stack([
        tasks["latitude"].to_numpy(np.float32),
        tasks["longitude"].to_numpy(np.float32),
        tasks["checkins"].to_numpy(np.float32) if "checkins" in tasks else np.ones(len(tasks), dtype=np.float32),
        tasks["unique_users"].to_numpy(np.float32) if "unique_users" in tasks else np.ones(len(tasks), dtype=np.float32),
        demands.astype(np.float32),
        rewards.astype(np.float32),
    ])
    feat = np.zeros((len(tasks), dim), dtype=np.float32)
    usable = min(dim, raw.shape[1])
    for j in range(usable):
        feat[:, j] = minmax(raw[:, j])
    for j in range(usable, dim):
        feat[:, j] = 0.5 * np.sin((j + 1) * feat[:, 0]) + 0.5 * np.cos((j + 1) * feat[:, 1])
    return feat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", required=True, type=pathlib.Path)
    p.add_argument("--output-root", required=True, type=pathlib.Path)
    p.add_argument("--top-m", type=int, default=5)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--sim-k", type=int, default=10)
    args = p.parse_args()

    src = args.processed_dir
    out = args.output_root
    env = out / "env_params"
    model = out / "model_inputs"
    env.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)

    arrays = np.load(src / "benchmark_arrays.npz")
    mapping = pd.read_csv(src / "user_mapping.csv")
    stats = pd.read_csv(src / "user_statistics.csv")
    tasks = pd.read_csv(src / "tasks.csv")
    workers = pd.read_csv(src / "candidate_workers.csv")
    metadata = json.loads((src / "metadata.json").read_text(encoding="utf-8"))

    full_q = np.asarray(arrays["q_matrix"], dtype=np.float32)
    full_a = np.asarray(arrays["a_matrix"], dtype=np.float32)
    edge_index = np.asarray(arrays["edge_index"], dtype=np.int64)
    if edge_index.shape[0] == 2:
        edge_pairs = edge_index.T
    else:
        edge_pairs = edge_index
    edge_weight = np.asarray(arrays["edge_weight"], dtype=np.float32)
    worker_indices = np.asarray(arrays["worker_indices"], dtype=np.int64)
    demands = np.asarray(arrays["task_demands"], dtype=np.float32)
    rewards = np.asarray(arrays["task_rewards"], dtype=np.float32)
    q = full_q[worker_indices]
    a = full_a[worker_indices]

    np.savetxt(env / "edge_index.txt", edge_pairs, fmt="%d")
    np.savetxt(env / "w_ij.txt", edge_weight, fmt="%.8f")
    np.savetxt(env / "worker_indices.txt", worker_indices, fmt="%d")
    np.savetxt(env / "q_matrix.txt", q, fmt="%.8f")
    np.savetxt(env / "a_matrix.txt", a, fmt="%.8f")
    np.savetxt(env / "full_q_matrix.txt", full_q, fmt="%.8f")
    np.savetxt(env / "full_a_matrix.txt", full_a, fmt="%.8f")
    np.savetxt(env / "task_demands.txt", demands, fmt="%.8f")
    np.savetxt(env / "task_rewards.txt", rewards, fmt="%.8f")

    node_to_user = mapping.sort_values("node_index")["user_id"].to_numpy()
    worker_user_ids = node_to_user[worker_indices]
    wf = build_worker_features(stats, worker_user_ids, args.feature_dim)
    tf = build_task_features(tasks, demands, rewards, args.feature_dim)
    np.savetxt(model / "worker_features.txt", wf, fmt="%.8f")
    np.savetxt(model / "task_features.txt", tf, fmt="%.8f")

    local = {int(node): i for i, node in enumerate(worker_indices)}
    social = np.zeros((len(worker_indices), len(worker_indices)), dtype=np.float32)
    for (u, v), w in zip(edge_pairs, edge_weight):
        if int(u) in local and int(v) in local:
            social[local[int(u)], local[int(v)]] = float(w)
    np.savetxt(model / "social_adj.txt", social, fmt="%.8f")

    top_m = min(args.top_m, q.shape[1])
    top = np.argpartition(-q, top_m - 1, axis=1)[:, :top_m]
    wt = np.stack([np.repeat(np.arange(q.shape[0]), top_m), top.reshape(-1)], axis=1)
    np.savetxt(model / "hetero_edge_index.txt", wt, fmt="%d")
    np.savetxt(model / "worker_sim_adj.txt", row_topk_similarity(wf, args.sim_k), fmt="%.8f")
    np.savetxt(model / "task_sim_adj.txt", row_topk_similarity(tf, args.sim_k), fmt="%.8f")

    report = {
        "valid": bool(
            full_q.shape == full_a.shape
            and q.shape == a.shape
            and np.isfinite(full_q).all()
            and np.isfinite(full_a).all()
            and np.isfinite(edge_weight).all()
            and (0 <= full_q).all() and (full_q <= 1).all()
            and (0 <= full_a).all() and (full_a <= 1).all()
            and (0 <= edge_weight).all() and (edge_weight <= 1).all()
        ),
        "source": "processed_snap_benchmark",
        "processed_dir": str(src),
        "dataset": metadata.get("dataset"),
        "graph_size": int(full_q.shape[0]),
        "num_edges": int(edge_pairs.shape[0]),
        "num_candidate_workers": int(len(worker_indices)),
        "num_tasks": int(full_q.shape[1]),
        "top_m": int(top_m),
        "attribute_source": "per-node real check-in history",
        "imputation": "none",
        "q_range_full": [float(full_q.min()), float(full_q.max())],
        "a_range_full": [float(full_a.min()), float(full_a.max())],
        "edge_weight_range": [float(edge_weight.min()), float(edge_weight.max())],
        "demand_range": [float(demands.min()), float(demands.max())],
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()