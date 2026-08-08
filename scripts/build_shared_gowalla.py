"""Rebuild Gowalla v3000/v5000 with a SHARED task set, SHARED global
normalization basis (v5000), and MIXED (quality + degree) candidate pools.

This makes the comparison fair across scales: both sizes see the same 100
tasks and the same affinity normalization denominator, so any residual gain on
v5000 reflects real 'bigger graph -> more/better matches' rather than scale
artifacts.

Reuses pure helpers from build_snap_benchmark.py without modifying it.
Writes processed dirs plus env_params (for v3000 / v5000 under *_mix names).
"""
from __future__ import annotations

import json
import pathlib
import shutil
from dataclasses import asdict

import networkx as nx
import numpy as np
import pandas as pd

import build_snap_benchmark as bsb


def compute_global_basis(train, nodes, tasks, cfg):
    """Return (d_norm_global, aff_max_global) over the given node set."""
    task_lat = tasks.latitude.to_numpy(float)
    task_lon = tasks.longitude.to_numpy(float)
    grouped = {int(uid): g for uid, g in train[train.user_id.isin(nodes)].groupby("user_id")}
    all_min = []
    all_aff = []
    for uid in nodes:
        g = grouped[int(uid)]
        d = bsb.haversine_matrix(g.latitude.to_numpy(float), g.longitude.to_numpy(float), task_lat, task_lon)
        all_min.append(d.min(axis=0))
        all_aff.append(np.exp(-d / cfg.affinity_tau_km).sum(axis=0))
    min_dist_all = np.vstack(all_min)
    raw_aff_all = np.vstack(all_aff)
    d_norm = float(np.quantile(min_dist_all, cfg.distance_quantile))
    d_norm = max(d_norm, 1e-6)
    aff_max = float(raw_aff_all.max())
    return d_norm, aff_max


def build_task_matrices_shared(train, nodes, tasks, cfg, d_norm_global, aff_max_global):
    task_lat = tasks.latitude.to_numpy(float)
    task_lon = tasks.longitude.to_numpy(float)
    q = np.zeros((len(nodes), len(tasks)), dtype=np.float32)
    raw_aff = np.zeros_like(q)
    min_dist_all = np.zeros_like(q)
    grouped = {int(uid): g for uid, g in train[train.user_id.isin(nodes)].groupby("user_id")}
    for i, uid in enumerate(nodes):
        g = grouped[int(uid)]
        d = bsb.haversine_matrix(g.latitude.to_numpy(float), g.longitude.to_numpy(float), task_lat, task_lon)
        min_d = d.min(axis=0)
        min_dist_all[i] = min_d
        raw_aff[i] = np.exp(-d / cfg.affinity_tau_km).sum(axis=0)
    q[:] = np.clip(1.0 - min_dist_all / max(d_norm_global, 1e-6), 0.0, 1.0)
    affinity = raw_aff / max(aff_max_global, 1e-12)
    rewards = np.ones(len(tasks), dtype=np.float32)
    affinity = np.clip(affinity * rewards[None, :], 0.0, 1.0).astype(np.float32)
    density = (min_dist_all <= cfg.demand_radius_km).sum(axis=0).astype(float)
    dn = (density - density.min()) / max(density.max() - density.min(), 1e-12)
    demands = (cfg.demand_min + (cfg.demand_max - cfg.demand_min) * dn).astype(np.float32)
    return q, affinity, rewards, demands


def pick_mixed_workers(fq, fa, G, count, alpha=0.5):
    n_nodes = fq.shape[0]
    quality = np.max(fq * fa, axis=1).astype(float)
    degree = np.zeros(n_nodes, dtype=float)
    for u in G.nodes():
        i = int(u)
        if 0 <= i < n_nodes:
            degree[i] = G.degree(u)

    def norm(x):
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / max(hi - lo, 1e-12)

    score = alpha * norm(quality) + (1.0 - alpha) * norm(degree)
    return np.sort(np.argsort(-score)[:count].astype(int))


def write_shared_dataset(cfg, graph, train, test, tasks, d_norm_global,
                         aff_max_global, size, out_root, num_workers=None):
    nodes = sorted(int(x) for x in graph.nodes())
    mapping = pd.DataFrame({"user_id": nodes, "node_index": np.arange(len(nodes), dtype=int)})
    node_index = dict(zip(mapping.user_id, mapping.node_index))
    stats = bsb.user_statistics(train, nodes, graph)
    q, affinity, rewards, demands = build_task_matrices_shared(
        train, nodes, tasks, cfg, d_norm_global, aff_max_global)
    # MIXED candidate pool instead of stratified_workers
    count = cfg.num_workers if num_workers is None else int(num_workers)
    workers_ui = pick_mixed_workers(q, affinity, graph, count, alpha=0.5)
    worker_indices = np.array([node_index[nodes[i]] for i in workers_ui], dtype=np.int64)
    weights = bsb.directed_weights(graph, nodes, stats, cfg)
    weights["src_index"] = weights.src_user_id.map(node_index).astype(int)
    weights["dst_index"] = weights.dst_user_id.map(node_index).astype(int)

    out = pathlib.Path(out_root) / f"{cfg.dataset}_v{size}_seed{cfg.seed}_shared"
    out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / "user_mapping.csv", index=False)
    stats.to_csv(out / "user_statistics.csv", index=False)
    tasks.to_csv(out / "tasks.csv", index=False)
    weights.to_csv(out / "directed_social_edges.csv", index=False)
    pd.DataFrame({"user_id": [nodes[i] for i in workers_ui], "node_index": worker_indices}).to_csv(
        out / "candidate_workers.csv", index=False)
    cols = ["user_id", "timestamp", "latitude", "longitude", "location_id"]
    train[train.user_id.isin(nodes)][cols].to_csv(out / "train_checkins.csv", index=False)
    test[test.user_id.isin(nodes)][cols].to_csv(out / "test_checkins.csv", index=False)
    np.savez_compressed(
        out / "benchmark_arrays.npz",
        edge_index=weights[["src_index", "dst_index"]].to_numpy(np.int64).T,
        edge_weight=weights.weight.to_numpy(np.float32),
        worker_indices=worker_indices,
        task_locations=tasks[["latitude", "longitude"]].to_numpy(np.float32),
        task_rewards=rewards,
        task_demands=demands,
        q_matrix=q,
        a_matrix=affinity,
    )
    audit = {
        "dataset": cfg.dataset,
        "graph_nodes": len(nodes),
        "undirected_edges": graph.number_of_edges(),
        "candidate_workers": len(worker_indices),
        "tasks": len(tasks),
        "q_range": [float(q.min()), float(q.max())],
        "a_range": [float(affinity.min()), float(affinity.max())],
        "demand_range": [float(demands.min()), float(demands.max())],
        "shared_task_set": True,
        "shared_a_norm": float(aff_max_global),
        "shared_d_norm": float(d_norm_global),
    }
    (out / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = asdict(cfg) | {
        "graph_size": size,
        "quality_definition": "max(0, 1 - min_haversine_distance / GLOBAL_d_norm)",
        "affinity_definition": "task-normalized via GLOBAL raw_aff max (shared N_max)",
        "demand_definition": "scaled count of graph users within demand_radius_km",
        "candidate_pool": "mixed quality+degree top N",
        "ets_definition": "E[min(realized_quality / demand, 1)]",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def export_env_params(processed_dir, out_dir, top_m=5, dim=64, sim_k=10):
    src = pathlib.Path(processed_dir)
    out = pathlib.Path(out_dir)
    env = out / "env_params"
    model = out / "model_inputs"
    env.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    arrays = np.load(src / "benchmark_arrays.npz")
    mapping = pd.read_csv(src / "user_mapping.csv")
    stats = pd.read_csv(src / "user_statistics.csv")
    tasks = pd.read_csv(src / "tasks.csv")
    workers = pd.read_csv(src / "candidate_workers.csv")
    full_q = np.asarray(arrays["q_matrix"], dtype=np.float32)
    full_a = np.asarray(arrays["a_matrix"], dtype=np.float32)
    edge_index = np.asarray(arrays["edge_index"], dtype=np.int64)
    edge_pairs = edge_index.T if edge_index.shape[0] == 2 else edge_index
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
    import export_processed_to_gkd_inputs as eta
    build_worker_features, build_task_features, row_topk_similarity = (
        eta.build_worker_features, eta.build_task_features, eta.row_topk_similarity)
    wf = build_worker_features(stats, worker_user_ids, dim)
    tf = build_task_features(tasks, demands, rewards, dim)
    np.savetxt(model / "worker_features.txt", wf, fmt="%.8f")
    np.savetxt(model / "task_features.txt", tf, fmt="%.8f")
    local = {int(node): i for i, node in enumerate(worker_indices)}
    social = np.zeros((len(worker_indices), len(worker_indices)), dtype=np.float32)
    for (u, v), w in zip(edge_pairs, edge_weight):
        if int(u) in local and int(v) in local:
            social[local[int(u)], local[int(v)]] = float(w)
    np.savetxt(model / "social_adj.txt", social, fmt="%.8f")
    top_m = min(top_m, q.shape[1])
    top = np.argpartition(-q, top_m - 1, axis=1)[:, :top_m]
    wt = np.stack([np.repeat(np.arange(q.shape[0]), top_m), top.reshape(-1)], axis=1)
    np.savetxt(model / "hetero_edge_index.txt", wt, fmt="%d")
    np.savetxt(model / "worker_sim_adj.txt", row_topk_similarity(wf, sim_k), fmt="%.8f")
    np.savetxt(model / "task_sim_adj.txt", row_topk_similarity(tf, sim_k), fmt="%.8f")


def main() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    cfg = bsb.load_config(root / "configs" / "data_gowalla.yaml")
    import dataclasses
    cfg = dataclasses.replace(cfg, raw_dir=str((root / cfg.raw_dir).resolve()))
    edges_path, checkins_path = bsb.locate_raw(cfg)
    print("Reading raw SNAP...")
    edges = bsb.read_edges(edges_path)
    checkins = bsb.read_checkins(checkins_path)
    train, test = bsb.temporal_split(checkins, cfg.temporal_train_ratio)
    user_agg = train.groupby("user_id").agg(checkins=("location_id", "size"), unique_pois=("location_id", "nunique"))
    eligible = set(user_agg[(user_agg.checkins >= cfg.min_checkins_per_user) &
                            (user_agg.unique_pois >= cfg.min_unique_pois_per_user)].index.astype(int))
    max_size = max(cfg.graph_sizes)
    activity = user_agg.checkins
    max_graph = bsb.sample_connected_graph(edges, eligible, max_size, activity, cfg.seed)
    nodes_max = sorted(int(x) for x in max_graph.nodes())
    # SHARED task set from the largest graph nodes
    shared_tasks = bsb.choose_tasks(train, set(nodes_max), cfg)
    # SHARED global normalization basis computed on the largest graph (Q1=A)
    print("Compute shared global basis on v5000...")
    d_norm_global, aff_max_global = compute_global_basis(train, nodes_max, shared_tasks, cfg)
    print(f"  d_norm_global={d_norm_global:.3f}  aff_max_global={aff_max_global:.3f}")

    # v3000 用 300 候选，v5000 用 500 候选（更大图 → 更多高质量候选）
    worker_counts = {3000: 300, 5000: 500}
    for size in sorted(cfg.graph_sizes):
        if size == max_size:
            graph = max_graph
        else:
            sub_edges = nx.to_pandas_edgelist(max_graph).rename(columns={"source": "src", "target": "dst"})
            graph = bsb.sample_connected_graph(sub_edges, set(max_graph.nodes()), size, activity, cfg.seed + size)
        out = write_shared_dataset(cfg, graph, train, test, shared_tasks,
                                   d_norm_global, aff_max_global, size, root / "data" / "processed",
                                   num_workers=worker_counts.get(int(size), cfg.num_workers))
        print(f"Built processed: {out}")
        # export to env_params dir
        exp_root = root / "data" / "experiments" / f"gowalla_v{size}_seed42_shared"
        export_env_params(out, exp_root)
        print(f"Exported env_params: {exp_root}")


if __name__ == "__main__":
    main()