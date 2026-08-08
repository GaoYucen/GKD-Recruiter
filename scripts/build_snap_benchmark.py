#!/usr/bin/env python3
"""Build reproducible GKD-Recruiter benchmarks from SNAP check-in data.

Outputs one directory per graph size. All graph nodes receive task quality and
affinity values from their own training-period check-ins; no mean imputation is
used. Friendship edges are expanded into directed arcs with transparent,
behavior-derived asymmetric propagation probabilities.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pathlib
from dataclasses import dataclass, asdict
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import yaml

EARTH_RADIUS_KM = 6371.0088

@dataclass(frozen=True)
class Config:
    dataset: str
    raw_dir: str
    output_root: str
    seed: int
    graph_sizes: list[int]
    num_workers: int
    num_tasks: int
    min_checkins_per_user: int
    min_unique_pois_per_user: int
    min_task_checkins: int
    min_task_unique_users: int
    temporal_train_ratio: float
    candidate_poi_pool: int
    distance_quantile: float
    affinity_tau_km: float
    demand_radius_km: float
    demand_min: float
    demand_max: float
    reward_mode: str
    influence_mode: str
    influence_min: float
    influence_max: float
    influence_mobility_tau_km: float


def load_config(path: pathlib.Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config(**raw)


def locate_raw(cfg: Config) -> tuple[pathlib.Path, pathlib.Path]:
    root = pathlib.Path(cfg.raw_dir)
    if cfg.dataset == "gowalla":
        edge_candidates = [
            root / "loc-gowalla_edges.txt.gz",
            root / "loc-gowalla_edges.txt",
            root / "Gowalla_edges.txt.gz",
            root / "Gowalla_edges.txt",
        ]
        checkin_candidates = [
            root / "loc-gowalla_totalCheckins.txt.gz",
            root / "loc-gowalla_totalCheckins.txt",
            root / "Gowalla_totalCheckins.txt.gz",
            root / "Gowalla_totalCheckins.txt",
        ]
    elif cfg.dataset == "brightkite":
        edge_candidates = [
            root / "loc-brightkite_edges.txt.gz",
            root / "loc-brightkite_edges.txt",
            root / "Brightkite_edges.txt.gz",
            root / "Brightkite_edges.txt",
        ]
        checkin_candidates = [
            root / "loc-brightkite_totalCheckins.txt.gz",
            root / "loc-brightkite_totalCheckins.txt",
            root / "Brightkite_totalCheckins.txt.gz",
            root / "Brightkite_totalCheckins.txt",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")
    edges = next((p for p in edge_candidates if p.exists()), edge_candidates[0])
    checkins = next((p for p in checkin_candidates if p.exists()), checkin_candidates[0])
    missing = [str(p) for p in (edges, checkins) if not p.exists()]
    if missing:
        expected = edge_candidates + checkin_candidates
        raise FileNotFoundError("Missing official SNAP files. Tried: " + ", ".join(str(p) for p in expected))
    return edges, checkins


def compression_for(path: pathlib.Path) -> str | None:
    return "gzip" if path.suffix == ".gz" else None


def read_edges(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", names=["src", "dst"], dtype=np.int64,
                     compression=compression_for(path), comment="#")
    df = df.dropna().astype(np.int64)
    df = df[df.src != df.dst]
    lo = np.minimum(df.src.to_numpy(), df.dst.to_numpy())
    hi = np.maximum(df.src.to_numpy(), df.dst.to_numpy())
    out = pd.DataFrame({"src": lo, "dst": hi}).drop_duplicates(ignore_index=True)
    return out


def read_checkins(path: pathlib.Path) -> pd.DataFrame:
    names = ["user_id", "timestamp", "latitude", "longitude", "location_id"]
    df = pd.read_csv(path, sep="\t", names=names, compression=compression_for(path),
                     dtype={"user_id": "int64", "latitude": "float64", "longitude": "float64", "location_id": "string"},
                     parse_dates=["timestamp"])
    valid = (
        df.user_id.notna() & df.timestamp.notna() & df.location_id.notna()
        & df.latitude.between(-90, 90) & df.longitude.between(-180, 180)
        & ~((df.latitude.abs() < 1e-12) & (df.longitude.abs() < 1e-12))
    )
    df = df.loc[valid].copy()
    df = df.drop_duplicates(["user_id", "timestamp", "location_id"])
    return df


def temporal_split(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < ratio < 1:
        raise ValueError("temporal_train_ratio must be in (0,1)")
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort")
    pos = df.groupby("user_id").cumcount()
    count = df.groupby("user_id")["user_id"].transform("size")
    cut = np.floor(count * ratio).astype(int).clip(lower=1)
    train = df[pos < cut].copy()
    test = df[pos >= cut].copy()
    return train, test


def rank01(values: pd.Series) -> pd.Series:
    if len(values) <= 1:
        return pd.Series(np.ones(len(values)), index=values.index, dtype=float)
    return values.rank(method="average", pct=True).astype(float)


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray,
                     lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Pairwise distances, shape [len(lat1), len(lat2)]."""
    a1 = np.radians(lat1)[:, None]
    o1 = np.radians(lon1)[:, None]
    a2 = np.radians(lat2)[None, :]
    o2 = np.radians(lon2)[None, :]
    da = a2 - a1
    do = o2 - o1
    h = np.sin(da / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin(do / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def sample_connected_graph(edges: pd.DataFrame, eligible: set[int], target: int,
                           activity: pd.Series, seed: int) -> nx.Graph:
    filt = edges.src.isin(eligible) & edges.dst.isin(eligible)
    g = nx.from_pandas_edgelist(edges.loc[filt], "src", "dst")
    if g.number_of_nodes() < target:
        raise ValueError(f"Only {g.number_of_nodes()} eligible graph nodes; need {target}")
    component = max(nx.connected_components(g), key=len)
    g = g.subgraph(component).copy()
    if g.number_of_nodes() < target:
        raise ValueError(f"Largest eligible component has {g.number_of_nodes()} nodes; need {target}")
    rng = np.random.default_rng(seed)
    candidates = np.array(list(g.nodes()), dtype=np.int64)
    scores = activity.reindex(candidates).fillna(0).to_numpy()
    top = candidates[np.argsort(scores)[-min(50, len(candidates)):]]
    start = int(rng.choice(top))
    selected = {start}
    frontier = [start]
    while frontier and len(selected) < target:
        idx = int(rng.integers(len(frontier)))
        u = frontier.pop(idx)
        nbrs = list(g.neighbors(u))
        rng.shuffle(nbrs)
        for v in nbrs:
            if v not in selected:
                selected.add(v)
                frontier.append(v)
                if len(selected) == target:
                    break
        if not frontier and len(selected) < target:
            boundary = [v for v in g.nodes() if v not in selected and any(n in selected for n in g.neighbors(v))]
            if boundary:
                frontier.append(int(rng.choice(boundary)))
    return g.subgraph(selected).copy()


def choose_tasks(train: pd.DataFrame, users: set[int], cfg: Config) -> pd.DataFrame:
    x = train[train.user_id.isin(users)].copy()
    stats = x.groupby("location_id").agg(
        checkins=("user_id", "size"),
        unique_users=("user_id", "nunique"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    ).reset_index()
    stats = stats[(stats.checkins >= cfg.min_task_checkins) &
                  (stats.unique_users >= cfg.min_task_unique_users)]
    stats = stats.sort_values(["unique_users", "checkins"], ascending=False).head(cfg.candidate_poi_pool)
    if len(stats) < cfg.num_tasks:
        raise ValueError(f"Only {len(stats)} eligible POIs; need {cfg.num_tasks}")
    coords = stats[["latitude", "longitude"]].to_numpy(float)
    pop = np.log1p(stats.unique_users.to_numpy(float))
    chosen = [int(np.argmax(pop))]
    min_dist = haversine_matrix(coords[:, 0], coords[:, 1], coords[chosen, 0], coords[chosen, 1]).ravel()
    while len(chosen) < cfg.num_tasks:
        score = min_dist * (0.5 + 0.5 * pop / max(pop.max(), 1e-12))
        score[chosen] = -1
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        d = haversine_matrix(coords[:, 0], coords[:, 1], coords[[nxt], 0], coords[[nxt], 1]).ravel()
        min_dist = np.minimum(min_dist, d)
    tasks = stats.iloc[chosen].reset_index(drop=True)
    tasks.insert(0, "task_index", np.arange(len(tasks), dtype=int))
    return tasks


def user_statistics(train: pd.DataFrame, nodes: list[int], graph: nx.Graph) -> pd.DataFrame:
    x = train[train.user_id.isin(nodes)]
    base = x.groupby("user_id").agg(
        checkins=("location_id", "size"),
        active_days=("timestamp", lambda s: s.dt.date.nunique()),
        unique_pois=("location_id", "nunique"),
        center_lat=("latitude", "median"),
        center_lon=("longitude", "median"),
    )
    counts = x.groupby(["user_id", "location_id"]).size().rename("n").reset_index()
    totals = counts.groupby("user_id").n.transform("sum")
    p = counts.n / totals
    counts["term"] = -p * np.log(np.maximum(p, 1e-12))
    ent = counts.groupby("user_id").term.sum().rename("mobility_entropy")
    base = base.join(ent)
    base["degree"] = pd.Series(dict(graph.degree()))
    base = base.reindex(nodes)
    if base.isna().any().any():
        raise ValueError("Every graph node must have valid training check-ins")
    base["sender_influence"] = rank01(np.log1p(base.checkins) + 0.5 * np.log1p(base.degree))
    base["receiver_susceptibility"] = rank01(np.log1p(base.active_days) + 0.5 * base.mobility_entropy)
    return base.reset_index().rename(columns={"index": "user_id"})


def build_task_matrices(train: pd.DataFrame, nodes: list[int], tasks: pd.DataFrame,
                        cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task_lat = tasks.latitude.to_numpy(float)
    task_lon = tasks.longitude.to_numpy(float)
    q = np.zeros((len(nodes), len(tasks)), dtype=np.float32)
    raw_aff = np.zeros_like(q)
    min_dist_all = np.zeros_like(q)
    grouped = {int(uid): g for uid, g in train[train.user_id.isin(nodes)].groupby("user_id")}
    for i, uid in enumerate(nodes):
        g = grouped[uid]
        d = haversine_matrix(g.latitude.to_numpy(float), g.longitude.to_numpy(float), task_lat, task_lon)
        min_d = d.min(axis=0)
        min_dist_all[i] = min_d
        raw_aff[i] = np.exp(-d / cfg.affinity_tau_km).sum(axis=0)
    d_norm = float(np.quantile(min_dist_all, cfg.distance_quantile))
    d_norm = max(d_norm, 1e-6)
    q[:] = np.clip(1.0 - min_dist_all / d_norm, 0.0, 1.0)
    denom = raw_aff.max(axis=0, keepdims=True)
    affinity = raw_aff / np.maximum(denom, 1e-12)
    if cfg.reward_mode == "uniform":
        rewards = np.ones(len(tasks), dtype=np.float32)
    elif cfg.reward_mode == "inverse_density":
        density = (min_dist_all <= cfg.demand_radius_km).sum(axis=0).astype(float)
        inv = 1.0 / np.sqrt(density + 1.0)
        rewards = ((inv - inv.min()) / max(inv.max() - inv.min(), 1e-12)).astype(np.float32)
        rewards = 0.5 + 0.5 * rewards
    else:
        raise ValueError(f"Unknown reward_mode: {cfg.reward_mode}")
    affinity = np.clip(affinity * rewards[None, :], 0.0, 1.0).astype(np.float32)
    density = (min_dist_all <= cfg.demand_radius_km).sum(axis=0).astype(float)
    dn = (density - density.min()) / max(density.max() - density.min(), 1e-12)
    demands = (cfg.demand_min + (cfg.demand_max - cfg.demand_min) * dn).astype(np.float32)
    return q, affinity, rewards, demands


def stratified_workers(stats: pd.DataFrame, count: int, seed: int) -> np.ndarray:
    if len(stats) < count:
        raise ValueError("Not enough graph nodes for requested candidate workers")
    rng = np.random.default_rng(seed)
    order = stats.sort_values("checkins").user_id.to_numpy(dtype=np.int64)
    bins = np.array_split(order, 3)
    allocations = [count // 3] * 3
    for i in range(count % 3):
        allocations[2 - i] += 1
    chosen = [rng.choice(b, size=n, replace=False) for b, n in zip(bins, allocations)]
    return np.sort(np.concatenate(chosen))


def directed_weights(graph: nx.Graph, nodes: list[int], stats: pd.DataFrame,
                     cfg: Config) -> pd.DataFrame:
    idx = stats.set_index("user_id")
    node_sets = {u: set(graph.neighbors(u)) for u in nodes}
    records = []
    for u, v in graph.edges():
        nu, nv = node_sets[u], node_sets[v]
        union = len(nu | nv)
        jaccard = len(nu & nv) / union if union else 0.0
        d = haversine_matrix(
            np.array([idx.at[u, "center_lat"]]), np.array([idx.at[u, "center_lon"]]),
            np.array([idx.at[v, "center_lat"]]), np.array([idx.at[v, "center_lon"]]),
        )[0, 0]
        mobility = math.exp(-d / cfg.influence_mobility_tau_km)
        relation = 0.7 * mobility + 0.3 * jaccard
        for src, dst in ((u, v), (v, u)):
            directional = float(idx.at[src, "sender_influence"] * idx.at[dst, "receiver_susceptibility"] * relation)
            w = cfg.influence_min + (cfg.influence_max - cfg.influence_min) * directional
            records.append((src, dst, float(np.clip(w, cfg.influence_min, cfg.influence_max)), mobility, jaccard))
    return pd.DataFrame(records, columns=["src_user_id", "dst_user_id", "weight", "mobility_similarity", "neighbor_jaccard"])


def write_dataset(cfg: Config, graph: nx.Graph, train: pd.DataFrame, test: pd.DataFrame,
                  size: int) -> pathlib.Path:
    nodes = sorted(int(x) for x in graph.nodes())
    mapping = pd.DataFrame({"user_id": nodes, "node_index": np.arange(len(nodes), dtype=int)})
    node_index = dict(zip(mapping.user_id, mapping.node_index))
    tasks = choose_tasks(train, set(nodes), cfg)
    stats = user_statistics(train, nodes, graph)
    q, affinity, rewards, demands = build_task_matrices(train, nodes, tasks, cfg)
    workers = stratified_workers(stats, cfg.num_workers, cfg.seed + size)
    weights = directed_weights(graph, nodes, stats, cfg)
    weights["src_index"] = weights.src_user_id.map(node_index).astype(int)
    weights["dst_index"] = weights.dst_user_id.map(node_index).astype(int)
    worker_indices = np.array([node_index[int(u)] for u in workers], dtype=np.int64)

    out = pathlib.Path(cfg.output_root) / f"{cfg.dataset}_v{size}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out / "user_mapping.csv", index=False)
    stats.to_csv(out / "user_statistics.csv", index=False)
    tasks.to_csv(out / "tasks.csv", index=False)
    weights.to_csv(out / "directed_social_edges.csv", index=False)
    pd.DataFrame({"user_id": workers, "node_index": worker_indices}).to_csv(out / "candidate_workers.csv", index=False)
    cols = ["user_id", "timestamp", "latitude", "longitude", "location_id"]
    train[train.user_id.isin(nodes)][cols].to_parquet(out / "train_checkins.parquet", index=False)
    test[test.user_id.isin(nodes)][cols].to_parquet(out / "test_checkins.parquet", index=False)
    np.savez_compressed(out / "benchmark_arrays.npz",
        edge_index=weights[["src_index", "dst_index"]].to_numpy(np.int64).T,
        edge_weight=weights.weight.to_numpy(np.float32),
        worker_indices=worker_indices,
        task_locations=tasks[["latitude", "longitude"]].to_numpy(np.float32),
        task_rewards=rewards,
        task_demands=demands,
        q_matrix=q,
        a_matrix=affinity,
    )
    asym = weights.merge(weights, left_on=["src_user_id", "dst_user_id"], right_on=["dst_user_id", "src_user_id"], suffixes=("", "_rev"))
    asymmetry = np.abs(asym.weight - asym.weight_rev)
    audit = {
        "dataset": cfg.dataset,
        "graph_nodes": len(nodes),
        "undirected_edges": graph.number_of_edges(),
        "directed_arcs": len(weights),
        "candidate_workers": len(workers),
        "tasks": len(tasks),
        "train_checkins": int(train.user_id.isin(nodes).sum()),
        "test_checkins": int(test.user_id.isin(nodes).sum()),
        "q_range": [float(q.min()), float(q.max())],
        "a_range": [float(affinity.min()), float(affinity.max())],
        "demand_range": [float(demands.min()), float(demands.max())],
        "weight_range": [float(weights.weight.min()), float(weights.weight.max())],
        "mean_directional_asymmetry": float(asymmetry.mean()),
        "fraction_asymmetric_pairs": float((asymmetry > 1e-8).mean()),
        "no_missing_values": bool(not mapping.isna().any().any() and not stats.isna().any().any() and not tasks.isna().any().any() and not weights.isna().any().any()),
        "all_probabilities_valid": bool((weights.weight.between(0, 1)).all() and (affinity >= 0).all() and (affinity <= 1).all()),
    }
    (out / "audit_report.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    metadata = asdict(cfg) | {
        "graph_size": size,
        "quality_definition": "max(0, 1 - min_haversine_distance / train_distance_quantile)",
        "affinity_definition": "task-normalized sum exp(-haversine_distance/tau) times normalized reward",
        "demand_definition": "scaled count of graph users within demand_radius_km",
        "influence_definition": "w_min + (w_max-w_min)*sender_influence*receiver_susceptibility*(0.7*mobility_similarity+0.3*neighbor_jaccard)",
        "ets_definition": "E[min(realized_quality / demand, 1)]",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=pathlib.Path)
    args = p.parse_args()
    cfg = load_config(args.config)
    edges_path, checkins_path = locate_raw(cfg)
    print("Reading official SNAP files...")
    edges = read_edges(edges_path)
    checkins = read_checkins(checkins_path)
    train, test = temporal_split(checkins, cfg.temporal_train_ratio)
    user_agg = train.groupby("user_id").agg(checkins=("location_id", "size"), unique_pois=("location_id", "nunique"))
    eligible = set(user_agg[(user_agg.checkins >= cfg.min_checkins_per_user) &
                            (user_agg.unique_pois >= cfg.min_unique_pois_per_user)].index.astype(int))
    max_size = max(cfg.graph_sizes)
    activity = user_agg.checkins
    max_graph = sample_connected_graph(edges, eligible, max_size, activity, cfg.seed)
    for size in sorted(cfg.graph_sizes):
        if size == max_size:
            graph = max_graph
        else:
            sub_edges = nx.to_pandas_edgelist(max_graph).rename(columns={"source": "src", "target": "dst"})
            graph = sample_connected_graph(sub_edges, set(max_graph.nodes()), size, activity, cfg.seed + size)
        out = write_dataset(cfg, graph, train, test, size)
        print(f"Built {out}")

if __name__ == "__main__":
    main()
