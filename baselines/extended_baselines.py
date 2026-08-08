"""Additional auditable baselines for task-aware social recruitment.

The functions in this module are intentionally lightweight and deterministic.
They share one selector interface so aligned 3K/5K experiments can compare
heuristic, IM-style, and random baselines under identical top-m and U_max
constraints.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from models.candidates import build_topm_actions


SeedPair = Tuple[int, int]


@dataclass(frozen=True)
class SelectionResult:
    name: str
    seed_pairs: List[SeedPair]
    selection_time_sec: float
    params: Dict[str, float | int | str]


def topm_candidates(q_workers: np.ndarray, a_workers: np.ndarray, worker_indices: np.ndarray, top_m: int) -> List[SeedPair]:
    """Build worker-task candidates ranked locally by q*a then q."""
    actions, _ = build_topm_actions(q_workers=q_workers, a_workers=a_workers, top_m=top_m)
    return [(int(worker_indices[worker_local]), int(task_idx)) for worker_local, task_idx in actions]


def enforce_constraints(ranked: Iterable[SeedPair], budget: int, u_max: int) -> List[SeedPair]:
    selected: List[SeedPair] = []
    seen: Set[SeedPair] = set()
    load: Dict[int, int] = {}
    for w, t in ranked:
        pair = (int(w), int(t))
        if pair in seen or load.get(pair[0], 0) >= int(u_max):
            continue
        selected.append(pair)
        seen.add(pair)
        load[pair[0]] = load.get(pair[0], 0) + 1
        if len(selected) >= int(budget):
            break
    return selected


def _degree_scores(graph: nx.DiGraph, workers: Sequence[int]) -> Dict[int, float]:
    return {int(w): float(sum(graph[int(w)][v].get("weight", 0.0) for v in graph.successors(int(w)))) for w in workers}


def _rank_percentiles(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    ordered = sorted(values, key=lambda x: (values[x], x))
    denom = max(len(ordered) - 1, 1)
    return {w: i / denom for i, w in enumerate(ordered)}


def select_random(candidates: List[SeedPair], budget: int, u_max: int, seed: int) -> List[SeedPair]:
    rng = np.random.default_rng(int(seed))
    idx = rng.permutation(len(candidates))
    return enforce_constraints((candidates[i] for i in idx), budget, u_max)


def select_quality_greedy(candidates: List[SeedPair], full_q: np.ndarray, full_a: np.ndarray, budget: int, u_max: int) -> List[SeedPair]:
    ranked = sorted(candidates, key=lambda p: (full_q[p[0], p[1]] * full_a[p[0], p[1]], full_q[p[0], p[1]], p[0]), reverse=True)
    return enforce_constraints(ranked, budget, u_max)


def select_deg_greedy(graph: nx.DiGraph, candidates: List[SeedPair], full_q: np.ndarray, budget: int, u_max: int) -> List[SeedPair]:
    ranked = sorted(candidates, key=lambda p: (graph.out_degree(p[0]), full_q[p[0], p[1]], p[0]), reverse=True)
    return enforce_constraints(ranked, budget, u_max)


def select_ndd(graph: nx.DiGraph, candidates: List[SeedPair], worker_indices: np.ndarray, budget: int, u_max: int) -> List[SeedPair]:
    scores = _degree_scores(graph, worker_indices)
    by_worker: Dict[int, List[int]] = {}
    for w, t in candidates:
        by_worker.setdefault(int(w), []).append(int(t))
    ranked: List[SeedPair] = []
    remaining = set(int(w) for w in worker_indices)
    while remaining:
        best = max(remaining, key=lambda w: (scores.get(w, 0.0), w))
        ranked.extend((best, t) for t in by_worker.get(best, []))
        remaining.remove(best)
        for nb in graph.successors(best):
            if nb in remaining:
                scores[nb] = max(0.0, scores.get(nb, 0.0) - graph[best][nb].get("weight", 0.0))
    return enforce_constraints(ranked, budget, u_max)


def select_com_greedy(graph: nx.DiGraph, candidates: List[SeedPair], full_q: np.ndarray, full_a: np.ndarray, budget: int, u_max: int) -> List[SeedPair]:
    scored = []
    for w, t in candidates:
        score = sum(graph[w][nb].get("weight", 0.0) * full_a[nb, t] * full_q[nb, t] for nb in graph.successors(w))
        scored.append((score, full_q[w, t], w, t))
    ranked = [(w, t) for _, _, w, t in sorted(scored, reverse=True)]
    return enforce_constraints(ranked, budget, u_max)


def select_fast_selector(graph: nx.DiGraph, candidates: List[SeedPair], full_q: np.ndarray, full_a: np.ndarray, worker_indices: np.ndarray, budget: int, u_max: int, alpha: float = 0.5) -> List[SeedPair]:
    """FastSelector proxy: degree rank + task relevance rank.

    The paper defines a hybrid degree/trajectory-difference score. The processed
    benchmark exposes q/a rather than raw trajectories, so task relevance is
    proxied by q*a for each worker-task pair and documented in output params.
    """
    degree_rank = _rank_percentiles(_degree_scores(graph, worker_indices))
    raw_task = {(w, t): float(full_q[w, t] * full_a[w, t]) for w, t in candidates}
    task_rank = _rank_percentiles({i: v for i, v in enumerate(raw_task.values())})
    pairs = list(raw_task.keys())
    scored = []
    for i, (w, t) in enumerate(pairs):
        score = float(alpha) * degree_rank.get(w, 0.0) + (1.0 - float(alpha)) * task_rank.get(i, 0.0)
        scored.append((score, raw_task[(w, t)], w, t))
    ranked = [(w, t) for _, _, w, t in sorted(scored, reverse=True)]
    return enforce_constraints(ranked, budget, u_max)


def select_celf_ets(candidates: List[SeedPair], budget: int, u_max: int, evaluate_fn: Callable[[List[SeedPair]], float]) -> List[SeedPair]:
    selected: List[SeedPair] = []
    current = 0.0
    heap: List[Tuple[float, int, int, int]] = []
    for w, t in candidates:
        gain = evaluate_fn([(w, t)])
        heapq.heappush(heap, (-gain, int(w), int(t), 0))
    load: Dict[int, int] = {}
    seen: Set[SeedPair] = set()
    while heap and len(selected) < int(budget):
        neg_gain, w, t, stamp = heapq.heappop(heap)
        if (w, t) in seen or load.get(w, 0) >= int(u_max):
            continue
        if stamp == len(selected):
            selected.append((w, t))
            seen.add((w, t))
            load[w] = load.get(w, 0) + 1
            current += -neg_gain
        else:
            true_gain = evaluate_fn(selected + [(w, t)]) - current
            heapq.heappush(heap, (-true_gain, w, t, len(selected)))
    return selected


def select_tsim_lite(graph: nx.DiGraph, candidates: List[SeedPair], worker_indices: np.ndarray, budget: int, u_max: int, evaluate_fn: Callable[[List[SeedPair]], float], candidate_factor: int = 8) -> List[SeedPair]:
    """Two-stage IM-style selector: NDD pruning followed by delayed ETS forward.

    Stage 1 (pruning): keep a shortlist of the top budget*candidate_factor
    (worker, task) pairs according to the NDD influence-aware ordering.
    Stage 2 (delayed ETS forward): run a CELF-style lazy forward on the
    shortlist using the MC-based ``evaluate_fn`` so that marginal gains over
    the already-selected set are recomputed, matching the paper's two-stage
    TSIM description.
    """
    shortlist_budget = min(len(candidates), max(int(budget) * int(candidate_factor), int(budget)))
    shortlist = select_ndd(graph, candidates, worker_indices, shortlist_budget, max(u_max, candidate_factor))
    if not shortlist:
        return []
    shortlist_workers = {w for w, _ in shortlist}
    worker_degree = _degree_scores(graph, sorted(shortlist_workers))
    shortlist_rank = {pair: idx for idx, pair in enumerate(shortlist)}
    scored: List[tuple[float, int, int]] = []
    for w, t in shortlist:
        structural_bonus = worker_degree.get(int(w), 0.0)
        positional_bonus = float(shortlist_budget - shortlist_rank[(w, t)]) / max(float(shortlist_budget), 1.0)
        scored.append((structural_bonus + 0.05 * positional_bonus, int(w), int(t)))
    ranked = [(w, t) for _, w, t in sorted(scored, reverse=True)]
    if evaluate_fn is None:
        return enforce_constraints(ranked, budget, u_max)
    # Stage 2: delayed ETS forward on the NDD shortlist (CELF-style lazy gain).
    pruned = enforce_constraints(ranked, shortlist_budget, max(u_max, int(candidate_factor)))
    selected: List[SeedPair] = []
    current = 0.0
    heap: List[Tuple[float, int, int, int]] = []
    for w, t in pruned:
        gain = evaluate_fn([(w, t)])
        heapq.heappush(heap, (-gain, int(w), int(t), 0))
    load: Dict[int, int] = {}
    seen: Set[SeedPair] = set()
    while heap and len(selected) < int(budget):
        neg_gain, w, t, stamp = heapq.heappop(heap)
        if (w, t) in seen or load.get(w, 0) >= int(u_max):
            continue
        if stamp == len(selected):
            selected.append((w, t))
            seen.add((w, t))
            load[w] = load.get(w, 0) + 1
            current += -neg_gain
        else:
            true_gain = evaluate_fn(selected + [(w, t)]) - current
            heapq.heappush(heap, (-true_gain, w, t, len(selected)))
    return selected


def run_extended_baseline(name: str, graph: nx.DiGraph, q_workers: np.ndarray, a_workers: np.ndarray, full_q: np.ndarray, full_a: np.ndarray, worker_indices: np.ndarray, budget: int, top_m: int, u_max: int, seed: int, evaluate_fn: Callable[[List[SeedPair]], float] | None = None, celf_limit: int = 600, fast_alpha: float = 0.5, tsim_factor: int = 8) -> SelectionResult:
    start = time.time()
    candidates = topm_candidates(q_workers, a_workers, worker_indices, top_m)
    params: Dict[str, float | int | str] = {"top_m": int(top_m), "u_max": int(u_max)}
    if name == "Random":
        seeds = select_random(candidates, budget, u_max, seed)
    elif name == "QualityGreedy":
        seeds = select_quality_greedy(candidates, full_q, full_a, budget, u_max)
    elif name == "DegGreedy":
        seeds = select_deg_greedy(graph, candidates, full_q, budget, u_max)
    elif name == "NDD":
        seeds = select_ndd(graph, candidates, worker_indices, budget, u_max)
    elif name == "ComGreedy":
        seeds = select_com_greedy(graph, candidates, full_q, full_a, budget, u_max)
    elif name == "FastSelector":
        params["alpha"] = float(fast_alpha)
        params["task_relevance_proxy"] = "q_times_a"
        seeds = select_fast_selector(graph, candidates, full_q, full_a, worker_indices, budget, u_max, fast_alpha)
    elif name in {"CELF-ETS", "TSIM-lite"}:
        if evaluate_fn is None:
            raise ValueError(f"{name} requires evaluate_fn")
        # Keep simulation-heavy baselines practical and comparable by using the
        # strongest q*a candidates for the lazy forward stage.
        pruned = select_quality_greedy(candidates, full_q, full_a, min(len(candidates), int(celf_limit)), max(u_max, int(celf_limit)))
        params["celf_limit"] = int(celf_limit)
        if name == "CELF-ETS":
            seeds = select_celf_ets(pruned, budget, u_max, evaluate_fn)
        else:
            params["candidate_factor"] = int(tsim_factor)
            seeds = select_tsim_lite(graph, pruned, worker_indices, budget, u_max, evaluate_fn, tsim_factor)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return SelectionResult(name=name, seed_pairs=seeds, selection_time_sec=time.time() - start, params=params)
