"""Reproducible task-aware IC simulation and ETS metrics."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple
import hashlib
import numpy as np
import networkx as nx


@dataclass
class LiveEdgeWorldCache:
    worlds: Dict[tuple[int, int], Dict[int, Set[int]]] = field(default_factory=dict)

    def get(self, task: int, world_seed: int) -> Dict[int, Set[int]] | None:
        return self.worlds.get((int(task), int(world_seed)))

    def put(self, task: int, world_seed: int, live_edges: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        self.worlds[(int(task), int(world_seed))] = live_edges
        return live_edges

class GKDEvaluator:
    def __init__(self, social_graph:nx.DiGraph, q_matrix:np.ndarray, a_matrix:np.ndarray,
                 task_demands:np.ndarray, worker_indices:np.ndarray, num_simulations:int=100,
                 full_q_matrix:np.ndarray|None=None, full_a_matrix:np.ndarray|None=None, seed:int=42):
        self.G=social_graph; self.nodes=np.asarray(sorted(social_graph.nodes()),dtype=int)
        self.num_nodes=max(int(self.nodes.max())+1, q_matrix.shape[0]); self.num_tasks=q_matrix.shape[1]
        self.task_demands=np.asarray(task_demands,float); self.num_simulations=int(num_simulations); self.seed=int(seed)
        if self.task_demands.shape!=(self.num_tasks,) or np.any(self.task_demands<=0): raise ValueError('task_demands must be positive [T]')
        wi=np.asarray(worker_indices,int)
        if full_q_matrix is None or full_a_matrix is None:
            # Safe fallback: non-candidate nodes can propagate but do not contribute/participate.
            fq=np.zeros((self.num_nodes,self.num_tasks)); fa=np.zeros_like(fq); fq[wi]=q_matrix; fa[wi]=a_matrix
        else: fq=np.asarray(full_q_matrix,float); fa=np.asarray(full_a_matrix,float)
        if fq.shape!=(self.num_nodes,self.num_tasks) or fa.shape!=fq.shape: raise ValueError('full matrices shape mismatch')
        if not (np.isfinite(fq).all() and np.isfinite(fa).all()): raise ValueError('non-finite q/a')
        self.full_q_matrix=np.clip(fq,0,1); self.full_a_matrix=np.clip(fa,0,1)
        self._succ={u:list(self.G.successors(u)) for u in self.G.nodes()}

    @staticmethod
    def _stable_seed(*parts: object) -> int:
        payload = "|".join(str(part) for part in parts).encode("utf-8")
        return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)

    def _sample_live_edge_world(self, task: int, world_seed: int, cache: LiveEdgeWorldCache | None = None) -> Dict[int, Set[int]]:
        task = int(task)
        if cache is not None:
            cached = cache.get(task, world_seed)
            if cached is not None:
                return cached
        rng = np.random.default_rng(self._stable_seed(self.seed, int(world_seed), task))
        live: Dict[int, Set[int]] = {}
        for u in self.G.nodes():
            succ_live: Set[int] = set()
            for v in self._succ.get(u, ()):
                p = float(np.clip(self.G[u][v].get('weight', 0.0) * self.full_a_matrix[int(v), task], 0, 1))
                if rng.random() < p:
                    succ_live.add(int(v))
            live[int(u)] = succ_live
        return cache.put(task, world_seed, live) if cache is not None else live

    def _spread_live_world(self, seeds: Set[int], live_edges: Dict[int, Set[int]]) -> Set[int]:
        active = set(int(w) for w in seeds)
        frontier = list(active)
        while frontier:
            nxt = []
            for u in frontier:
                for v in live_edges.get(int(u), ()):
                    if v not in active:
                        active.add(int(v))
                        nxt.append(int(v))
            frontier = nxt
        return active

    def evaluate(self, seed_pairs:List[Tuple[int,int]], seed:int|None=None)->Dict[str,float]:
        pairs=list(dict.fromkeys((int(w),int(t)) for w,t in seed_pairs))
        for w,t in pairs:
            if w<0 or w>=self.num_nodes or t<0 or t>=self.num_tasks: raise ValueError(f'invalid pair {(w,t)}')
        by_task={t:set() for t in range(self.num_tasks)}
        for w,t in pairs: by_task[t].add(w)
        rng=np.random.default_rng(self.seed if seed is None else seed)
        task_ets=[]; exp_quality=[]
        for t in range(self.num_tasks):
            sats,quals=self._simulate_task(t,by_task[t],rng)
            task_ets.append(float(np.mean(sats))); exp_quality.append(float(np.mean(quals)))
        spread=self._simulate_standard(set(w for w,_ in pairs),rng)
        return {'Effective_Task_Satisfaction':float(np.mean(task_ets)),
                'Mean_Cumulative_Quality':float(np.mean(exp_quality)),
                'Expected_Influence_Spread':float(spread),'Seed_Set_Size':len(pairs),
                'Unique_Seed_Users':len(set(w for w,_ in pairs)),
                'Per_Task_ETS':task_ets}

    def evaluate_task(self, task: int, seeds: Set[int], seed: int | None = None) -> float:
        """Evaluate one task only; useful for exact incremental environment rewards."""
        task = int(task)
        if task < 0 or task >= self.num_tasks:
            raise ValueError(f'invalid task {task}')
        rng = np.random.default_rng(self.seed if seed is None else seed)
        sats, _ = self._simulate_task(task, set(int(w) for w in seeds), rng)
        return float(np.mean(sats))

    def evaluate_task_with_worlds(self, task: int, seeds: Set[int], world_seeds: Iterable[int], cache: LiveEdgeWorldCache | None = None) -> float:
        """Evaluate one task on a fixed bank of shared live-edge worlds."""
        task = int(task)
        worlds = [int(world_seed) for world_seed in world_seeds]
        if task < 0 or task >= self.num_tasks:
            raise ValueError(f'invalid task {task}')
        if not worlds:
            raise ValueError('world_seeds must be non-empty')
        seeds = set(int(w) for w in seeds)
        if not seeds:
            return 0.0
        sats = []
        demand = float(self.task_demands[task])
        for world_seed in worlds:
            live_edges = self._sample_live_edge_world(task, world_seed, cache=cache)
            active = self._spread_live_world(seeds, live_edges)
            quality = float(self.full_q_matrix[list(active), task].sum()) if active else 0.0
            sats.append(min(quality / demand, 1.0))
        return float(np.mean(sats))

    def evaluate_task_delta_with_worlds(self, task: int, old_seeds: Set[int], new_seeds: Set[int], world_seeds: Iterable[int], cache: LiveEdgeWorldCache | None = None) -> float:
        """Paired delta on exactly the same shared worlds."""
        old_value = self.evaluate_task_with_worlds(task, old_seeds, world_seeds, cache=cache)
        new_value = self.evaluate_task_with_worlds(task, new_seeds, world_seeds, cache=cache)
        return float(new_value - old_value)

    def evaluate_task_marginals_with_worlds(
        self,
        task: int,
        base_seeds: Set[int],
        candidate_workers: Iterable[int],
        world_seeds: Iterable[int],
        cache: LiveEdgeWorldCache | None = None,
    ) -> Dict[int, float]:
        """Evaluate marginal ETS gains for many candidate workers on the same task/world bank."""
        task = int(task)
        worlds = [int(world_seed) for world_seed in world_seeds]
        if task < 0 or task >= self.num_tasks:
            raise ValueError(f'invalid task {task}')
        if not worlds:
            raise ValueError('world_seeds must be non-empty')
        base = set(int(w) for w in base_seeds)
        candidates = [int(w) for w in candidate_workers if int(w) not in base]
        if not candidates:
            return {}
        gains = {worker: 0.0 for worker in candidates}
        demand = float(self.task_demands[task])
        for world_seed in worlds:
            live_edges = self._sample_live_edge_world(task, world_seed, cache=cache)
            active = self._spread_live_world(base, live_edges) if base else set()
            base_quality = float(self.full_q_matrix[list(active), task].sum()) if active else 0.0
            base_sat = min(base_quality / demand, 1.0)
            for worker in candidates:
                if worker in active:
                    new_sat = base_sat
                else:
                    expanded = self._spread_live_world({worker}, live_edges)
                    merged = active | expanded
                    quality = float(self.full_q_matrix[list(merged), task].sum()) if merged else 0.0
                    new_sat = min(quality / demand, 1.0)
                gains[worker] += float(new_sat - base_sat)
        inv = 1.0 / max(len(worlds), 1)
        return {worker: float(total * inv) for worker, total in gains.items()}

    def evaluate_task_marginals_with_worlds_fast(
        self,
        task: int,
        base_seeds: Set[int],
        candidate_workers: Iterable[int],
        world_seeds: Iterable[int],
        cache: LiveEdgeWorldCache | None = None,
    ) -> Dict[int, float]:
        """Faster marginal ETS gains by reusing world-level active sets and singleton spreads.

        This avoids recomputing `_spread_live_world({worker}, ...)` for the same worker/world pair
        across repeated calls and shares the base active set per task/world across all candidates.
        """
        task = int(task)
        worlds = [int(world_seed) for world_seed in world_seeds]
        if task < 0 or task >= self.num_tasks:
            raise ValueError(f'invalid task {task}')
        if not worlds:
            raise ValueError('world_seeds must be non-empty')
        base = set(int(w) for w in base_seeds)
        candidates = [int(w) for w in candidate_workers if int(w) not in base]
        if not candidates:
            return {}
        gains = {worker: 0.0 for worker in candidates}
        demand = float(self.task_demands[task])
        singleton_cache: dict[tuple[int, int, int], set[int]] = {}
        for world_seed in worlds:
            live_edges = self._sample_live_edge_world(task, world_seed, cache=cache)
            active = self._spread_live_world(base, live_edges) if base else set()
            active_list = list(active)
            active_quality = float(self.full_q_matrix[active_list, task].sum()) if active_list else 0.0
            base_sat = min(active_quality / demand, 1.0)
            active_mask = np.zeros(self.num_nodes, dtype=bool)
            if active_list:
                active_mask[np.asarray(active_list, dtype=int)] = True
            for worker in candidates:
                if worker in active:
                    gains[worker] += 0.0
                    continue
                cache_key = (task, world_seed, worker)
                if cache_key not in singleton_cache:
                    singleton_cache[cache_key] = self._spread_live_world({worker}, live_edges)
                expanded = singleton_cache[cache_key]
                new_nodes = [node for node in expanded if not active_mask[int(node)]]
                if not new_nodes:
                    new_sat = base_sat
                else:
                    quality = active_quality + float(self.full_q_matrix[np.asarray(new_nodes, dtype=int), task].sum())
                    new_sat = min(quality / demand, 1.0)
                gains[worker] += float(new_sat - base_sat)
        inv = 1.0 / max(len(worlds), 1)
        return {worker: float(total * inv) for worker, total in gains.items()}

    def evaluate_with_worlds(self, seed_pairs: List[Tuple[int, int]], world_seeds: Iterable[int], cache: LiveEdgeWorldCache | None = None) -> Dict[str, float]:
        pairs=list(dict.fromkeys((int(w),int(t)) for w,t in seed_pairs))
        for w,t in pairs:
            if w<0 or w>=self.num_nodes or t<0 or t>=self.num_tasks: raise ValueError(f'invalid pair {(w,t)}')
        worlds = [int(world_seed) for world_seed in world_seeds]
        if not worlds:
            raise ValueError('world_seeds must be non-empty')
        by_task={t:set() for t in range(self.num_tasks)}
        for w,t in pairs: by_task[t].add(w)
        task_ets=[self.evaluate_task_with_worlds(t, by_task[t], worlds, cache=cache) for t in range(self.num_tasks)]
        spread_samples = []
        for world_seed in worlds:
            task_spreads = []
            for t in range(self.num_tasks):
                seeds = by_task[t]
                if not seeds:
                    task_spreads.append(0.0)
                    continue
                live_edges = self._sample_live_edge_world(t, world_seed, cache=cache)
                active = self._spread_live_world(seeds, live_edges)
                task_spreads.append(float(len(active)))
            spread_samples.append(float(np.mean(task_spreads)))
        spread=float(np.mean(spread_samples)) if spread_samples else 0.0
        return {'Effective_Task_Satisfaction':float(np.mean(task_ets)),
                'Mean_Cumulative_Quality':float(np.mean(task_ets)),
                'Expected_Influence_Spread':spread,'Seed_Set_Size':len(pairs),
                'Unique_Seed_Users':len(set(w for w,_ in pairs)),
                'Per_Task_ETS':task_ets}

    def evaluate_with_worlds_delta(self, old_seed_pairs: List[Tuple[int, int]], new_seed_pairs: List[Tuple[int, int]], world_seeds: Iterable[int], cache: LiveEdgeWorldCache | None = None) -> float:
        old_value=self.evaluate_with_worlds(old_seed_pairs, world_seeds, cache=cache)
        new_value=self.evaluate_with_worlds(new_seed_pairs, world_seeds, cache=cache)
        return float(new_value['Effective_Task_Satisfaction'] - old_value['Effective_Task_Satisfaction'])

    def _simulate_task(self,t:int,seeds:Set[int],rng):
        if not seeds: return np.zeros(self.num_simulations),np.zeros(self.num_simulations)
        sats=np.empty(self.num_simulations); quals=np.empty(self.num_simulations); demand=self.task_demands[t]
        for r in range(self.num_simulations):
            active=set(seeds); frontier=list(seeds)
            while frontier:
                nxt=[]
                for u in frontier:
                    for v in self._succ.get(u,()):
                        if v in active: continue
                        p=float(np.clip(self.G[u][v].get('weight',0.0)*self.full_a_matrix[v,t],0,1))
                        if rng.random()<p: active.add(v); nxt.append(v)
                frontier=nxt
            q=float(self.full_q_matrix[list(active),t].sum()); quals[r]=q; sats[r]=min(q/demand,1.0)
        return sats,quals

    def _simulate_standard(self,seeds:Set[int],rng):
        if not seeds:return 0.0
        total=0
        for _ in range(self.num_simulations):
            active=set(seeds); frontier=list(seeds)
            while frontier:
                nxt=[]
                for u in frontier:
                    for v in self._succ.get(u,()):
                        if v not in active and rng.random()<float(np.clip(self.G[u][v].get('weight',0),0,1)):
                            active.add(v); nxt.append(v)
                frontier=nxt
            total+=len(active)
        return total/self.num_simulations
