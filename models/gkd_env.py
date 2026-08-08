from __future__ import annotations
import os, numpy as np, networkx as nx, torch
from .evaluate import GKDEvaluator, LiveEdgeWorldCache

class GKDEnv:
    def __init__(self,env_dir='data/env_params',budget_K=50,u_max=1,num_simulations=20,seed=42,incremental=True,reward_scale=1.0,
                 live_edge_worlds=None, live_edge_cache: LiveEdgeWorldCache | None = None):
        self.env_dir=env_dir; self.budget_K=int(budget_K); self.u_max=int(u_max); self.seed=int(seed); self.incremental=bool(incremental); self.reward_scale=float(reward_scale)
        if not np.isfinite(self.reward_scale):
            raise ValueError('reward_scale must be finite')
        load=lambda n,d=float: np.loadtxt(os.path.join(env_dir,n),dtype=d)
        self.q_matrix=np.atleast_2d(load('q_matrix.txt')); self.a_matrix=np.atleast_2d(load('a_matrix.txt'))
        self.task_demands=np.atleast_1d(load('task_demands.txt')); self.worker_indices=np.atleast_1d(load('worker_indices.txt',int))
        e=np.atleast_2d(load('edge_index.txt',int)); w=np.atleast_1d(load('w_ij.txt'))
        self.G=nx.DiGraph(); self.G.add_weighted_edges_from([(int(a),int(b),float(c)) for (a,b),c in zip(e,w)])
        fq=load('full_q_matrix.txt') if os.path.exists(os.path.join(env_dir,'full_q_matrix.txt')) else None
        fa=load('full_a_matrix.txt') if os.path.exists(os.path.join(env_dir,'full_a_matrix.txt')) else None
        self.evaluator=GKDEvaluator(self.G,self.q_matrix,self.a_matrix,self.task_demands,self.worker_indices,num_simulations,fq,fa,seed)
        self.live_edge_worlds=None if live_edge_worlds is None else tuple(int(world) for world in live_edge_worlds)
        if self.live_edge_worlds is not None and len(self.live_edge_worlds)==0:
            raise ValueError('live_edge_worlds must be non-empty when provided')
        self.live_edge_cache=live_edge_cache if live_edge_cache is not None else LiveEdgeWorldCache()
        self.num_workers=len(self.worker_indices); self.num_tasks=len(self.task_demands); self.reset()
    def reset(self):
        self.current_step=0; self.selected_seeds=[]; self.selected_set=set(); self.worker_load=np.zeros(self.num_workers,int)
        self.current_ets=0.; self.task_ets=np.zeros(self.num_tasks,float); return self.state_vector()
    def state_vector(self):
        remain=1-self.current_step/max(self.budget_K,1)
        load=self.worker_load/max(self.u_max,1)
        return torch.tensor(np.concatenate([[remain,self.current_ets],self.task_ets,load]),dtype=torch.float32)
    def valid_action_mask(self,allowed_actions=None):
        mask=np.zeros(self.num_workers*self.num_tasks,dtype=bool)
        if allowed_actions is None: mask[:]=True
        else: mask[np.asarray(allowed_actions,dtype=int)]=True
        for wi in range(self.num_workers):
            if self.worker_load[wi]>=self.u_max: mask[wi*self.num_tasks:(wi+1)*self.num_tasks]=False
        for a in self.selected_set: mask[a]=False
        return mask
    def step(self,action_idx):
        action_idx=int(action_idx); valid=self.valid_action_mask()
        if action_idx<0 or action_idx>=len(valid) or not valid[action_idx]: raise ValueError(f'invalid action {action_idx}')
        old_seeds=list(self.selected_seeds)
        wi,t=divmod(action_idx,self.num_tasks); pair=(int(self.worker_indices[wi]),t)
        self.selected_set.add(action_idx); self.selected_seeds.append(pair); self.worker_load[wi]+=1; self.current_step+=1
        simulation_seed=self.seed+self.current_step
        if self.live_edge_worlds is not None:
            old_task_seeds={w for w, task in old_seeds if task == t}
            new_task_seeds={w for w, task in self.selected_seeds if task == t}
            if self.incremental:
                reward=self.evaluator.evaluate_task_delta_with_worlds(t, old_task_seeds, new_task_seeds, self.live_edge_worlds, cache=self.live_edge_cache) * self.reward_scale
                new_task=self.evaluator.evaluate_task_with_worlds(t, new_task_seeds, self.live_edge_worlds, cache=self.live_edge_cache)
                self.task_ets[t]=new_task; self.current_ets=float(np.mean(self.task_ets))
            else:
                reward=self.evaluator.evaluate_with_worlds_delta(old_seeds, self.selected_seeds, self.live_edge_worlds, cache=self.live_edge_cache) * self.reward_scale
                new_res=self.evaluator.evaluate_with_worlds(self.selected_seeds, self.live_edge_worlds, cache=self.live_edge_cache)
                self.current_ets=float(new_res['Effective_Task_Satisfaction']); self.task_ets=np.asarray(new_res['Per_Task_ETS'],dtype=float)
        elif self.incremental:
            old_task = self.evaluator.evaluate_task(t, {w for w, task in old_seeds if task == t}, seed=simulation_seed)
            new_task = self.evaluator.evaluate_task(t, {w for w, task in self.selected_seeds if task == t}, seed=simulation_seed)
            reward=(new_task-old_task) * self.reward_scale
            self.task_ets[t]=new_task; self.current_ets=float(np.mean(self.task_ets))
        else:
            old_res=self.evaluator.evaluate(old_seeds,seed=simulation_seed)
            new_res=self.evaluator.evaluate(self.selected_seeds,seed=simulation_seed)
            old=float(old_res['Effective_Task_Satisfaction']); new=float(new_res['Effective_Task_Satisfaction']); reward=(new-old) * self.reward_scale; self.current_ets=new; self.task_ets=np.asarray(new_res['Per_Task_ETS'],dtype=float)
        done=self.current_step>=self.budget_K or not self.valid_action_mask().any()
        return self.state_vector(),reward,done,self.current_ets
