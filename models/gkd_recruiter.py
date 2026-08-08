"""Paper-aligned representation and state-aware seed-selection networks."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def row_norm(a): return a/a.sum(-1,keepdim=True).clamp_min(1e-8)

class DirectionalIGAT(nn.Module):
    """Separate incoming/outgoing attention conditioned on edge probability."""
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.proj=nn.Linear(in_dim,out_dim,bias=False)
        self.att_in=nn.Linear(2*out_dim+1,1,bias=False); self.att_out=nn.Linear(2*out_dim+1,1,bias=False)
        self.gate=nn.Linear(2*out_dim,2*out_dim)
    def _aggregate(self,h,adj,att):
        n=h.size(0); hi=h[:,None,:].expand(n,n,-1); hj=h[None,:,:].expand(n,n,-1)
        e=F.leaky_relu(att(torch.cat([hi,hj,adj[...,None]],-1)).squeeze(-1),.2)
        e=e.masked_fill(adj<=0,-torch.inf); isolated=(adj>0).sum(-1)==0; e[isolated]=0
        alpha=torch.softmax(e,-1); alpha=alpha.masked_fill(isolated[:,None],0.0)
        return F.elu(alpha@h)
    def forward(self,x,adj):
        h=self.proj(x); hout=self._aggregate(h,adj,self.att_out); hin=self._aggregate(h,adj.t(),self.att_in)
        z=torch.cat([hin,hout],-1); g=torch.sigmoid(self.gate(z)); return g*z

class RelationLayer(nn.Module):
    def __init__(self,d):
        super().__init__(); self.self_w=nn.Linear(d,d); self.task_to_worker=nn.Linear(d,d,bias=False); self.worker_to_task=nn.Linear(d,d,bias=False)
    def forward(self,w,t,wt):
        wn=row_norm(wt); tn=row_norm(wt.t())
        return F.relu(self.self_w(w)+wn@self.task_to_worker(t)), F.relu(self.self_w(t)+tn@self.worker_to_task(w))

class CorrelationLayer(nn.Module):
    def __init__(self,d): super().__init__(); self.w=nn.Linear(d,d,bias=False); self.t=nn.Linear(d,d,bias=False)
    def forward(self,w,t,ww,tt): return F.relu(row_norm(ww)@self.w(w)),F.relu(row_norm(tt)@self.t(t))

class Gate2(nn.Module):
    def __init__(self,d): super().__init__(); self.g=nn.Linear(2*d,d)
    def forward(self,a,b):
        g=torch.sigmoid(self.g(torch.cat([a,b],-1))); return g*a+(1-g)*b

class Gate3(nn.Module):
    def __init__(self,d): super().__init__(); self.g=nn.Linear(3*d,3*d)
    def forward(self,a,b,c):
        g=torch.softmax(self.g(torch.cat([a,b,c],-1)).view(*a.shape[:-1],3,a.shape[-1]),-2)
        return g[...,0,:]*a+g[...,1,:]*b+g[...,2,:]*c

class GKDFeatureExtractor(nn.Module):
    def __init__(self,feature_dim,hidden_dim,use_igat=True,use_rel=True,use_corr=True,num_igat_layers=1):
        super().__init__(); self.use_igat=bool(use_igat); self.use_rel=bool(use_rel); self.use_corr=bool(use_corr)
        self.num_igat_layers=max(1,int(num_igat_layers))
        self.wp=nn.Linear(feature_dim,hidden_dim); self.tp=nn.Linear(feature_dim,hidden_dim)
        # Keep the first layer at self.igat so L=1 state_dict keys stay identical to the original
        # single-layer model (backward compatible). Additional layers live in igat_extra.
        self.igat=DirectionalIGAT(hidden_dim,hidden_dim//2)
        self.igat_extra=nn.ModuleList([DirectionalIGAT(hidden_dim,hidden_dim//2) for _ in range(max(0,self.num_igat_layers-1))])
        self.rel=RelationLayer(hidden_dim); self.corr=CorrelationLayer(hidden_dim)
        self.task_gate_w=Gate2(hidden_dim); self.task_gate_t=Gate2(hidden_dim); self.teacher=Gate3(hidden_dim)
    def forward(self,raw_w,raw_t,social_adj,wt_adj,worker_sim=None,task_sim=None):
        w=F.relu(self.wp(raw_w)); t=F.relu(self.tp(raw_t)); worker_sim=social_adj if worker_sim is None else worker_sim
        task_sim=torch.eye(t.size(0),device=t.device) if task_sim is None else task_sim
        if self.use_igat:
            hs=self.igat(w,social_adj)
            for layer in self.igat_extra:
                hs=layer(hs,social_adj)
        else:
            hs=w
        if self.use_rel: wr,tr=self.rel(w,t,wt_adj)
        else: wr,tr=w,t
        if self.use_corr: wc,tc=self.corr(w,t,worker_sim,task_sim)
        else: wc,tc=w,t
        hrc=self.task_gate_w(wr,wc); trc=self.task_gate_t(tr,tc)
        neigh=row_norm((social_adj>0).float())@hrc; hf=self.teacher(hs,hrc,neigh)
        return hs,hrc,hf,trc

class NoisyLinear(nn.Module):
    def __init__(self,inp,out,std=.5):
        super().__init__(); self.inp=inp; self.out=out; self.w_mu=nn.Parameter(torch.empty(out,inp)); self.w_sigma=nn.Parameter(torch.empty(out,inp)); self.b_mu=nn.Parameter(torch.empty(out)); self.b_sigma=nn.Parameter(torch.empty(out)); self.std=std; self.use_noise=True; self.reset_parameters()
    def reset_parameters(self):
        bound=1/self.inp**.5; nn.init.uniform_(self.w_mu,-bound,bound); nn.init.uniform_(self.b_mu,-bound,bound); nn.init.constant_(self.w_sigma,self.std/self.inp**.5); nn.init.constant_(self.b_sigma,self.std/self.out**.5)
    def enable_noise(self): self.use_noise=True
    def disable_noise(self): self.use_noise=False
    def forward(self,x):
        if self.training and self.use_noise: return F.linear(x,self.w_mu+self.w_sigma*torch.randn_like(self.w_sigma),self.b_mu+self.b_sigma*torch.randn_like(self.b_sigma))
        return F.linear(x,self.w_mu,self.b_mu)

class StateAwareDuelingQNetwork(nn.Module):
    def __init__(self,hidden_dim,state_dim):
        super().__init__(); self.pair=nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.ReLU()); self.state=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.ReLU())
        self.value=nn.Sequential(NoisyLinear(hidden_dim,64),nn.ReLU(),NoisyLinear(64,1)); self.adv=nn.Sequential(NoisyLinear(2*hidden_dim,64),nn.ReLU(),NoisyLinear(64,1))
    def encode_pairs(self, worker_embeds, task_embeds, actions):
        """Encode static worker-task pairs once for repeated inference steps."""
        if actions.dim() != 2:
            raise ValueError("actions must have shape [A, 2] when encoding pairs")
        if worker_embeds.dim() == 2:
            w = worker_embeds[actions[:, 0]]
            t = task_embeds[actions[:, 1]]
        elif worker_embeds.dim() == 3:
            w = worker_embeds[:, actions[:, 0]]
            t = task_embeds[:, actions[:, 1]]
        else:
            raise ValueError("worker_embeds must have shape [W,D] or [B,W,D]")
        return self.pair(torch.cat([w, t], -1))

    def set_noise_enabled(self, enabled: bool):
        """Toggle NoisyLinear exploration without changing train/eval mode."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.use_noise = bool(enabled)

    def disable_noise(self):
        self.set_noise_enabled(False)

    def enable_noise(self):
        self.set_noise_enabled(True)

    def forward(self,state,worker_embeds,task_embeds,actions=None,pair_features=None,valid_mask=None):
        # actions [A,2] or [B,A,2], avoiding a dense W*T expansion.
        b=state.size(0); s=self.state(state)
        if actions is None:
            wi=torch.arange(worker_embeds.size(1),device=state.device).repeat_interleave(task_embeds.size(1)); ti=torch.arange(task_embeds.size(1),device=state.device).repeat(worker_embeds.size(1)); actions=torch.stack([wi,ti],-1)
        if actions.dim()==2: actions=actions.unsqueeze(0).expand(b,-1,-1)
        if pair_features is None:
            bi=torch.arange(b,device=state.device)[:,None]; w=worker_embeds[bi,actions[...,0]]; t=task_embeds[bi,actions[...,1]]; p=self.pair(torch.cat([w,t],-1))
        else:
            if pair_features.dim() == 2:
                p = pair_features.unsqueeze(0).expand(b, -1, -1)
            else:
                p = pair_features
            if p.shape[:2] != actions.shape[:2]:
                raise ValueError("pair_features and actions have incompatible shapes")
        v=self.value(s); a=self.adv(torch.cat([s[:,None,:].expand_as(p),p],-1)).squeeze(-1)
        if valid_mask is None:
            valid_mean = a.mean(1, keepdim=True)
        else:
            if valid_mask.shape != a.shape:
                raise ValueError("valid_mask and action scores have incompatible shapes")
            mask = valid_mask.to(dtype=torch.bool, device=a.device)
            valid_count = mask.sum(1, keepdim=True).clamp_min(1)
            valid_mean = a.masked_fill(~mask, 0.0).sum(1, keepdim=True) / valid_count
        q = v + a - valid_mean
        if valid_mask is not None:
            q = q.masked_fill(~mask, torch.finfo(q.dtype).min)
        return q

# Compatibility alias
DuelingQNetwork=StateAwareDuelingQNetwork
