from __future__ import annotations
import argparse, os, sys, random
import numpy as np, torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gkd_recruiter import GKDFeatureExtractor
from models.runtime import configure_runtime, maybe_compile

def bpr(scores,positive):
    rows,cols=positive.nonzero(as_tuple=True); neg=torch.randint(scores.size(1),(len(rows),),device=scores.device)
    # ensure sampled negatives are not positive
    for _ in range(5):
        bad=positive[rows,neg]; neg[bad]=torch.randint(scores.size(1),(int(bad.sum()),),device=scores.device)
    return -F.logsigmoid(scores[rows,cols]-scores[rows,neg]).mean()

ABLATION_MAP = {
    'full': dict(use_igat=True, use_rel=True, use_corr=True),
    'wo_igat': dict(use_igat=False, use_rel=True, use_corr=True),
    'wo_rgcn': dict(use_igat=True, use_rel=False, use_corr=True),
    'wo_corr': dict(use_igat=True, use_rel=True, use_corr=False),
    'wo_dist': dict(use_igat=True, use_rel=True, use_corr=True),
}

def main(epochs=200,seed=42,model_input_dir='data/experiments/gowalla_v3000_seed42/model_inputs',pretrain_dir='data/experiments/gowalla_v3000_seed42/pretrain', device='auto', cpu_threads=0, amp=True, compile_model=False, ablation='full', num_igat_layers=1, init_checkpoint=None, save_weights_name='gkd_extractor_weights.pth'):
    if ablation not in ABLATION_MAP:
        raise ValueError(f"unknown ablation '{ablation}', expected one of {sorted(ABLATION_MAP)}")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device, amp_enabled, _ = configure_runtime(device, cpu_threads, amp, compile_model)
    load=lambda p,d=float: np.loadtxt(p,dtype=d)
    wf=load(os.path.join(model_input_dir,'worker_features.txt')); tf=load(os.path.join(model_input_dir,'task_features.txt')); social=load(os.path.join(model_input_dir,'social_adj.txt')); ws=load(os.path.join(model_input_dir,'worker_sim_adj.txt')); ts=load(os.path.join(model_input_dir,'task_sim_adj.txt')); e=np.atleast_2d(load(os.path.join(model_input_dir,'hetero_edge_index.txt'),int))
    wt=np.zeros((len(wf),len(tf)),np.float32); wt[e[:,0],e[:,1]]=1
    W,T,SOC,WS,TS,WT=[torch.tensor(x,dtype=torch.float32,device=device) for x in [wf,tf,social,ws,ts,wt]]
    model=maybe_compile(GKDFeatureExtractor(W.shape[1],64,num_igat_layers=num_igat_layers,**ABLATION_MAP[ablation]).to(device), compile_model)
    if init_checkpoint:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device, weights_only=True))
        print(f'loaded_init_checkpoint={init_checkpoint}')
    opt=torch.optim.Adam(model.parameters(),1e-3,weight_decay=1e-4)
    scaler=torch.amp.GradScaler('cuda', enabled=amp_enabled)
    use_dist = ablation != 'wo_dist'
    for ep in range(epochs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled):
            hs,hr,hf,tr=model(W,T,SOC,WT,WS,TS)
            sf,ss,sr=hf@tr.t(),hs@tr.t(),hr@tr.t()
            cf=bpr(sf,WT.bool())+bpr(ss,WT.bool())+bpr(sr,WT.bool())
            kd=F.mse_loss(hs,hf)+F.mse_loss(hr,hf)+.5*F.mse_loss(hs,hr)
            loss=cf+(.5*kd if use_dist else 0.0)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5); scaler.step(opt); scaler.update()
        if (ep+1)%20==0: print(f'{ep+1}/{epochs} loss={loss.item():.5f} cf={cf.item():.5f} kd={kd.item():.5f}')
    model.eval();
    with torch.no_grad(): _,_,hw,ht=model(W,T,SOC,WT,WS,TS)
    os.makedirs(pretrain_dir,exist_ok=True); torch.save(hw.cpu(),os.path.join(pretrain_dir,'distilled_worker_embeds.pt')); torch.save(ht.cpu(),os.path.join(pretrain_dir,'distilled_task_embeds.pt')); torch.save(model.state_dict(),os.path.join(pretrain_dir,save_weights_name))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=200); p.add_argument('--seed',type=int,default=42); p.add_argument('--model-input-dir',default='data/experiments/gowalla_v3000_seed42/model_inputs'); p.add_argument('--pretrain-dir',default='data/experiments/gowalla_v3000_seed42/pretrain'); p.add_argument('--device',default='auto'); p.add_argument('--cpu-threads',type=int,default=0); p.add_argument('--no-amp',action='store_true'); p.add_argument('--compile',action='store_true'); p.add_argument('--ablation',choices=sorted(ABLATION_MAP),default='full'); p.add_argument('--num-igat-layers',type=int,default=1); p.add_argument('--init-checkpoint',default=None); p.add_argument('--save-weights-name',default='gkd_extractor_weights.pth'); a=p.parse_args(); main(a.epochs,a.seed,a.model_input_dir,a.pretrain_dir,a.device,a.cpu_threads,not a.no_amp,a.compile,a.ablation,a.num_igat_layers,a.init_checkpoint,a.save_weights_name)
