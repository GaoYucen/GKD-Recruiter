import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Ensure models module can be imported
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gkd_recruiter import GKDFeatureExtractor

def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalized Ranking (BPR) Loss"""
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

def train_gkd_representation():
    print("🚀 Starting Stage 1: Graph Knowledge Distillation (Representation Learning)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    epochs = 200
    lr = 0.001
    lambda_kd = 0.5 
    h_dim = 64
    
    # ==========================================
    # Load REAL Data from data/model_inputs/
    # ==========================================
    print("📊 Loading real graph structure and features from data/model_inputs/...")
    try:
        # Note: If your txt files are comma-separated, please add delimiter=',' in loadtxt
        w_feats_np = np.loadtxt('data/model_inputs/worker_features.txt')
        t_feats_np = np.loadtxt('data/model_inputs/task_features.txt')
        ww_adj_np = np.loadtxt('data/model_inputs/worker_sim_adj.txt')
        hetero_edges = np.loadtxt('data/model_inputs/hetero_edge_index.txt', dtype=int)
        
        num_w, f_dim = w_feats_np.shape
        num_t = t_feats_np.shape[0]
        
        # Build worker-task heterogeneous graph adjacency matrix
        wt_adj_np = np.zeros((num_w, num_t))
        # Support both edge_index shapes [2, num_edges] and [num_edges, 2]
        if hetero_edges.shape[0] == 2 and hetero_edges.shape[1] > 2:
            hetero_edges = hetero_edges.T
            
        # 🌟 Core Fix: Automatically detect and correct 1-based indexing
        # If the maximum ID in edge list equals total node count, data is 1-indexed
        shift_w = 1 if np.max(hetero_edges[:, 0]) == num_w else 0
        shift_t = 1 if np.max(hetero_edges[:, 1]) == num_t else 0
        
        for e in hetero_edges:
            w_idx = int(e[0]) - shift_w
            t_idx = int(e[1]) - shift_t
            
            # Security check: prevent boundary errors from dirty data
            if 0 <= w_idx < num_w and 0 <= t_idx < num_t:
                wt_adj_np[w_idx, t_idx] = 1.0
            
    except Exception as e:
        print(f"❌ Failed to read data, please check file format or path: {e}")
        return

    raw_w_x = torch.tensor(w_feats_np, dtype=torch.float32).to(device)
    raw_t_x = torch.tensor(t_feats_np, dtype=torch.float32).to(device)
    ww_adj = torch.tensor(ww_adj_np, dtype=torch.float32).to(device)
    wt_adj = torch.tensor(wt_adj_np, dtype=torch.float32).to(device)

    # ==========================================
    # Initialize Model & Training
    # ==========================================
    extractor = GKDFeatureExtractor(feature_dim=f_dim, hidden_dim=h_dim).to(device)
    optimizer = optim.Adam(extractor.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    
    extractor.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Row normalization for adjacency matrices
        ww_degree = ww_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        ww_adj_norm = ww_adj / ww_degree
        wt_degree = wt_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        wt_adj_norm = wt_adj / wt_degree
        
        h_w_s, h_w_rc, h_w_f, h_t_rc = extractor(raw_w_x, raw_t_x, ww_adj_norm, wt_adj_norm)
        
        # Match scores & BPR Loss
        match_scores = torch.matmul(h_w_f, h_t_rc.t())
        pos_mask = (wt_adj > 0)
        neg_mask = (wt_adj == 0)
        
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = match_scores[pos_mask]
            neg_indices = torch.randint(0, neg_mask.sum(), (pos_scores.size(0),))
            neg_scores = match_scores[neg_mask][neg_indices]
            l_cf = bpr_loss(pos_scores, neg_scores)
        else:
            l_cf = torch.tensor(0.0).to(device)

        # KD Loss
        l_kd_social = mse_loss(h_w_s, h_w_f.detach())
        l_kd_task = mse_loss(h_w_rc, h_w_f.detach())
        l_kd = l_kd_social + l_kd_task
        
        total_loss = l_cf + lambda_kd * l_kd
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {total_loss.item():.4f} | L_CF: {l_cf.item():.4f} | L_KD: {l_kd.item():.4f}")

    print("✅ Pre-training complete! Saving distilled high-quality node features...")
    os.makedirs('data/pretrain', exist_ok=True)
    
    extractor.eval()
    with torch.no_grad():
        _, _, final_worker_embeds, final_task_embeds = extractor(raw_w_x, raw_t_x, ww_adj_norm, wt_adj_norm)
        
    torch.save(final_worker_embeds.cpu(), 'data/pretrain/distilled_worker_embeds.pt')
    torch.save(final_task_embeds.cpu(), 'data/pretrain/distilled_task_embeds.pt')
    torch.save(extractor.state_dict(), 'data/pretrain/gkd_extractor_weights.pth')
    
    print("💾 Features saved to data/pretrain/ directory.")

if __name__ == "__main__":
    train_gkd_representation()