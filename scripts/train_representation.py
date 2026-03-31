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
    """
    Bayesian Personalized Ranking (BPR) Loss
    Used for recommendation/matching tasks, encourages positive sample scores to be higher than negative sample scores.
    """
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

def train_gkd_representation():
    print("🚀 Starting Stage 1: Graph Knowledge Distillation (Representation Learning)...")
    
    # 1. Set parameters and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_w, num_t = 3000, 100 # Example using Brightkite as in the paper
    f_dim, h_dim = 32, 64
    epochs = 200
    lr = 0.001
    lambda_kd = 0.5 # KD Loss weight in Eq. 11 of the paper
    
    # 2. Load data (Using Dummy data simulation here; replace with actual data/model_inputs/ loading code)
    print("📊 Loading graph structure and initial features...")
    raw_w_x = torch.randn(num_w, f_dim).to(device)
    raw_t_x = torch.randn(num_t, f_dim).to(device)
    ww_adj = (torch.rand(num_w, num_w) > 0.98).float().to(device) 
    wt_adj = (torch.rand(num_w, num_t) > 0.95).float().to(device)
    
    # 3. Initialize model and optimizer
    extractor = GKDFeatureExtractor(feature_dim=f_dim, hidden_dim=h_dim).to(device)
    optimizer = optim.Adam(extractor.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()
    
    # 4. Start training loop
    extractor.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        # --- Simple row normalization for adjacency matrices before passing to extractor ---
        ww_degree = ww_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        ww_adj_norm = ww_adj / ww_degree

        wt_degree = wt_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        wt_adj_norm = wt_adj / wt_degree
        
        # Forward pass: get features from all views
        h_w_s, h_w_rc, h_w_f, h_t_rc = extractor(raw_w_x, raw_t_x, ww_adj_norm, wt_adj_norm)
        
        # --- Compute L_CF (Prediction loss, e.g., predicting matching score between Worker and Task) ---
        # Demonstrating matching score calculation with dot product:
        match_scores = torch.matmul(h_w_f, h_t_rc.t()) # [num_w, num_t]
        
        # Simple masked positive and negative sampling (simulation)
        pos_mask = (wt_adj > 0)
        neg_mask = (wt_adj == 0)
        
        # Avoid errors from lack of positive/negative samples
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = match_scores[pos_mask]
            # Randomly sample the same number of negative samples
            neg_indices = torch.randint(0, neg_mask.sum(), (pos_scores.size(0),))
            neg_scores = match_scores[neg_mask][neg_indices]
            l_cf = bpr_loss(pos_scores, neg_scores)
        else:
            l_cf = torch.tensor(0.0).to(device)

        # --- Compute L_KD (Knowledge Distillation loss) per Eq. 11 in the paper ---
        # Constrain student views (h_s, h_rc) to approach the teacher fusion view (h_f)
        # Note: use .detach() to prevent gradient backpropagation to Teacher, avoiding model collapse
        l_kd_social = mse_loss(h_w_s, h_w_f.detach())
        l_kd_task = mse_loss(h_w_rc, h_w_f.detach())
        l_kd = l_kd_social + l_kd_task
        
        # Joint optimization target
        total_loss = l_cf + lambda_kd * l_kd
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Total Loss: {total_loss.item():.4f} | L_CF: {l_cf.item():.4f} | L_KD: {l_kd.item():.4f}")

    # 5. Save fixed features after distillation
    print("✅ Pre-training complete! Saving distilled high-quality node features...")
    os.makedirs('data/pretrain', exist_ok=True)
    
    extractor.eval()
    with torch.no_grad():
        _, _, final_worker_embeds, final_task_embeds = extractor(raw_w_x, raw_t_x, ww_adj, wt_adj)
        
    torch.save(final_worker_embeds.cpu(), 'data/pretrain/distilled_worker_embeds.pt')
    torch.save(final_task_embeds.cpu(), 'data/pretrain/distilled_task_embeds.pt')
    torch.save(extractor.state_dict(), 'data/pretrain/gkd_extractor_weights.pth')
    
    print("💾 Features saved to data/pretrain/ directory. RL stage will load these features directly.")

if __name__ == "__main__":
    train_gkd_representation()
