import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HeteroRGCNLayer, self).__init__()
        self.W_worker = nn.Linear(in_dim, out_dim)
        self.W_task = nn.Linear(in_dim, out_dim)
        self.W_ww_rel = nn.Linear(in_dim, out_dim, bias=False) 
        self.W_wt_rel = nn.Linear(in_dim, out_dim, bias=False) 
        self.W_tw_rel = nn.Linear(in_dim, out_dim, bias=False) 

    def forward(self, worker_x, task_x, ww_adj, wt_adj):
        # NOTE: wt_adj is a [3000, 100] 2D matrix, transpose it with .t()
        ww_msg = torch.matmul(ww_adj, self.W_ww_rel(worker_x)) 
        tw_msg = torch.matmul(wt_adj, self.W_tw_rel(task_x))
        wt_msg = torch.matmul(wt_adj.t(), self.W_wt_rel(worker_x)) 
        
        new_worker_x = F.relu(self.W_worker(worker_x) + ww_msg + tw_msg)
        new_task_x = F.relu(self.W_task(task_x) + wt_msg)
        return new_worker_x, new_task_x

class IGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IGATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(size=(out_dim, 1)))
        self.a_dst = nn.Parameter(torch.empty(size=(out_dim, 1)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, worker_x, ww_adj):
        Wh = self.W(worker_x) # [Batch, N, out_dim]
        
        f_src = torch.matmul(Wh, self.a_src) # [Batch, N, 1]
        f_dst = torch.matmul(Wh, self.a_dst) # [Batch, N, 1]
        
        # Use .transpose(-2, -1) to safely flip the last two dimensions for batch compatibility
        e = self.leakyrelu(f_src + f_dst.transpose(-2, -1)) # [Batch, N, N]
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(ww_adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1) # Softmax across the last dimension
        
        h_prime = torch.matmul(attention, Wh) 
        return F.elu(h_prime)

class GKDFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(GKDFeatureExtractor, self).__init__()
        self.worker_proj = nn.Linear(feature_dim, hidden_dim)
        self.task_proj = nn.Linear(feature_dim, hidden_dim)
        self.rgcn = HeteroRGCNLayer(hidden_dim, hidden_dim)
        self.igat = IGATLayer(hidden_dim, hidden_dim)

    def forward(self, raw_worker_x, raw_task_x, ww_adj, wt_adj):
        h_w = F.relu(self.worker_proj(raw_worker_x))
        h_t = F.relu(self.task_proj(raw_task_x))
        h_w_rgcn, h_t_rgcn = self.rgcn(h_w, h_t, ww_adj, wt_adj)
        h_w_igat = self.igat(h_w_rgcn, ww_adj)
        
        final_worker_embed = h_w_igat + h_w 
        final_task_embed = h_t_rgcn + h_t
        return final_worker_embed, final_task_embed
    
class DuelingQNetwork(nn.Module):
    def __init__(self, hidden_dim):
        super(DuelingQNetwork, self).__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, worker_embeds, task_embeds):
        # Memory-optimized implementation: compute Advantage in chunks to save GPU memory
        batch_size = worker_embeds.size(0)
        num_w = worker_embeds.size(1)
        num_t = task_embeds.size(1)
        
        # Mean across node dimensions, preserving Batch dimension
        global_w = worker_embeds.mean(dim=1) 
        global_t = task_embeds.mean(dim=1)
        global_state = torch.cat([global_w, global_t], dim=-1) # [Batch, hidden*2]
        V = self.value_stream(global_state) # [Batch, 1]
        
        # Use broadcasting with an equivalent of Bilinear to avoid expanding large tensors
        # Bilinear(x, y) = x^T W y + b
        
        # Transpose worker_embeds to [Batch * num_w, hidden_dim]
        # Perform weight transformation: W_i * x_batch_w
        # Bilinear has weight (out, in1, in2), here out=1
        
        # Get predefined weight
        weight = self.advantage_bilinear.weight[0] # [hidden_dim, hidden_dim]
        bias = self.advantage_bilinear.bias[0]   # [1]
        
        # (Batch, W, H) @ (H, H) -> (Batch, W, H)
        w_transformed = torch.matmul(worker_embeds, weight) 
        # (Batch, W, H) @ (Batch, H, T) -> (Batch, W, T)
        A_matrix = torch.matmul(w_transformed, task_embeds.transpose(-2, -1)) + bias
        
        A = A_matrix.view(batch_size, -1) # [Batch, W * T]
        
        Q = V + (A - A.mean(dim=1, keepdim=True)) # Maintain batch independence
        return Q

class GKDRecruiterModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super(GKDRecruiterModel, self).__init__()
        self.extractor = GKDFeatureExtractor(feature_dim, hidden_dim)
        self.q_net = DuelingQNetwork(hidden_dim)
    
    def forward(self, raw_node_x, raw_task_x, ww_adj, wt_adj, worker_indices):
        node_embeds, task_embeds = self.extractor(raw_node_x, raw_task_x, ww_adj, wt_adj)
        
        # 🌟 修复此处：必须加上切片的前缀 `:`, 以保证切的是 Node 维度而不是 Batch 维度
        worker_embeds = node_embeds[:, worker_indices, :]
        
        q_values = self.q_net(worker_embeds, task_embeds)
        # 🌟 修复此处：移除 .squeeze()，确保输出严格保持 [Batch, Action] 形状
        return q_values

# 简单的模块测试
if __name__ == "__main__":
    print("🧠 正在测试 GKD-Recruiter 的 Batch 维度大脑...")
    batch, num_w, num_t, f_dim, h_dim = 16, 3000, 100, 17, 64
    candidate_indices = torch.randint(0, num_w, (300,))
    
    # 模拟输入数据 [Batch, N, Dim]
    dummy_w_x = torch.randn(batch, num_w, f_dim)
    dummy_t_x = torch.randn(batch, num_t, f_dim)
    dummy_ww_adj = torch.rand(num_w, num_w) * (torch.rand(num_w, num_w) > 0.95).float() 
    dummy_wt_adj = torch.rand(num_w, num_t)
    
    model = GKDRecruiterModel(feature_dim=f_dim, hidden_dim=h_dim)
    q_vals = model(dummy_w_x, dummy_t_x, dummy_ww_adj, dummy_wt_adj, candidate_indices)
    
    print(f"✅ 成功! 输出 Q 值维度: {q_vals.shape} (预期: [{batch}, {300 * num_t}])")