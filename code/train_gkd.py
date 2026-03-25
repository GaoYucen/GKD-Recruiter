import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import networkx as nx

# 导入模型与环境 (请确保路径正确)
from gkd_env import GKDEnv
from gkd_recruiter import GKDRecruiterModel

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"⚡️ 当前计算后端: {device}")

class GraphReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, dyn_w_feat, dyn_t_feat, action, reward, next_dyn_w, next_dyn_t, done):
        self.buffer.append((dyn_w_feat, dyn_t_feat, action, reward, next_dyn_w, next_dyn_t, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        w_f, t_f, act, rew, nw_f, nt_f, don = zip(*batch)
        # 将整个 Batch 一次性转为 Tensor，这是提速的关键
        return (torch.stack(w_f).to(device), torch.stack(t_f).to(device), 
                torch.tensor(act, device=device), torch.tensor(rew, dtype=torch.float32, device=device), 
                torch.stack(nw_f).to(device), torch.stack(nt_f).to(device), 
                torch.tensor(don, dtype=torch.float32, device=device))
    
    # 🌟 请在这里补上这两行：
    def __len__(self):
        return len(self.buffer)

class GKDGraphWrapper:
    def __init__(self, env):
        self.env = env
        self.num_nodes = env.G.number_of_nodes()
        self.num_tasks = env.num_tasks
        # 预先转为 Tensor 并固定在 device 上，避免每一轮都转换
        self.ww_adj = torch.tensor(nx.to_numpy_array(env.G), dtype=torch.float32, device=device)
        self.wt_adj_full = torch.zeros(self.num_nodes, self.num_tasks, device=device)
        self.wt_adj_full[env.worker_indices] = torch.tensor(env.q_matrix * env.a_matrix, dtype=torch.float32, device=device)
        self.base_node_feat = torch.randn(self.num_nodes, 16, device=device)
        self.base_t_feat = torch.randn(self.num_tasks, 16, device=device)

    def get_dynamic_graph_state(self):
        node_selected = torch.zeros(self.num_nodes, 1, device=device)
        for w_id, _ in self.env.selected_seeds:
            node_selected[w_id] = 1.0
        t_progress = torch.zeros(self.num_tasks, 1, device=device)
        for _, t_id in self.env.selected_seeds:
            t_progress[t_id] += 0.05
        return torch.cat([self.base_node_feat, node_selected], dim=1), torch.cat([self.base_t_feat, t_progress], dim=1)


from torch.utils.data import TensorDataset, DataLoader

def pretrain_with_distillation(model, wrapper, worker_idx_tensor, data_path='data/expert_data.pt', epochs=15, batch_size=4):
    print(f"\n🎓 启动 GKD 知识蒸馏预训练 (加载专家数据: {data_path})...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到专家数据 {data_path}，请先运行 generate_expert_data.py")
        
    # 1. 加载数据
    expert_data = torch.load(data_path, weights_only=True)
    dataset = TensorDataset(expert_data['w_states'], expert_data['t_states'], expert_data['actions'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss() # 将 Q 值预测转化为多分类问题
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # 遍历专家经验池
        for b_w, b_t, b_act in dataloader:
            b_w, b_t, b_act = b_w.to(device), b_t.to(device), b_act.to(device)
            
            optimizer.zero_grad()
            # 模型输出全量 Q 值 [Batch, 30000]
            q_values = model(b_w, b_t, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
            
            # 计算交叉熵损失 (让专家选择的动作的 Q 值远大于其他动作)
            loss = criterion(q_values, b_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Distillation Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
    print("✅ 预训练完成！GKD-Recruiter 已具备专家的基础常识。")
    return model

# def train_gkd_agent(episodes=50, batch_size=16, update_every=5):
#     env = GKDEnv(env_dir='data/env_params')
#     wrapper = GKDGraphWrapper(env)
    
#     # 🌟 修复关键：在此处定义全图索引，供模型切片使用
#     worker_idx_tensor = torch.tensor(env.worker_indices, dtype=torch.long, device=device)
    
#     model = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
#     target_net = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
#     target_net.load_state_dict(model.state_dict())
    
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     buffer = GraphReplayBuffer()
    
#     epsilon, epsilon_min, decay = 1.0, 0.1, 0.95
#     gamma = 0.99
#     global_step = 0

# 修改函数签名，接收预训练模型和环境参数
def train_gkd_agent(model, env, wrapper, worker_idx_tensor, episodes=50, batch_size=16, update_every=5, initial_epsilon=0.1):
    print("\n🚀 启动 RL 强化微调 (探索未知领域以超越专家)...")
    
    target_net = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
    target_net.load_state_dict(model.state_dict())
    
    # 降低学习率进行微调
    optimizer = optim.Adam(model.parameters(), lr=5e-5) 
    buffer = GraphReplayBuffer()
    
    # Epsilon 从较低的值开始 (比如 0.1)
    epsilon, epsilon_min, decay = initial_epsilon, 0.01, 0.95
    gamma = 0.99
    global_step = 0

    for ep in range(episodes):
        env.reset()
        w_f, t_f = wrapper.get_dynamic_graph_state()
        done = False
        
        while not done:
            global_step += 1
            # 1. 探索与决策
            if random.random() < epsilon:
                action = random.randint(0, env.num_workers * env.num_tasks - 1)
            else:
                with torch.no_grad():
                    # 增加 Batch 维度进行推理 [1, 3000, 17]
                    q_vals = model(w_f.unsqueeze(0), t_f.unsqueeze(0), wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                    action = torch.argmax(q_vals.view(-1)).item()
            
            _, reward, done, final_ets = env.step(action)
            nw_f, nt_f = wrapper.get_dynamic_graph_state()
            buffer.push(w_f, t_f, action, reward, nw_f, nt_f, done)
            w_f, t_f = nw_f, nt_f

            # 2. 向量化 Batch 更新 (性能核心优化)
            if len(buffer) > batch_size and global_step % update_every == 0:
                b_w, b_t, b_act, b_rew, b_nw, b_nt, b_don = buffer.sample(batch_size)
                
                # 一次性计算全 Batch 的 Q 值 [Batch, 30000]
                q_all = model(b_w, b_t, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                current_q = q_all.gather(1, b_act.unsqueeze(1)).squeeze(1)
                
                # 一次性计算 Target Q 值
                with torch.no_grad():
                    next_q_all = target_net(b_nw, b_nt, wrapper.ww_adj, wrapper.wt_adj_full, worker_idx_tensor)
                    max_next_q = next_q_all.max(dim=1)[0]
                    target_q = b_rew + gamma * max_next_q * (1 - b_don)
                
                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 更新策略
        if ep % 5 == 0:
            target_net.load_state_dict(model.state_dict())
            torch.mps.empty_cache() # 及时释放显存
        
        epsilon = max(epsilon_min, epsilon * decay)
        print(f"   Episode {ep+1:02d} | ETS: {final_ets:.4f} | Epsilon: {epsilon:.2f} | Buffer: {len(buffer)}")

# if __name__ == "__main__":
#     train_gkd_agent()


if __name__ == "__main__":
    # 1. 初始化统一的环境和包装器
    env = GKDEnv(env_dir='data/env_params')
    wrapper = GKDGraphWrapper(env)
    worker_idx_tensor = torch.tensor(env.worker_indices, dtype=torch.long, device=device)
    
    # 2. 实例化一个全新的模型
    gkd_model = GKDRecruiterModel(feature_dim=17, hidden_dim=64).to(device)
    
    # 🌟 阶段一：图知识蒸馏 (Knowledge Distillation)
    gkd_model = pretrain_with_distillation(gkd_model, wrapper, worker_idx_tensor, epochs=15)
    
    # 🌟 阶段二：强化学习微调 (RL Fine-tuning)
    train_gkd_agent(gkd_model, env, wrapper, worker_idx_tensor, episodes=30, initial_epsilon=0.1)
    
    # 保存最终称霸的王者模型！
    torch.save(gkd_model.state_dict(), 'data/gkd_recruiter_final.pth')
    print("🎉 全部训练闭环完成！最佳模型已保存！")