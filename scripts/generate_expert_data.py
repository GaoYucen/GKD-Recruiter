import sys
from pathlib import Path
# 自动将项目根目录加入搜索路径
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import torch
import numpy as np
from tqdm import tqdm # 用于显示进度条
import networkx as nx

# 导入已经写好的环境和特征包装器
from models.gkd_env import GKDEnv
from scripts.train_gkd import GKDGraphWrapper

def generate_comgreedy_expert_data(num_episodes=100, save_path='data/expert_data.pt'):
    print(f"🎓 开始收集 ComGreedy 专家数据 (总轮数: {num_episodes})...")
    
    # 初始化环境
    env = GKDEnv(env_dir='data/env_params')
    # 强制将 Wrapper 的设备设为 cpu，避免在生成海量数据时把 MPS 显存撑爆
    from scripts import train_gkd
    train_gkd.device = torch.device("cpu") 
    wrapper = GKDGraphWrapper(env)
    # 由于我们在独立脚本中，手动重置一下 wrapper 的内部 tensor 到 CPU
    wrapper.ww_adj = wrapper.ww_adj.to(train_gkd.device)
    wrapper.wt_adj_full = wrapper.wt_adj_full.to(train_gkd.device)
    wrapper.base_node_feat = wrapper.base_node_feat.to(train_gkd.device)
    wrapper.base_t_feat = wrapper.base_t_feat.to(train_gkd.device)
    
    # 1. 预计算专家的“直觉” (Heuristic Scores)
    # ComGreedy 的核心：工人的社交影响力 (出度权重和) * 工人对任务的亲和力
    print("   => 正在预计算专家启发式得分...")
    worker_influence = np.zeros(env.num_workers)
    for i, w_id in enumerate(env.worker_indices):
        # 计算该候选工人的预期传播范围 (出边权重和)
        worker_influence[i] = sum([data.get('weight', 0.1) for _, _, data in env.G.out_edges(w_id, data=True)])
        
    # [300, 100] 的静态得分矩阵: 影响力 * 质量 * 偏好
    static_expert_score = worker_influence[:, None] * (env.q_matrix * env.a_matrix)
    
    # 用于存储专家轨迹的列表
    expert_w_states = []
    expert_t_states = []
    expert_actions = []
    
    best_ets_list = []

    # 2. 开始让专家在环境里打怪升级
    for ep in tqdm(range(num_episodes), desc="专家演练进度"):
        env.reset()
        done = False
        
        # 为了让专家数据有一定的多样性（防止全是一模一样的死板轨迹）
        # 我们稍微加一点点随机扰动 (0.05 的噪声)
        noise = np.random.uniform(0.95, 1.05, size=static_expert_score.shape)
        current_ep_score = static_expert_score * noise
        
        # 记录每个任务被分配的次数，用于避免“饱和陷阱”
        task_alloc_count = np.zeros(env.num_tasks)
        
        while not done:
            # 获取当前动态图特征
            w_f, t_f = wrapper.get_dynamic_graph_state()
            
            # --- ComGreedy 专家的决策大脑 ---
            # 根据当前任务进度动态打折：如果任务快满了，分数断崖式下跌
            dynamic_discount = np.ones(env.num_tasks)
            for t in range(env.num_tasks):
                # 简单模拟：如果一个任务已经分了 3 个人，收益大减
                if task_alloc_count[t] >= 3: 
                    dynamic_discount[t] = 0.01 
                    
            # 计算当前步的综合得分 [300, 100]
            step_score = current_ep_score * dynamic_discount[None, :]
            
            # 选出全场得分最高的那一对 (worker, task)
            best_flat_idx = np.argmax(step_score)
            best_w_local_idx = best_flat_idx // env.num_tasks
            best_t_idx = best_flat_idx % env.num_tasks
            
            # 记录轨迹 (保存到 CPU 内存)
            expert_w_states.append(w_f.clone())
            expert_t_states.append(t_f.clone())
            expert_actions.append(best_flat_idx)
            
            # 执行动作
            _, _, done, final_ets = env.step(best_flat_idx)
            
            # 更新局部追踪器
            task_alloc_count[best_t_idx] += 1
            # 把被选中的工人得分清零 (不能重复招募同一个人)
            current_ep_score[best_w_local_idx, :] = -1.0 
            
        best_ets_list.append(final_ets)

    print(f"\n📊 专家数据收集完毕！专家平均 ETS: {np.mean(best_ets_list):.4f}")
    
    # 3. 打包并保存到磁盘
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    expert_dataset = {
        'w_states': torch.stack(expert_w_states),   # [5000, 3000, 17]
        't_states': torch.stack(expert_t_states),   # [5000, 100, 17]
        'actions': torch.tensor(expert_actions, dtype=torch.long) # [5000]
    }
    torch.save(expert_dataset, save_path)
    print(f"💾 专家知识已压缩并保存至: {save_path} (数据量: {len(expert_actions)} 步)")

if __name__ == "__main__":
    generate_comgreedy_expert_data(num_episodes=100)