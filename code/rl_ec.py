import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from ec_func import readGraph, influence_spread_ic

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class ECEnv:
    """
    强化学习环境：针对含subarea的EC问题
    """
    def __init__(self, G, n_subarea, n_seed, num_sim=50):
        self.G = G
        self.n_subarea = n_subarea
        self.n_seed = n_seed
        self.num_sim = num_sim
        self.node_list = list(G.nodes)
        self.reset()

    def reset(self):
        self.selected = set()
        self.coverage = np.zeros(self.n_subarea)
        self.steps = 0
        self.available = set(self.node_list)
        self.prev_ec = 0  # 新增：记录上一步EC
        return self._get_state()

    def _get_state(self):
        # 状态编码：已选节点one-hot + 当前subarea覆盖均值
        node_state = np.zeros(len(self.node_list))
        for idx, node in enumerate(self.node_list):
            if node in self.selected:
                node_state[idx] = 1
        # 归一化覆盖
        coverage_norm = (self.coverage - np.mean(self.coverage)) / (np.std(self.coverage) + 1e-6)
        return np.concatenate([node_state, coverage_norm])

    def step(self, action_idx):
        node = self.node_list[action_idx]
        if node in self.selected:
            return self._get_state(), -1.0, self.steps >= self.n_seed, {}
        self.selected.add(node)
        self.available.discard(node)
        self.steps += 1
        ec = influence_spread_ic(self.G, self.selected, self.n_subarea, self.num_sim)
        reward = ec - self.prev_ec  # 奖励为边际提升
        self.prev_ec = ec
        self.coverage = self._estimate_coverage(self.selected)
        done = self.steps >= self.n_seed
        return self._get_state(), reward, done, {}

    def _estimate_coverage(self, S):
        # 返回每个subarea的覆盖均值
        total_quality = np.zeros(self.n_subarea)
        for node in S:
            total_quality += np.array(self.G.nodes[node]['weight'])
        return total_quality / (len(S)+1e-6)

    def action_space(self):
        return len(self.node_list)

    def state_dim(self):
        return len(self.node_list) + self.n_subarea

# 简单DQN网络
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def train_rl_ec(
    nodefilename, edgefilename, n_subarea=100, n_seed=5, episodes=500, lr=1e-3
):
    G = readGraph(nodefilename, edgefilename, n_subarea)
    env = ECEnv(G, n_subarea, n_seed)
    state_dim = env.state_dim()
    action_dim = env.action_space()
    dqn = DQN(state_dim, action_dim).to(device)
    target_dqn = DQN(state_dim, action_dim).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    memory = []
    batch_size = 64
    gamma = 0.98
    epsilon = 1.0
    min_epsilon = 0.01
    decay = 0.995
    tau = 0.005  # 软更新更慢
    best_ec = -float('inf')
    best_model = None

    for ep in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).to(device)
        total_reward = 0
        prev_ec = 0
        for t in range(n_seed):
            mask = np.array([1 if env.node_list[i] not in env.selected else 0 for i in range(action_dim)])
            # 每隔20轮用贪婪策略采样一次
            if (ep+1) % 20 == 0:
                with torch.no_grad():
                    q = dqn(state).cpu().numpy()
                    q[mask == 0] = -1e9
                    action = int(np.argmax(q))
            else:
                if random.random() < epsilon:
                    available_indices = np.where(mask == 1)[0]
                    action = random.choice(available_indices)
                else:
                    with torch.no_grad():
                        q = dqn(state).cpu().numpy()
                        q[mask == 0] = -1e9
                        action = int(np.argmax(q))
            next_state, reward, done, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            memory.append((state, action, reward, next_state, done))
            if len(memory) > 5000:
                memory.pop(0)
            state = next_state
            total_reward += reward
            if done:
                break
            # Double DQN训练
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.stack(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.stack(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)
                # Double DQN
                q_values = dqn(states).gather(1, actions)
                with torch.no_grad():
                    next_actions = dqn(next_states).max(1, keepdim=True)[1]
                    q_next = target_dqn(next_states).gather(1, next_actions)
                    q_target = rewards + gamma * q_next * (~dones)
                loss = nn.functional.mse_loss(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                soft_update(target_dqn, dqn, tau)
        epsilon = max(min_epsilon, epsilon * decay)
        # 每轮结束后用当前策略评估一次
        if (ep+1) % 10 == 0:
            # 贪婪策略评估
            eval_env = ECEnv(G, n_subarea, n_seed)
            eval_state = eval_env.reset()
            eval_state = torch.FloatTensor(eval_state).to(device)
            eval_selected = []
            for t in range(n_seed):
                mask = np.array([1 if eval_env.node_list[i] not in eval_env.selected else 0 for i in range(action_dim)])
                with torch.no_grad():
                    q = dqn(eval_state).cpu().numpy()
                    q[mask == 0] = -1e9
                    action = int(np.argmax(q))
                next_state, reward, done, _ = eval_env.step(action)
                eval_selected.append(eval_env.node_list[action])
                eval_state = torch.FloatTensor(next_state).to(device)
                if done:
                    break
            eval_ec = influence_spread_ic(G, set(eval_selected), n_subarea, num_simulations=50)
            print(f"Episode {ep+1}, total reward: {total_reward:.4f}, epsilon: {epsilon:.3f}, eval EC: {eval_ec:.4f}")
            if eval_ec > best_ec:
                best_ec = eval_ec
                best_model = dqn.state_dict()
    # 用历史最优模型推断
    if best_model is not None:
        dqn.load_state_dict(best_model)
    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    selected = []
    for t in range(n_seed):
        mask = np.array([1 if env.node_list[i] not in env.selected else 0 for i in range(action_dim)])
        with torch.no_grad():
            q = dqn(state).cpu().numpy()
            q[mask == 0] = -1e9
            action = int(np.argmax(q))
        next_state, reward, done, _ = env.step(action)
        selected.append(env.node_list[action])
        state = torch.FloatTensor(next_state).to(device)
        if done:
            break
    print("RL selected seeds:", sorted(selected))
    print("RL selected seed set:", set(selected))
    ec = influence_spread_ic(G, set(selected), n_subarea, num_simulations=100)
    print(f"Estimated EC (100 simulations): {ec:.4f}")

if __name__ == '__main__':
    # 修改为你的数据路径
    train_rl_ec(
        nodefilename='data/input_node_3000_0.txt',
        edgefilename='data/input_edge_3000_0.txt',
        n_subarea=100,
        n_seed=5,
        episodes=500  # 增加训练轮数
    )
