
# Data Generation and Processing Guide

本项目的数据生成与预处理逻辑分为两个核心部分：模拟物理环境的 Raw Data 以及专为 GKD-Recruiter 模型设计的 Processed Features。

## 1. 目录结构

```
data/
├── raw_data/                # 原始物理世界数据（环境引擎输入）
└── processed_features/      # 预处理后的张量（模型训练输入）
```

## 2. 数据说明

### A. 原始数据 (Raw Data)

该目录包含描述空间众包环境和社交网络的基础数据。这些文件主要用于评估器 (Evaluator)，以确保不同算法在相同的物理约束下进行公平对比。

| 文件名 | 格式 | 说明 |
|--------|------|------|
| edge_index.txt | [E, 2] | 社交网络边索引，表示用户间的有向社交关系。 |
| edge_weight.txt | [E] | 基础社交扩散概率 w_ij，模拟真实世界极低的转化率 (1%~10%)。 |
| q_matrix.txt | [U, T] | 质量潜力矩阵：工人 u 对任务 t 的潜在贡献质量 q_ut ∈[0,1]。 |
| a_matrix.txt | [U, T] | 任务亲和力矩阵：工人对特定任务的参与意愿 a_ut ∈[0,1]。 |
| task_demands.txt | [T] | 任务需求量 d_l，定义了任务达到“饱和陷阱”之前的质量阈值。 |
| worker_indices.txt | [U] | 从全局节点中挑选出的候选工人 ID 集合。 |

### B. 预处理特征 (Processed Features)

该目录包含专为 GKD-Recruiter 架构设计的特征矩阵和辅助结构，用于加速 GNN 和 Reinforcement Learning 的训练过程。

- 初始特征：node_features.txt (3000x64) 和 task_features.txt (100x64)，用于嵌入层初始化。
- 异构图结构：针对 RGCN 模块，预先根据质量潜力选出每个工人的 Top-5 关联任务，存储于 hetero_edge_index.txt。
- 相似度矩阵：针对 Correlation Layer，预先计算好的 worker_sim_adj 与 task_sim_adj，用于捕捉全局关联性。

## 3. 生成逻辑核心约束

为了确保合成数据具有真实科研价值，data_gen_2.py 引入了以下关键物理约束：

- **低社交转化率**：社交边的传播概率被严格约束在 [0.01,0.1]，防止出现非理性的全网大规模扩散。
- **需求饱和机制**：任务需求量设定在 [10,50]，配合归一化后的亲和力矩阵，使得任务不会轻易达到饱和，从而考验算法的精准分配能力。
- **任务感知扩散**：激活概率遵循 p_ij(t_l)=w_ij × a_jl，体现了线上社交向线下参与转化时的折损。

## 4. 如何使用

### 生成合成数据

运行以下命令生成用于模型调试的合成数据集：

```bash
python data/data_gen_2.py
```

### 使用真实数据

若使用 Gowalla 或 Brightkite 真实数据集，请确保将处理后的文件按上述目录结构放置，并使用本项目提供的 Node Mapping 脚本确保 ID 连续性。