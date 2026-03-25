
# Data Directory Structure and Specification

This directory stores all data required for the GKD-Recruiter project. To decouple "physical environment" from "model algorithms", data is strictly divided into three layers: raw data layer, environment parameters layer, and model input layer.

## 1. Directory Overview

```
data/
├── source_data/       # Raw objective data (simulating raw information collected from the real world)
├── env_params/        # Physical environment parameters (game environment ground truth calculated based on paper formulas)
└── model_inputs/      # Model preprocessed features (tensors optimized for neural network training)
```

## 2. Detailed Description

### A. Raw Data Layer (source_data/)

This directory simulates raw information directly obtained from LBSN (e.g., Gowalla). These data reflect the objective existence of the physical world, without any processing tailored to tasks or models.

- raw_edge_index.txt: Raw social network friend relationship pairs.
- worker_locations.txt: 2D geographical coordinates of candidate workers in a 10×10 space.
- task_locations.txt: 2D geographical coordinates of task publications.
- raw_visit_freq.txt: Historical visit frequencies of workers to various task areas, reflecting spatial preferences.

### B. Environment Parameters Layer (env_params/)

This directory stores the environment ground truth (Ground Truth) that the evaluator (evaluate.py) and all comparison baselines depend on. These parameters are generated based on the core formulas in the paper.

- w_ij.txt: Base influence probability w_ij on social edges. Simulates extremely low social conversion rates in the real world (0.01∼0.1).
- q_matrix.txt: Quality potential matrix (q_il). Calculated based on spatial distance decay, reflecting the potential contribution of workers to tasks.
- a_matrix.txt: Task affinity matrix (a_il). Calculated and normalized by combining task rewards and visit frequencies, determining the probability of social influence converting to offline participation.
- task_demands.txt: Task demands d_l. Defines the saturation threshold for each task, a key constraint for solving the "saturation trap" problem.
- worker_indices.txt: Set of candidate worker IDs with recruitment value.

### C. Model Input Layer (model_inputs/)

This directory contains preprocessed tensors designed specifically for the GKD-Recruiter deep architecture, aimed at accelerating the computation of IGAT and RGCN modules.

- node_features.txt / task_features.txt: Initial Embedding vectors for neural network input layers.
- hetero_edge_index.txt: Heterogeneous edges pruned for RGCN. Only retains the top-m (Top-5) "worker-task" associations by quality potential to reduce computational complexity.
- worker_sim_adj.txt / task_sim_adj.txt: Cosine similarity matrices used by the Correlation Layer, capturing global spatial/interest associations between nodes.

## 3. Experimental Data Flow

- **Training Phase**: The model reads model_inputs/ to obtain heterogeneous graph structures and initial features for strategy learning.
- **Decision Phase**: The model outputs a set of seed pairs (u,t) based on the current state.
- **Evaluation Phase**: The evaluation engine only reads w, q, a, d from env_params/ for Monte Carlo simulation, calculating the ETS (Effective Task Satisfaction) metric.

```bash
python data/data_gen.py
```

### 使用真实数据

若使用 Gowalla 或 Brightkite 真实数据集，请确保将处理后的文件按上述目录结构放置，并使用本项目提供的 Node Mapping 脚本确保 ID 连续性。