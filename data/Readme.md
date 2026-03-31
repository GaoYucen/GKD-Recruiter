# Data Directory Structure and Specification

This directory manages all data required for the GKD-Recruiter project. Data is strictly divided into three layers: raw objective data, environment parameters, and model-specific inputs.

## 1. Directory Overview

```
data/
├── source_data/       # Raw objective data (directly collected from the environment)
├── env_params/        # Physical environment parameters (derived ground truth for evaluation)
└── model_inputs/      # Optimized features (preprocessed tensors for neural training)
```

## 2. Detailed Description

### A. Raw Data (source_data/)

This directory stores raw observation data. These files reflect the objective physical state or historical behavior and are not tailored for specific models.

- `raw_edge_index.txt`: Raw social network adjacency list (friend pairs).
- `worker_locations.txt`: 2D geographical coordinates of candidate workers.
- `task_locations.txt`: 2D geographical coordinates of task publication points.
- `raw_visit_freq.txt`: Historical visit frequencies of workers to task areas (check-in behavior).

### B. Environment Parameters (env_params/)

This directory contains the ground truth environment settings that the evaluator (`evaluate.py`) and all baseline algorithms depend on.

- `w_ij.txt`: Social influence probabilities on edges.
- `q_matrix.txt`: Quality potential matrix ($q_{il}$) based on spatial decay.
- `a_matrix.txt`: Task affinity matrix ($a_{il}$) combining rewards and visit frequencies.
- `task_demands.txt`: Saturation thresholds for each task ($d_l$).
- `worker_indices.txt`: Mapping of worker IDs to social network node indices.

### C. Model Inputs (model_inputs/)

Preprocessed tensors designed for the GKD-Recruiter deep architecture (IGAT/RGCN modules) to accelerate training and inference.

- `node_features.txt` / `task_features.txt`: Initial embedding vectors.
- `hetero_edge_index.txt`: Pruned heterogeneous graph structure (Top-5 worker-task associations).
- `worker_sim_adj.txt` / `task_sim_adj.txt`: Global similarity matrices used by the Correlation Layer.

## 3. Data Flow

1. **Source Data**: The raw facts from the real world.
2. **Preprocessing**:
   ```bash
   python data/data_preprocess.py
   ```
   This script reads from `source_data/` and generates both `env_params/` (for physical simulation) and `model_inputs/` (for neural learning).

3. **In-Game Flow**:
   - **Training**: Reads from `model_inputs/`.
   - **Evaluation**: The engine calculates Effective Task Satisfaction (ETS) using parameters in `env_params/`.
