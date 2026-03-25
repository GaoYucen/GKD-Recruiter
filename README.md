# GKD-Recruiter

GKD-Recruiter: A Graph Neural Network and Distillation-based Framework for Worker Recruitment in Spatial Crowdsourcing with Social Networks

## Overview

This project implements **GKD-Recruiter**, a novel recruitment framework for spatial crowdsourcing that leverages social network information. The method combines **Heterogeneous Graph Neural Networks (GNNs)** with **Knowledge Distillation** (GKD) to optimize worker selection based on quality potential, task affinity, and social influence.

## Key Features

- **Heterogeneous Graph Modeling**: Captures complex relationships between workers, tasks, and social connections.
- **Knowledge Distillation (GKD)**: Distills expert recruitment strategies (e.g., ComGreedy) into an efficient GNN-based reinforcement learning agent.
- **Memory-Optimized GNN**: Custom implementation to handle large-scale social graphs on memory-constrained devices (e.g., Apple Silicon MPS).
- **Scalable Baselines**: Inclusion of various Influence Maximization (IM) and Crowdsensing baselines for comprehensive evaluation.

## Project Structure

```text
GKD-Recruiter/
├── models/             # Core logic and architecture
│   ├── gkd_env.py      # Spatial Crowdsourcing Environment
│   ├── gkd_recruiter.py# Heterogeneous GNN (RGCN + GAT) Model
│   └── evaluate.py     # Evaluation metrics (ETS, Coverage, etc.)
├── scripts/            # Execution entry points
│   ├── generate_expert_data.py # Expert strategy data collection
│   └── train_gkd.py    # Distillation pre-training & RL fine-tuning
├── baselines/          # Comparative algorithms
│   ├── baselines_im.py # CELF, Degree Discount, TSIM
│   ├── maim.py         # Multi-Agent Independent Models
│   └── dqn_selector.py # Vanilla DQN benchmark
├── data/               # Dataset and environment parameters
│   ├── env_params/     # Simulation parameters
│   └── model_inputs/   # Processed features and indices
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GaoYucen/GKD-Recruiter.git
cd GKD-Recruiter
```

2. Environment Setup (Recommended Python 3.11):
```bash
conda create -n py11 python=3.11
conda activate py11
# Install PyTorch (Compatible with MPS/CUDA)
pip install torch torchvision torchaudio 
# Install Dependencies
pip install torch-geometric networkx numpy tqdm
```

## Usage

### 1. Data Preparation
Ensure your environment parameters are placed in `data/env_params/`.

### 2. Expert Knowledge Collection
Collect expert trajectories (ComGreedy) for offline distillation:
```bash
python scripts/generate_expert_data.py
```

### 3. Model Training
Run knowledge distillation followed by reinforcement learning fine-tuning:
```bash
python scripts/train_gkd.py
```

### 4. Evaluation & Baselines
Run baseline comparisons:
```bash
python baselines/baselines_im.py
python baselines/baselines_heuristic.py
```

## License
MIT License


- Recruitment effectiveness
- Social influence coverage
- Computational efficiency
- Quality vs. cost trade-off

## Citation

If you use this code in your research, please cite:

```
@article{gao2024gkd,
  title={GKD-Recruiter: Graph Neural Network and Distillation for Worker Recruitment in Spatial Crowdsourcing},
  author={Gao, Yucen},
  journal={ICML},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Gao Yucen - gyc@example.com

Project Link: https://github.com/GaoYucen/GKD-Recruiter

