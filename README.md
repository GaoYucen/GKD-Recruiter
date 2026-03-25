# GKD-Recruiter: Graph Knowledge Distillation for Spatial Crowdsourcing

This repository contains the official PyTorch implementation for our paper: **"GKD-Recruiter: Jointly Modeling Social Influence and Physical Affinities for Spatial Crowdsourcing"** (Under Review for ICML 2026).

## 💡 Overview

Recruiting workers via social networks for spatial crowdsourcing presents unique challenges. Traditional Influence Maximization (IM) algorithms often fail due to the **Social-Physical Gap** (influential users may not be physically suitable for the task) and the **Saturation Trap** (tasks have strict capacity limits, rendering the objective function non-submodular).

GKD-Recruiter proposes a novel deep reinforcement learning framework that overcomes these limitations by:

*   **Heterogeneous Graph Convolutional Networks (RGCN)**: Extracting structural features from both worker-worker social graphs and worker-task physical affinity graphs.
*   **Influence Graph Attention Networks (IGAT)**: Anticipating social propagation paths to precisely evaluate expected task participation.
*   **Graph Knowledge Distillation (GKD)**: Leveraging expert heuristic trajectories to bootstrap a Rainbow Dueling DQN agent, efficiently navigating the massive 30,000-dimensional action space and bypassing local optima.

## 📂 Repository Structure

The codebase is modularized for readability and reproducibility:

```text
GKD-Recruiter/
├── data/                       # Datasets and environment parameters
│   ├── env_params/             # Adjacency matrices, task demands, q/a matrices
│   ├── model_inputs/           # Node and task initial features
│   ├── source_data/            # Raw simulated spatial & social data
│   ├── data_gen.py             # Script to synthesize the environment
│   └── Readme.md               # Data description
├── models/                     # Core neural network architectures
│   ├── gkd_recruiter.py        # The RGCN + IGAT + Dueling DQN model
│   ├── gkd_env.py              # RL Environment wrapper for spatial recruiting
│   └── evaluate.py             # Monte Carlo simulator for accurate ETS calculation
├── baselines/                  # Comprehensive baseline algorithms
│   ├── baselines_heuristic.py  # ComGreedy, DegGreedy
│   ├── baselines_im.py         # CELF (Cost-Effective Lazy Forward), NDD
│   ├── dqn_selector.py         # Single-Agent RL
│   └── maim.py                 # Multi-Agent RL
├── scripts/                    # Execution and training scripts
│   ├── generate_expert_data.py # Collect expert trajectories via ComGreedy
│   └── train_gkd.py            # Phase 1: Knowledge Distillation -> Phase 2: RL Fine-tuning
└── README.md                   # You are here!
```

## ⚙️ Dependencies

This project is built with Python 3.x and PyTorch. The code is optimized to run on both CUDA GPUs and Apple Silicon (MPS).

```bash
pip install torch networkx numpy tqdm
```

## 🚀 Getting Started

### 1. Generate Expert Data (Knowledge Distillation)
To overcome the vast action space, GKD-Recruiter utilizes an expert-guided pre-training phase. First, generate the expert trajectories using the strong heuristic algorithm (ComGreedy):

```bash
python scripts/generate_expert_data.py
```
This will simulate 100 episodes and save the dynamic graph states and actions to `data/expert_data.pt`.

### 2. Train GKD-Recruiter (End-to-End)
Execute the main training script. This script automatically performs **Phase 1: Supervised Graph Knowledge Distillation** followed by **Phase 2: Reinforcement Learning Fine-tuning**:

```bash
python scripts/train_gkd.py
```
The script utilizes vectorized batch processing for efficient graph computations. The best model weights will be saved upon completion.

### 3. Run Baselines
To reproduce the baseline experiments and demonstrate the limitations of traditional algorithms (e.g., the Saturation Trap in CELF and the Social-Physical Gap in NDD), run the following scripts:

```bash
# Run Heuristic Baselines
python baselines/baselines_heuristic.py

# Run Traditional Influence Maximization Baselines (CELF, NDD)
python baselines/baselines_im.py

# Run Vanilla RL / Multi-Agent Baselines
python baselines/dqn_selector.py
python baselines/maim.py
```

## 📊 Evaluation Metrics

The primary metric used to evaluate recruiting performance is **ETS (Effective Task Satisfaction)**, which measures the aggregated quality of recruited workers capped by the strict demand (capacity) of each task. Expected Influence Spread (number of activated nodes) is also recorded to illustrate the discrepancy between traditional IM goals and spatial crowdsourcing needs.

## 📄 License
MIT License
