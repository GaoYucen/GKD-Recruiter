# GKD-Recruiter

This is the official implementation of the ICML 2026 paper *GKD-Recruiter: Jointly Modeling Social and Task Heterogeneity for Spatial Crowdsourcing via Graph Knowledge Distillation*.

GKD-Recruiter is a Task-Aware framework for worker recruitment in Spatial Crowdsourcing (SC). It jointly models **social influence** and **worker-task affinity** via a heterogeneous graph, fuses these distinct signals with **Graph Knowledge Distillation**, and navigates the non-submodular seed-selection search space with **Rainbow DQN**.

![Framework](https://github.com/GaoYucen/GKD-Recruiter)

## Key Components

- **Influential GAT (IGAT)**: captures directional influence flow conditioned on edge propagation probabilities.
- **Worker-Task Heterogeneous Modeling (RGCN + Correlation Layer)**: encodes task affinity and intra-type similarities.
- **Graph Knowledge Distillation**: fuses social and task views via a teacher-student mutual-learning objective.
- **Rainbow DQN**: selects worker-task seed pairs with long-term reward optimization, avoiding the "saturation trap" that traps greedy heuristics.

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies: `numpy`, `networkx`, `torch>=2.1`, `pandas`, `PyYAML`, `pyarrow`.

## Data Preparation

The code builds benchmarks from the official SNAP check-in datasets (Gowalla / Brightkite).

```bash
# 1. Download the raw SNAP dataset
python scripts/download_snap_data.py --dataset gowalla

# 2. Build the processed benchmark (writes data/processed/gowalla_v{3000,5000}_seed42/)
python scripts/build_snap_benchmark.py --config configs/data_gowalla.yaml

# 3. Export to the training/evaluation layout for a given scale
python scripts/export_processed_to_gkd_inputs.py \
    --processed-dir data/processed/gowalla_v3000_seed42 \
    --output-root data/experiments/gowalla_v3000_seed42
```

The `data/` directory is gitignored; users are expected to generate datasets locally.

## Training

### 1. Train the representation extractor (IGAT + RGCN + Correlation + Distillation)

```bash
python scripts/train_representation.py \
    --model-input-dir data/experiments/gowalla_v3000_seed42/model_inputs \
    --pretrain-dir data/experiments/gowalla_v3000_seed42/pretrain
```

### 2. Train the Rainbow DQN seed selector

```bash
python scripts/train_gkd.py \
    --env-dir data/experiments/gowalla_v3000_seed42/env_params \
    --pretrain-dir data/experiments/gowalla_v3000_seed42/pretrain \
    --checkpoint-dir checkpoints/a4_rl_v3000
```

## Reproducing the Main Experiments

The main experiment compares GKD-Recruiter against all baselines (`DegGreedy`, `NDD`, `FastSelector`, `ComGreedy`, `CELF`, `TSIM`, `MAIM`, `DQNSelector`) across budgets `K ∈ {25, 50, 75, 100, 150}` (corresponding to Figure 3 and Table 1 in the paper):

```bash
python scripts/run_paired_policy_benchmark.py --config configs/main.yaml
```

Outputs are written to `reports/` (CSV / JSON / Markdown).

## Repository Structure

```text
GKD-Recruiter/
├── LICENSE                  # MIT License
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data_gowalla.yaml    # data construction config
│   └── main.yaml            # main experiment (Figure 3 / Table 1)
├── models/                  # GKD-Recruiter core (IGAT, RGCN, Correlation, Distillation, Rainbow DQN)
│   ├── gkd_recruiter.py     # paper-aligned networks
│   ├── gkd_env.py           # training environment
│   ├── evaluate.py          # ETS evaluation (IC model simulation)
│   ├── candidates.py        # candidate generation
│   ├── action_features.py   # action/state features
│   ├── marginal_q_network.py
│   └── runtime.py
├── baselines/               # baselines (DegGreedy, NDD, FastSelector, ComGreedy, CELF, TSIM, MAIM, DQNSelector)
└── scripts/
    ├── download_snap_data.py
    ├── build_snap_benchmark.py
    ├── build_shared_gowalla.py   # shared-normalization protocol (multi-scale fair comparison)
    ├── export_processed_to_gkd_inputs.py
    ├── train_representation.py
    ├── train_gkd.py
    ├── run_gkd_inference.py
    └── run_paired_policy_benchmark.py
```

## Citation

If you find this code useful, please cite:

```bibtex
@inproceedings{gao2026gkdrecruiter,
  title={GKD-Recruiter: Jointly Modeling Social and Task Heterogeneity for Spatial Crowdsourcing via Graph Knowledge Distillation},
  author={Gao, Yucen and Yu, Zhemeng and Li, Zhuoran and Guo, Jianxiong and Gao, Xiaofeng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).