# GKD-Recruiter

GKD-Recruiter: A Graph Neural Network and Distillation-based Framework for Worker Recruitment in Spatial Crowdsourcing with Social Networks

## Overview

This project implements GKD-Recruiter, a novel approach for recruiting workers in spatial crowdsourcing scenarios that leverages social network information. The method combines Graph Neural Networks (GNNs) with knowledge distillation to optimize worker selection based on quality potential, task affinity, and social influence.

## Key Features

- **Heterogeneous Graph Modeling**: Models workers, tasks, and social relationships using heterogeneous graphs
- **Quality Potential Estimation**: Uses spatial distance to estimate worker quality for tasks
- **Task Affinity Learning**: Incorporates historical visit patterns and task rewards
- **Social Influence Propagation**: Leverages social network diffusion for better recruitment
- **Knowledge Distillation**: Distills knowledge from complex GNN models to efficient recruitment strategies

## Dataset

The project uses the Gowalla dataset for evaluation, which contains real-world check-in data from location-based social networks.

### Data Structure

- **Social Graph**: User-user social connections
- **Spatial Data**: Worker and task locations in 2D space
- **Task Attributes**: Rewards, demands, and features
- **Worker Attributes**: Quality potentials, affinities, and features

### Generated Sample Data

Synthetic data can be generated using `data/data_gen_2.py`, which creates:

- Node features (3000 users × 64 dimensions)
- Edge indices and weights for social graph
- Worker and task features
- Heterogeneous graph structures
- Similarity matrices and quality metrics

Data is saved in `data/sample/` as `.txt` files.

## Code Structure

```
code/
├── README.md
├── RainbowGD.ipynb          # Main implementation (Colab notebook)
├── RainbowGD.py             # Main Python script
├── baselines-RainbowGD/     # Baseline methods
│   ├── IMBaseline/          # Influence Maximization baselines
│   └── crowdsensingBaseline/# Crowdsensing-specific baselines
├── data/                    # Data generation and processing
│   ├── data_gen.py          # Original data generator
│   ├── data_gen_2.py        # Enhanced data generator
│   ├── Gowalla/             # Real dataset
│   └── sample/              # Generated sample data
├── ec_func.py               # Evaluation functions
├── rl_ec.py                 # Reinforcement learning components
└── random_base.py           # Random baseline
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GaoYucen/GKD-Recruiter.git
cd GKD-Recruiter
```

2. Install dependencies:
```bash
conda create -n py11 python=3.11
conda activate py11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install networkx numpy matplotlib wandb
```

## Usage

### Data Generation

Generate synthetic data:
```bash
cd data
python data_gen_2.py
```

### Running the Model

Open `RainbowGD.ipynb` in Google Colab or Jupyter Notebook and run the cells.

For local execution:
```bash
python RainbowGD.py
```

### Baselines

Run baseline methods from `baselines-RainbowGD/` directory.

## Evaluation Metrics

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

