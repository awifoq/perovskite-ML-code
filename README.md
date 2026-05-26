# Perovskite Solar Cell PCE Prediction with Temporal Embedding & Flow Matching

Code accompanying the paper: **"Efficient Perovskite Solar Cell PCE Prediction and Formulation Optimization Based on Temporal Feature Enhancement and Flow Matching Microenvironment"**.

## Overview

This project proposes a unified framework combining temporal feature enhancement and flow matching microenvironment for accurate perovskite solar cell power conversion efficiency (PCE) prediction and formulation optimization. The main contributions include:

1. **Temporal Feature Embedding** — Incorporating temporal information (e.g., publication year, experimental timeline) into feature representations to capture the evolving nature of perovskite research.
2. **Flow Matching Microenvironment** — Modeling the local feature distribution via flow matching to improve generalization and robustness.
3. **Virtual Composition Screening (VCS)** — Enabling efficient exploration of the compositional space for optimal PCE.

## Repository Structure

```
F:.
├── 1.ipynb                         # Data exploration & preprocessing
├── 2.ipynb                         # Main modeling: temporal embedding, flow matching, training & evaluation
├── 3.ipynb                         # VCS optimization, ablation studies, & supplementary experiments
├── check_data.py                   # Compute 5-fold CV statistics (mean ± std) for comparison with paper
├── uniform_time_5cv_ablation.csv   # Ablation study results (baseline 106d feature set)
├── yearly_sequence_data.npz        # Precomputed yearly sequence embeddings
├── csv/
│   ├── perovskite_complete_2014_2026_v2.csv   # Main dataset (7,773 samples, 511 columns)
│   ├── uniform_time_5cv_results.csv            # Baseline 5-fold CV results
│   ├── uniform_time_5cv_temporal_embedding.csv # Temporal embedding enhanced 5-fold CV results
│   └── uniform_split_results.csv               # Random split results
├── model/                          # Saved model weights (excluded from git)
├── .gitignore
└── README.md
```

## Dataset

The dataset `perovskite_complete_2014_2026_v2.csv` is a comprehensive collection of perovskite solar cell experimental data from 2014 to 2026, featuring:

- **7,773** experimental records
- **511** columns covering composition, device architecture, fabrication conditions, and performance metrics
- Key target variable: **PCE** (Power Conversion Efficiency, %)

## Key Results

All experiments use **uniform time-split 5-fold cross-validation** (temporal split to avoid data leakage).

### LightGBM Performance (Mean ± Std over 5 folds)

| Feature Set | Dimensions | R² | RMSE | MAE |
|---|---|---|---|---|
| Baseline (106d) | 106 | ~0.705 | ~3.17 | ~2.30 |
| + Transformer Embedding (234d) | 234 | **0.906 ± 0.008** | **1.79 ± 0.07** | **1.21 ± 0.04** |
| + All Embeddings (490d) | 490 | **0.926 ± 0.007** | **1.58 ± 0.06** | **1.11 ± 0.02** |

### Models Evaluated

- LightGBM
- ExtraTrees
- RandomForest
- XGBoost
- CatBoost

## Usage

### 1. Verify Results

Run the data checking script to reproduce the 5-fold CV statistics:

```bash
python check_data.py
```

### 2. Run Notebooks (Recommended Order)

1. Open `1.ipynb` — Explore dataset, check distributions, preprocess features
2. Open `2.ipynb` — Train models with temporal embedding, evaluate using 5-fold CV
3. Open `3.ipynb` — Run ablation studies, VCS optimization, generate figures

### 3. Dependencies

Core dependencies:
- Python ≥ 3.8
- pandas, numpy
- scikit-learn
- lightgbm
- xgboost
- catboost
- matplotlib, seaborn (for visualization)