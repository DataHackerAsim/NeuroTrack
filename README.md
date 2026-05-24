# NeuroTrack

GNN-based charged particle track reconstruction on the TrackML dataset using a two-stage pipeline: metric-learning hit embeddings followed by an Interaction Network edge classifier.

---

## Overview

NeuroTrack reconstructs particle trajectories from detector hit points by learning which hits belong to the same particle. The pipeline has two trainable stages:

1. **Hit Embedding** — An MLP maps each hit's 24-dimensional feature vector to a 12-dimensional L2-normalised metric-learning space, trained with hinge loss and semi-hard negative mining. Hits from the same particle are pulled together; hits from different particles are pushed apart.

2. **Edge Classification** — A KNN graph is built in embedding space, then a Battaglia-style Interaction Network classifies each candidate edge as same-particle or different-particle, using focal BCE to handle the heavy class imbalance (true edges are 1–5% of candidates).

Track candidates are extracted via connected components on edges above a score threshold, with optional Kalman helix fitting for refinement and outlier rejection.

## Architecture

```
Raw Hits (24-D)
    │
    ▼
┌─────────────────────┐
│  HitEmbedNet (MLP)  │  Metric learning: hinge loss + semi-hard negative mining
│  24-D → 128 → 12-D  │  L2-normalised output
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   KNN Graph Build   │  k=8 nearest neighbours in embedding space
│   (2, E) edges      │  Pure PyTorch (CPU/CUDA compatible)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ InteractionNetwork  │  Battaglia-style relational + object MLPs
│  8 message-passing  │  Focal BCE loss (class imbalance)
│  iterations         │  Gradient checkpointing for VRAM efficiency
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Track Building    │  Threshold edges → connected components
│                     │  Optional: uncertainty-aware Union-Find
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Kalman Helix Fit   │  Optional: chi² arbitration + outlier rejection
└─────────────────────┘
```

## Tracking Modes

| Mode | Method | Use Case |
|------|--------|----------|
| `baseline` | Threshold → connected components | Fast, reproducible |
| `uncertainty` | Score-aware Union-Find with chain protection | Better purity |
| `uncertainty_kalman` | Above + Kalman chi² arbitration | Best physics accuracy |

## Evaluation

Metrics follow the official TrackML competition definition (Amrouche et al., 2019):

- **TrackML Score** — weighted fraction of correctly reconstructed hits (double-majority matching)
- **Track Efficiency** — fraction of true particles reconstructed above threshold
- **Fake Rate** — fraction of predicted tracks not matching any true particle
- **Edge AUC** — area under ROC for the GNN edge classifier
- **Embedding Recall@k** — fraction of same-particle hit pairs within k nearest neighbours

## Repository Structure

```
.
├── src/neurotrack/
│   ├── data/               # Dataset ingestion (TrackML, REDVID, Zenodo formats)
│   ├── models/
│   │   ├── embedding.py    # HitEmbedNet (metric-learning MLP)
│   │   ├── gnn.py          # InteractionNetwork (edge classifier)
│   │   ├── losses.py       # Hinge embedding loss + focal BCE
│   │   └── filter.py       # Kalman filter model
│   ├── graph/
│   │   ├── construction.py # KNN graph building + edge features
│   │   └── truth.py        # Ground-truth edge labelling
│   ├── tracking/
│   │   ├── builder.py      # Connected-components track building
│   │   ├── builder_uncertainty.py  # Score-aware Union-Find
│   │   ├── kalman.py       # Kalman helix fit
│   │   └── arbitrate.py    # Track arbitration
│   ├── eval/
│   │   ├── metrics.py      # TrackML score, efficiency, fake rate, AUC
│   │   ├── physics.py      # Physics-level evaluation
│   │   └── visualization.py
│   ├── train/
│   │   ├── embedding_trainer.py
│   │   └── gnn_trainer.py
│   ├── inference/
│   │   └── pipeline.py     # End-to-end: embedding → KNN → GNN → tracks
│   └── utils/              # Config, logging, seeding, profiling
├── scripts/
│   ├── train_embedding.py  # Train Stage 1
│   ├── train_gnn.py        # Train Stage 2
│   ├── evaluate.py         # End-to-end evaluation
│   ├── infer.py            # Inference on new events
│   └── benchmark.py        # Performance profiling
├── tests/
│   ├── unit/               # 17 unit tests
│   ├── integration/        # Smoke tests
│   └── fixtures/           # Sample TrackML event
└── .gitignore
```

## Data Setup

The TrackML dataset is not included due to size. To download:

```bash
# TrackML (Kaggle / Zenodo)
python scripts/download_zenodo.py --output data/raw

# Ingest and process into parquet shards
python scripts/ingest_all_shards.py --input data/raw --output data/processed/trackml_small
```

## Training

```bash
# Stage 1: Train hit embeddings (metric learning)
python scripts/train_embedding.py \
    --shard data/processed/trackml_small \
    --limit 1000 --epochs 6

# Stage 2: Train GNN edge classifier
python scripts/train_gnn.py \
    --emb-ckpt artifacts/checkpoints/embedding.pt \
    --shard data/processed/trackml_small \
    --limit 1000 --epochs 4 --knn-k 8
```

## Evaluation

```bash
python scripts/evaluate.py \
    --emb-ckpt artifacts/checkpoints/embedding.pt \
    --gnn-ckpt artifacts/checkpoints/gnn.pt \
    --shard data/processed/trackml_small \
    --limit 200 --threshold 0.7
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

## Implementation Notes

- Pure PyTorch — no PyTorch Geometric dependency (enables `torch.compile` and gradient checkpointing)
- VRAM budget: < 2 GB for events with N=2K hits and E≤16K edges (fits on 8 GB GPU)
- Supports bf16 mixed precision for inference
- KNN implementation is pure PyTorch; for O(10⁵)-hit events, swap in FAISS backend (same API)

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.0
- NumPy, SciPy, pandas, pyarrow

## References

- Amrouche et al., "The Tracking Machine Learning Challenge: Accuracy Phase," *arXiv:1904.06778*, 2019.
- Battaglia et al., "Interaction Networks for Learning about Objects, Relations and Physics," *NeurIPS*, 2016.
- Ju et al., "Performance of a Geometric Deep Learning Pipeline for HL-LHC Particle Tracking," *Eur. Phys. J. C*, 2021.

## License

MIT
