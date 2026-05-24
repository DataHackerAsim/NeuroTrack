"""Seeding helpers for python / numpy / torch / cuda."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Make the run reproducible up to non-deterministic CUDA kernels.

    ``deterministic=True`` flips on PyTorch's deterministic algorithms;
    this is slow and is intended only for golden-fixture regression runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


__all__ = ["seed_everything"]
