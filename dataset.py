"""
SEU Prediction Project — PyTorch Dataset & DataLoader
======================================================
Loads preprocessed .npz sliding-window arrays produced by preprocess.py
and wraps them in a PyTorch Dataset for use in train.py.

Design notes (Brownlee 2018, Ch.6):
  - Input shape: (batch, window_size, n_features)  — [samples, timesteps, features]
  - Target shape: (batch,)                          — scalar SEU rate per sample
  - DataLoader uses persistent_workers=True on HPC for faster epoch throughput

Usage:
    from dataset import build_dataloaders
    train_dl, val_dl, test_dl, n_features = build_dataloaders(
        processed_dir='./data/processed', batch_size=64
    )
"""

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SEUDataset(Dataset):
    """
    Wraps a single (X, y) split loaded from a .npz file.

    Args:
        npz_path : path to train.npz / val.npz / test.npz
        device   : tensors kept on CPU here; moved to device inside training loop
    """

    def __init__(self, npz_path: str | Path, device: str = "cpu"):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {npz_path}\n"
                "Run preprocess.py first."
            )

        data = np.load(npz_path, allow_pickle=True)
        self.X = torch.tensor(data["X"], dtype=torch.float32)  # (N, T, F)
        self.y = torch.tensor(data["y"], dtype=torch.float32)  # (N,)
        self.t = data["t"]                                      # (N,) timestamp strings

        log.info(
            f"Loaded {npz_path.stem}: "
            f"X={tuple(self.X.shape)}  y={tuple(self.y.shape)}  "
            f"[{self.t[0]} -> {self.t[-1]}]"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    processed_dir: str | Path = "./data/processed",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Build train / val / test DataLoaders from preprocessed .npz files.

    Args:
        processed_dir : directory containing train.npz, val.npz, test.npz
        batch_size    : samples per mini-batch
        num_workers   : parallel workers for data loading
                        (set 0 on login nodes, 4-8 on compute nodes)
        pin_memory    : speeds up CPU->GPU transfer on HPC (False if CPU-only)

    Returns:
        train_dl, val_dl, test_dl, n_features
    """
    processed_dir = Path(processed_dir)

    train_ds = SEUDataset(processed_dir / "train.npz")
    val_ds   = SEUDataset(processed_dir / "val.npz")
    test_ds  = SEUDataset(processed_dir / "test.npz")

    # Verify feature consistency across splits
    assert train_ds.n_features == val_ds.n_features == test_ds.n_features, (
        "Feature count mismatch across splits -- re-run preprocess.py"
    )
    n_features = train_ds.n_features

    # Shared DataLoader kwargs
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # NOTE: shuffle=False throughout -- chronological order must be preserved
    # for time series. Shuffling would leak future context into training.
    train_dl = DataLoader(train_ds, shuffle=False, **loader_kwargs)
    val_dl   = DataLoader(val_ds,  shuffle=False, **loader_kwargs)
    test_dl  = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    log.info(
        f"DataLoaders ready | "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} samples | "
        f"batch={batch_size}  workers={num_workers} | "
        f"features={n_features}  window={train_ds.window_size}"
    )

    return train_dl, val_dl, test_dl, n_features


# ---------------------------------------------------------------------------
# Quick sanity check (run directly: python dataset.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", default="./data/processed")
    p.add_argument("--batch-size",    type=int, default=64)
    p.add_argument("--num-workers",   type=int, default=0)
    args = p.parse_args()

    train_dl, val_dl, test_dl, n_features = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    X_batch, y_batch = next(iter(train_dl))
    log.info(f"Sample batch -- X: {X_batch.shape}  y: {y_batch.shape}")
    log.info(f"X dtype: {X_batch.dtype}  y dtype: {y_batch.dtype}")
    log.info(f"y range: [{y_batch.min():.4f}, {y_batch.max():.4f}]")
    log.info("Dataset OK")
