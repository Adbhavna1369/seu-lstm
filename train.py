"""
SEU Prediction Project — Training Loop
=======================================
Trains the stacked LSTM baseline with:
  - Huber loss (robust to SEP spike outliers)
  - Adam optimiser with cosine annealing LR schedule
  - Early stopping on validation loss
  - Checkpointing of best model
  - TensorBoard-compatible CSV logging

Usage (local):
    python train.py --processed-dir ./data/processed --out-dir ./runs/baseline

Usage (HPC — called from submit.slurm):
    python train.py --processed-dir $SCRATCH/data/processed \
                    --out-dir $SCRATCH/runs/baseline \
                    --batch-size 128 --epochs 100 --num-workers 4
"""

import argparse
import csv
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import build_dataloaders
from model import build_model, build_loss, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for regression evaluation."""
    with torch.no_grad():
        mae  = (y_pred - y_true).abs().mean().item()
        rmse = ((y_pred - y_true) ** 2).mean().sqrt().item()
        # MAPE: skip near-zero targets to avoid division by zero
        mask = y_true.abs() > 1e-6
        mape = ((y_pred[mask] - y_true[mask]).abs() /
                y_true[mask].abs()).mean().item() * 100 if mask.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    loss_fn:   nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = loss_fn(y_pred, y_batch)
        loss.backward()

        # Gradient clipping — stabilises LSTM training (common in practice)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        all_preds.append(y_pred.detach())
        all_targets.append(y_batch.detach())

    n = len(loader.dataset)
    metrics = compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    metrics["loss"] = total_loss / n
    return metrics


@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device:  str,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        y_pred  = model(X_batch)
        loss    = loss_fn(y_pred, y_batch)
        total_loss += loss.item() * len(y_batch)
        all_preds.append(y_pred)
        all_targets.append(y_batch)

    n = len(loader.dataset)
    metrics = compute_metrics(
        torch.cat(all_preds), torch.cat(all_targets)
    )
    metrics["loss"] = total_loss / n
    return metrics


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when val_loss has not improved for `patience` epochs.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            return False   # improvement — continue
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # no improvement — stop
        return False


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    def __init__(self, path: Path):
        self.path = path
        self._file   = None
        self._writer = None

    def __enter__(self):
        self._file   = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["epoch", "train_loss", "train_mae", "train_rmse",
                        "val_loss", "val_mae", "val_rmse", "val_mape", "lr", "epoch_time_s"]
        )
        self._writer.writeheader()
        return self

    def write(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def __exit__(self, *args):
        self._file.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # --- Data ---
    train_dl, val_dl, test_dl, n_features = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    window_size = train_dl.dataset.window_size

    # --- Model ---
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device=device)
        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resuming from epoch {start_epoch}")
    else:
        model = build_model(
            n_features=n_features,
            window_size=window_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            device=device,
        )
        start_epoch = 1

    loss_fn   = build_loss(delta=args.huber_delta)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    stopper   = EarlyStopping(patience=args.patience)

    best_ckpt_path = out_dir / "best_model.pt"
    log_path       = out_dir / "training_log.csv"

    log.info(f"Training for up to {args.epochs} epochs  |  early stop patience={args.patience}")

    with CSVLogger(log_path) as csv_log:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()

            train_metrics = train_one_epoch(model, train_dl, loss_fn, optimizer, device)
            val_metrics   = evaluate(model, val_dl, loss_fn, device)
            scheduler.step()

            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[0]

            log.info(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.5f}  "
                f"train_mae={train_metrics['mae']:.5f} | "
                f"val_loss={val_metrics['loss']:.5f}  "
                f"val_mae={val_metrics['mae']:.5f}  "
                f"val_rmse={val_metrics['rmse']:.5f} | "
                f"lr={lr_now:.2e}  {elapsed:.1f}s"
            )

            csv_log.write({
                "epoch":         epoch,
                "train_loss":    round(train_metrics["loss"],  6),
                "train_mae":     round(train_metrics["mae"],   6),
                "train_rmse":    round(train_metrics["rmse"],  6),
                "val_loss":      round(val_metrics["loss"],    6),
                "val_mae":       round(val_metrics["mae"],     6),
                "val_rmse":      round(val_metrics["rmse"],    6),
                "val_mape":      round(val_metrics["mape"],    4),
                "lr":            round(lr_now, 8),
                "epoch_time_s":  round(elapsed, 2),
            })

            # Save best checkpoint
            if val_metrics["loss"] < stopper.best_loss:
                save_checkpoint(model, optimizer, epoch,
                                val_metrics["loss"], best_ckpt_path)
                stopper.best_epoch = epoch

            if stopper.step(val_metrics["loss"]):
                log.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss={stopper.best_loss:.6f} at epoch {stopper.best_epoch}."
                )
                break

    # --- Final evaluation on test set ---
    log.info("Loading best checkpoint for test evaluation ...")
    best_model, _ = load_checkpoint(best_ckpt_path, device=device)
    test_metrics  = evaluate(best_model, test_dl, loss_fn, device)

    log.info(
        f"Test set | "
        f"loss={test_metrics['loss']:.5f}  "
        f"MAE={test_metrics['mae']:.5f}  "
        f"RMSE={test_metrics['rmse']:.5f}  "
        f"MAPE={test_metrics['mape']:.2f}%"
    )

    # Save test results
    test_result_path = out_dir / "test_results.txt"
    test_result_path.write_text(
        f"Test MAE  : {test_metrics['mae']:.6f}\n"
        f"Test RMSE : {test_metrics['rmse']:.6f}\n"
        f"Test MAPE : {test_metrics['mape']:.4f}%\n"
        f"Test Loss : {test_metrics['loss']:.6f}\n"
        f"Best epoch: {stopper.best_epoch}\n"
    )
    log.info(f"Test results saved -> {test_result_path}")
    log.info(f"Training log saved -> {log_path}")
    log.info("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SEU project — LSTM training")

    # Data
    p.add_argument("--processed-dir", default="./data/processed")
    p.add_argument("--out-dir",       default="./runs/baseline")
    p.add_argument("--num-workers",   type=int,   default=4)

    # Model
    p.add_argument("--hidden-size",   type=int,   default=128)
    p.add_argument("--num-layers",    type=int,   default=2)
    p.add_argument("--dropout",       type=float, default=0.3)

    # Training
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--huber-delta",   type=float, default=1.0)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--resume",        type=str,   default=None,
                   help="Path to checkpoint .pt to resume training from")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
