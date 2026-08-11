"""
SEU Prediction Project — Stacked LSTM Baseline Model
=====================================================
Architecture (justified from literature):

  Input  →  LSTM(128) → Dropout(0.3)
          →  LSTM(128) → Dropout(0.3)
          →  Linear(64) → ReLU
          →  Linear(1)   [SEU rate regression output]

Design decisions (documented for thesis):

  Hidden units = 128
    Brownlee (2018) Ch.20, Ch.25: multivariate time-series tasks use 100-200
    units. 128 is a conservative midpoint appropriate for ~10 input features.

  2 LSTM layers
    Atef et al. (2020): stacking beyond 2 layers yields no significant accuracy
    gain but roughly doubles training time. 2 layers provides sufficient depth
    for the non-linear flux-to-SEU mapping without overfitting.

  Dropout = 0.3
    Brownlee (2018) Ch.25: uses Dropout(0.5) for multivariate classification.
    We use 0.3 (more conservative) because the SEU regression dataset is
    relatively small and 0.5 risks underfitting.

  Loss = Huber (delta=1.0)
    Rashid et al. (2026): MSE breaks down at 5% label corruption; Huber
    remains robust at 50%. SEP spike events create exactly this heavy-tailed
    target distribution in SEU rate data.

Usage:
    from model import SEULSTMBaseline, build_model
    model = build_model(n_features=11, window_size=24)
    print(model)
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SEULSTMBaseline(nn.Module):
    """
    2-layer stacked LSTM for SEU rate regression.

    Args:
        n_features   : number of input features (F) from preprocessing
        window_size  : number of 5-min timesteps per sequence (T)
        hidden_size  : LSTM units per layer (default 128, Brownlee 2018)
        num_layers   : number of stacked LSTM layers (default 2, Atef et al. 2020)
        dropout      : dropout probability between layers (default 0.3)
        fc_size      : units in the intermediate fully-connected layer
    """

    def __init__(
        self,
        n_features:  int,
        window_size: int,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.3,
        fc_size:     int = 64,
    ):
        super().__init__()

        self.n_features  = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Stacked LSTM
        # dropout arg in nn.LSTM applies between layers (not after the last layer)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Explicit dropout after final LSTM layer (Brownlee 2018, Ch.25)
        self.dropout = nn.Dropout(p=dropout)

        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, window_size, n_features)

        Returns:
            out : (batch,) — predicted SEU rate
        """
        # lstm_out: (batch, seq_len, hidden_size)
        # We only use the output at the final timestep
        lstm_out, _ = self.lstm(x)           # (B, T, H)
        last_step   = lstm_out[:, -1, :]     # (B, H) — final timestep
        dropped     = self.dropout(last_step)
        out         = self.fc(dropped)       # (B, 1)
        return out.squeeze(1)                # (B,)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory + loss function
# ---------------------------------------------------------------------------

def build_model(
    n_features:  int,
    window_size: int,
    hidden_size: int   = 128,
    num_layers:  int   = 2,
    dropout:     float = 0.3,
    fc_size:     int   = 64,
    device:      str   = "cpu",
) -> SEULSTMBaseline:
    """
    Instantiate and log the baseline LSTM model.

    Args:
        n_features  : from DataLoader (train_ds.n_features)
        window_size : from DataLoader (train_ds.window_size)
        device      : 'cuda' or 'cpu'

    Returns:
        model on the requested device
    """
    model = SEULSTMBaseline(
        n_features=n_features,
        window_size=window_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_size=fc_size,
    ).to(device)

    log.info(f"Model: SEULSTMBaseline")
    log.info(f"  Input  : (batch, {window_size}, {n_features})")
    log.info(f"  LSTM   : {num_layers} x {hidden_size} units, dropout={dropout}")
    log.info(f"  FC head: {hidden_size} -> {fc_size} -> 1")
    log.info(f"  Params : {model.count_parameters():,}")
    log.info(f"  Device : {device}")

    return model


def build_loss(delta: float = 1.0) -> nn.HuberLoss:
    """
    Huber loss for robust SEU rate regression.

    delta controls the transition from L2 (quadratic, for small errors)
    to L1 (linear, for large errors / SEP spikes).
    delta=1.0 is the standard default (Rashid et al. 2026).

    Args:
        delta : threshold between L2 and L1 regime

    Returns:
        nn.HuberLoss instance
    """
    return nn.HuberLoss(reduction="mean", delta=delta)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     SEULSTMBaseline,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_loss:  float,
    path:      str | Path,
) -> None:
    """Save model + optimiser state for resuming training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":        epoch,
        "val_loss":     val_loss,
        "model_state":  model.state_dict(),
        "optim_state":  optimizer.state_dict(),
        "config": {
            "n_features":  model.n_features,
            "window_size": model.window_size,
            "hidden_size": model.hidden_size,
            "num_layers":  model.num_layers,
        },
    }, path)
    log.info(f"Checkpoint saved -> {path}  (epoch={epoch}, val_loss={val_loss:.6f})")


def load_checkpoint(
    path:    str | Path,
    device:  str = "cpu",
) -> tuple[SEULSTMBaseline, dict]:
    """
    Load a saved checkpoint and reconstruct the model.

    Returns:
        model   : SEULSTMBaseline with loaded weights
        ckpt    : full checkpoint dict (contains epoch, val_loss, optim_state)
    """
    path = Path(path)
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]

    model = build_model(
        n_features=cfg["n_features"],
        window_size=cfg["window_size"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        device=device,
    )
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Checkpoint loaded <- {path}  (epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f})")
    return model, ckpt


# ---------------------------------------------------------------------------
# Quick sanity check (run directly: python model.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Simulate a batch matching preprocess.py defaults:
    # window_size=24, n_features=11 (10 flux channels + sep_event_active)
    BATCH       = 8
    WINDOW_SIZE = 24
    N_FEATURES  = 11

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(N_FEATURES, WINDOW_SIZE, device=device)
    loss_fn = build_loss(delta=1.0)

    # Forward pass
    x_dummy = torch.randn(BATCH, WINDOW_SIZE, N_FEATURES, device=device)
    y_dummy = torch.rand(BATCH, device=device)

    y_pred = model(x_dummy)
    loss   = loss_fn(y_pred, y_dummy)

    log.info(f"Forward pass OK -- output shape: {y_pred.shape}")
    log.info(f"Huber loss on random data: {loss.item():.4f}")
    log.info(f"Model:\n{model}")
