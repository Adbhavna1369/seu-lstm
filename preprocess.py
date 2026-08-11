"""
SEU Prediction Project — Preprocessing Pipeline
=================================================
Steps:
  1. Resample all sources to 5-min intervals (forward-fill short gaps, NaN long gaps)
  2. Temporal alignment on a common UTC index
  3. SEP event flag injection (binary column)
  4. Log-transform flux features (heavy-tailed distributions)
  5. Robust normalisation (median/IQR — robust to SEP spike outliers)
  6. CREME96 SEU rate label alignment
  7. Sliding window construction → (X, y) tensors saved as .npz

Usage:
    python preprocess.py --cache-dir ./data/raw --out-dir ./data/processed \
                         --creme96-csv ./data/creme96_seu_rates.csv \
                         --window-size 24 --horizon 6

    window-size : number of 5-min steps in each input sequence  (24 × 5min = 2h lookback)
    horizon     : steps ahead to predict                        (6  × 5min = 30min)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Flux columns that receive log1p transform before scaling
LOG_TRANSFORM_COLS = [
    "flux_p1_gt1mev", "flux_p5_gt5mev", "flux_p6_gt10mev",
    "flux_p7_gt30mev", "flux_p8_gt50mev", "flux_p9_gt100mev", "flux_p11_gt500mev",
    "flux_e1_gt0_8mev", "flux_e2_gt2mev", "flux_e3_gt4mev",
]

FREQ = "5min"
GAP_FILL_LIMIT = 6   # max consecutive 5-min steps to forward-fill (~30 min)


# ---------------------------------------------------------------------------
# 1. Load cached parquet files
# ---------------------------------------------------------------------------

def load_raw(cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Load whatever parquet files are present in the cache directory."""
    sources = {}
    for name in ["goes_proton", "goes_electron", "sep_events", "radbelt"]:
        # Match any file whose stem starts with the name
        matches = list(cache_dir.glob(f"{name}*.parquet"))
        if matches:
            # If multiple (e.g. date-stamped archives), take the largest
            path = max(matches, key=lambda p: p.stat().st_size)
            sources[name] = pd.read_parquet(path)
            log.info(f"Loaded {name}: {sources[name].shape}")
        else:
            log.warning(f"No parquet found for {name} in {cache_dir} — skipping")
    return sources


# ---------------------------------------------------------------------------
# 2. Resample to 5-min UTC grid
# ---------------------------------------------------------------------------

def resample_to_5min(df: pd.DataFrame, value_cols: list[str],
                     time_col: str = "time_tag") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()

    # Keep only numeric value columns (drop satellite labels etc.)
    df = df[value_cols]

    # Resample: mean within each 5-min bin
    df = df.resample(FREQ).mean()

    # Forward-fill short data gaps, leave long gaps as NaN
    df = df.ffill(limit=GAP_FILL_LIMIT)

    return df


# ---------------------------------------------------------------------------
# 3. Align on a common UTC index
# ---------------------------------------------------------------------------

def align_sources(proton: pd.DataFrame,
                  electron: pd.DataFrame | None,
                  radbelt: pd.DataFrame | None) -> pd.DataFrame:
    """Outer-join all sources on the 5-min UTC index."""
    merged = proton.copy()

    if electron is not None:
        merged = merged.join(electron, how="outer", rsuffix="_elec")

    if radbelt is not None:
        # RADBELT is wide (one row per time_tag, energy pivoted to columns)
        rb_pivot = radbelt.pivot_table(
            index="time_tag", columns="energy_mev",
            values=["electron_flux", "proton_flux"], aggfunc="mean"
        )
        rb_pivot.columns = [f"rb_{v}_{e:.2f}mev" for v, e in rb_pivot.columns]
        rb_pivot.index = pd.to_datetime(rb_pivot.index, utc=True)
        merged = merged.join(rb_pivot, how="outer")

    log.info(f"Merged feature matrix: {merged.shape}")
    return merged


# ---------------------------------------------------------------------------
# 4. SEP event flag
# ---------------------------------------------------------------------------

def inject_sep_flag(df: pd.DataFrame, sep_df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary column 'sep_event_active' = 1 during any SEP event window."""
    df = df.copy()
    df["sep_event_active"] = 0

    for _, row in sep_df.iterrows():
        start = row["event_start"]
        end   = row["event_end"]
        if pd.isna(start) or pd.isna(end):
            continue
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "sep_event_active"] = 1

    log.info(f"SEP flag set for {df['sep_event_active'].sum()} timesteps "
             f"({df['sep_event_active'].mean()*100:.1f}% of data)")
    return df


# ---------------------------------------------------------------------------
# 5. Log-transform + RobustScaler
# ---------------------------------------------------------------------------

def apply_transforms(df: pd.DataFrame,
                     fit_scaler: bool = True,
                     scaler_path: Path | None = None) -> tuple[pd.DataFrame, RobustScaler]:
    df = df.copy()

    # Log1p on flux columns (clip negatives to 0 first — occasional GOES fill values)
    existing_log_cols = [c for c in LOG_TRANSFORM_COLS if c in df.columns]
    df[existing_log_cols] = np.log1p(df[existing_log_cols].clip(lower=0))

    feature_cols = [c for c in df.columns if c != "seu_rate"]

    if fit_scaler:
        scaler = RobustScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            log.info(f"Scaler saved → {scaler_path}")
    else:
        if scaler_path is None or not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


# ---------------------------------------------------------------------------
# 6. Load and align CREME96 SEU rate labels
# ---------------------------------------------------------------------------

def load_creme96_labels(csv_path: str | Path, index: pd.DatetimeIndex) -> pd.Series:
    """
    Load CREME96-simulated SEU rates and align to the 5-min feature index.

    Expected CSV columns:
        time_tag   : UTC timestamp
        seu_rate   : SEU counts per orbit (float)

    The label is the SEU rate at t+horizon (target to predict).
    Resampling: mean within each 5-min bin.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CREME96 CSV not found: {csv_path}\n"
            "Expected columns: time_tag (UTC), seu_rate (float)"
        )

    df = pd.read_csv(csv_path, parse_dates=["time_tag"])
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    df = df.set_index("time_tag").sort_index()
    df = df["seu_rate"].resample(FREQ).mean().ffill(limit=GAP_FILL_LIMIT)

    # Reindex to match feature matrix
    labels = df.reindex(index)
    log.info(f"CREME96 labels aligned: {labels.notna().sum()} / {len(labels)} valid timesteps")
    return labels


# ---------------------------------------------------------------------------
# 7. Sliding window → (X, y) arrays
# ---------------------------------------------------------------------------

def make_windows(features: pd.DataFrame,
                 labels: pd.Series,
                 window_size: int,
                 horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding window samples.

    Args:
        features   : (T, F) DataFrame — already scaled
        labels     : (T,) Series — SEU rate
        window_size: number of past timesteps per sample (lookback)
        horizon    : steps ahead to predict (target index offset)

    Returns:
        X      : (N, window_size, F) float32
        y      : (N,) float32
        t_idx  : (N,) array of prediction timestamps (for diagnostics)
    """
    feat_arr  = features.values.astype(np.float32)
    label_arr = labels.values.astype(np.float32)
    t_index   = features.index

    X_list, y_list, t_list = [], [], []

    for i in range(window_size, len(feat_arr) - horizon + 1):
        x_window = feat_arr[i - window_size : i]          # (window_size, F)
        y_target = label_arr[i + horizon - 1]             # scalar

        # Skip if any NaN in window or target
        if np.isnan(x_window).any() or np.isnan(y_target):
            continue

        X_list.append(x_window)
        y_list.append(y_target)
        t_list.append(t_index[i + horizon - 1])

    X = np.stack(X_list)                    # (N, window_size, F)
    y = np.array(y_list, dtype=np.float32)  # (N,)
    t = np.array(t_list)                    # (N,)

    log.info(f"Windows: X={X.shape}, y={y.shape}  "
             f"(dropped {len(feat_arr) - window_size - horizon + 1 - len(y_list)} NaN samples)")
    return X, y, t


# ---------------------------------------------------------------------------
# Train / Val / Test split (chronological — NO shuffle)
# ---------------------------------------------------------------------------

def chronological_split(X: np.ndarray, y: np.ndarray, t: np.ndarray,
                         val_frac: float = 0.15, test_frac: float = 0.15
                         ) -> dict[str, tuple]:
    N = len(X)
    n_test = int(N * test_frac)
    n_val  = int(N * val_frac)
    n_train = N - n_val - n_test

    splits = {
        "train": (X[:n_train],          y[:n_train],          t[:n_train]),
        "val":   (X[n_train:n_train+n_val], y[n_train:n_train+n_val], t[n_train:n_train+n_val]),
        "test":  (X[n_train+n_val:],    y[n_train+n_val:],    t[n_train+n_val:]),
    }
    for name, (Xs, ys, ts) in splits.items():
        log.info(f"  {name:5s}: {len(Xs):6d} samples  "
                 f"{pd.Timestamp(ts[0]).date()} → {pd.Timestamp(ts[-1]).date()}")
    return splits


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SEU project — preprocessing")
    p.add_argument("--cache-dir",   default="./data/raw",       help="Raw parquet cache dir")
    p.add_argument("--out-dir",     default="./data/processed",  help="Output dir for .npz files")
    p.add_argument("--creme96-csv", default=None,               help="Path to CREME96 SEU rate CSV")
    p.add_argument("--window-size", type=int, default=24,       help="Lookback steps (×5 min)")
    p.add_argument("--horizon",     type=int, default=6,        help="Forecast horizon steps (×5 min)")
    p.add_argument("--val-frac",    type=float, default=0.15)
    p.add_argument("--test-frac",   type=float, default=0.15)
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw data ---
    sources = load_raw(cache_dir)
    if "goes_proton" not in sources:
        raise RuntimeError("Proton flux data missing. Run data_acquisition.py first.")

    # --- Resample ---
    proton_cols = [c for c in sources["goes_proton"].columns
                   if c not in ("time_tag", "satellite")]
    proton_5min = resample_to_5min(sources["goes_proton"], proton_cols)

    electron_5min = None
    if "goes_electron" in sources:
        elec_cols = [c for c in sources["goes_electron"].columns
                     if c not in ("time_tag", "satellite")]
        electron_5min = resample_to_5min(sources["goes_electron"], elec_cols)

    radbelt = sources.get("radbelt")

    # --- Align ---
    merged = align_sources(proton_5min, electron_5min, radbelt)

    # --- SEP flag ---
    if "sep_events" in sources:
        merged = inject_sep_flag(merged, sources["sep_events"])

    # --- Log-transform + scale ---
    scaler_path = out_dir / "robust_scaler.pkl"
    merged, scaler = apply_transforms(merged, fit_scaler=True, scaler_path=scaler_path)

    # --- CREME96 labels ---
    if args.creme96_csv:
        labels = load_creme96_labels(args.creme96_csv, merged.index)
        merged["seu_rate"] = labels
    else:
        log.warning("No --creme96-csv supplied. Using synthetic placeholder labels (all zeros).")
        log.warning("Replace with real CREME96 output before training.")
        merged["seu_rate"] = 0.0

    # Separate features from label
    label_series = merged["seu_rate"]
    feature_df   = merged.drop(columns=["seu_rate"])

    log.info(f"Feature columns ({feature_df.shape[1]}): {list(feature_df.columns)}")

    # Save feature column names for the model
    col_path = out_dir / "feature_columns.txt"
    col_path.write_text("\n".join(feature_df.columns))

    # --- Sliding windows ---
    X, y, t = make_windows(feature_df, label_series, args.window_size, args.horizon)

    # --- Split ---
    log.info(f"Splitting (val={args.val_frac}, test={args.test_frac}):")
    splits = chronological_split(X, y, t, args.val_frac, args.test_frac)

    # --- Save ---
    for split_name, (Xs, ys, ts) in splits.items():
        path = out_dir / f"{split_name}.npz"
        np.savez_compressed(path, X=Xs, y=ys, t=ts.astype(str))
        log.info(f"Saved {path}")

    log.info(f"\n✓ Preprocessing complete.")
    log.info(f"  Window size : {args.window_size} × 5 min = {args.window_size * 5} min lookback")
    log.info(f"  Horizon     : {args.horizon} × 5 min = {args.horizon * 5} min ahead")
    log.info(f"  Features    : {feature_df.shape[1]}")
    log.info(f"  Output dir  : {out_dir.resolve()}")
    log.info(f"\nNext step: python model.py  (then  python train.py)")


if __name__ == "__main__":
    main()
