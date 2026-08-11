# SEU Prediction — LSTM Baseline

PhD research project: predicting cosmic ray-induced **Single Event Upset (SEU)** rates in LEO satellite and UAV onboard electronics using an LSTM baseline model.

## Data Sources
- **GOES-16/18** integral proton & electron flux (NOAA SWPC)
- **NOAA SWPC SEP event list**
- **NASA AE9/AP9 RADBELT** (pre-downloaded NetCDF)
- **CREME96** simulated SEU rates (ground truth)

## Pipeline

```
data_acquisition.py  →  preprocess.py  →  dataset.py  →  train.py
```

## Setup

```bash
pip install -r requirements.txt --user
```

## Usage

```bash
# 1. Fetch and cache raw data (run on HPC login node)
python data_acquisition.py --start 2020-01-01 --end 2023-12-31 --cache-dir ./data/raw

# 2. Preprocess → sliding windows
python preprocess.py \
    --cache-dir ./data/raw \
    --out-dir ./data/processed \
    --creme96-csv ./data/creme96_seu_rates.csv \
    --window-size 24 --horizon 6

# 3. Train (submit via SLURM)
sbatch submit.slurm
```

## Project Status
- [x] Data acquisition
- [x] Preprocessing pipeline
- [ ] PyTorch Dataset / DataLoader
- [ ] LSTM baseline model
- [ ] Training loop
- [ ] Evaluation & metrics

## Notes
- `data/` is git-ignored (raw NASA/NOAA data, processed arrays)
- Scaler fitted on train split only — no leakage