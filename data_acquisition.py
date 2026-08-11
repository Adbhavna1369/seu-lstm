"""
SEU Prediction Project — Data Acquisition
==========================================
Fetches and caches:
  - GOES-16/18 proton flux (NOAA SWPC JSON API)
  - GOES-16/18 electron flux (NOAA SWPC JSON API)
  - NOAA SWPC SEP event list
  - AE9/AP9 RADBELT (file-based: expects pre-downloaded NetCDF)

Usage:
    python data_acquisition.py --start 2020-01-01 --end 2023-12-31 --cache-dir ./data/raw

HPC note: Run this ONCE on a login node before submitting the training job.
          All outputs are cached as parquet for fast DataLoader access.
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA SWPC API endpoints (public, no auth required)
# ---------------------------------------------------------------------------
GOES_PROTON_7DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json"
GOES_ELECTRON_7DAY_URL = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-7-day.json"
SEP_EVENTS_URL = "https://services.swpc.noaa.gov/json/solar-proton-events-observable.json"

# NOAA archives for historical data (>7 days)
GOES_ARCHIVE_BASE = "https://sohoftp.nascom.nasa.gov/sdb/goes/particle/"

# Proton energy channels available in GOES-16/18 (MeV)
PROTON_CHANNELS = [">=1", ">=5", ">=10", ">=30", ">=50", ">=100", ">=500"]
# Electron energy channels
ELECTRON_CHANNELS = [">=0.8", ">=2.0", ">=4.0"]

RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fetch_json(url: str, timeout: int = 30) -> list | dict:
    """GET a JSON endpoint with retries."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/{RETRY_ATTEMPTS} failed for {url}: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"Failed to fetch {url} after {RETRY_ATTEMPTS} attempts.")


def _cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.parquet"


def _load_or_fetch(cache_dir: Path, name: str, fetch_fn, force: bool = False) -> pd.DataFrame:
    path = _cache_path(cache_dir, name)
    if path.exists() and not force:
        log.info(f"Loading cached {name} from {path}")
        return pd.read_parquet(path)
    log.info(f"Fetching {name} ...")
    df = fetch_fn()
    df.to_parquet(path, index=False)
    log.info(f"Saved {name} → {path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# GOES Proton Flux (7-day live + archive merge)
# ---------------------------------------------------------------------------

def fetch_goes_proton_live() -> pd.DataFrame:
    """Fetch the NOAA SWPC 7-day integral proton flux JSON."""
    raw = _fetch_json(GOES_PROTON_7DAY_URL)
    records = []
    for entry in raw:
        records.append({
            "time_tag": pd.to_datetime(entry["time_tag"], utc=True),
            "satellite": entry.get("satellite", "unknown"),
            "flux_p1_gt1mev":   entry.get("p1"),    # >=1 MeV  (cm⁻² s⁻¹ sr⁻¹)
            "flux_p5_gt5mev":   entry.get("p5"),    # >=5 MeV
            "flux_p6_gt10mev":  entry.get("p6"),    # >=10 MeV
            "flux_p7_gt30mev":  entry.get("p7"),    # >=30 MeV
            "flux_p8_gt50mev":  entry.get("p8"),    # >=50 MeV
            "flux_p9_gt100mev": entry.get("p9"),    # >=100 MeV
            "flux_p11_gt500mev":entry.get("p11"),   # >=500 MeV
        })
    df = pd.DataFrame(records).sort_values("time_tag").reset_index(drop=True)
    return df


def fetch_goes_proton_archive(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch GOES proton flux from NOAA archive for date ranges > 7 days.

    Archive URL pattern:
      https://sohoftp.nascom.nasa.gov/sdb/goes/particle/G16_GOES-16_intflux_1min_YYYYMM.txt

    NOTE: For very long ranges (years), this loops month by month.
          On HPC, consider parallelising with multiprocessing.Pool.
    """
    dfs = []
    current = start.replace(day=1)
    while current <= end:
        yyyymm = current.strftime("%Y%m")
        # Try GOES-18 first (primary after Apr 2023), fall back to GOES-16
        for sat in ["G18", "G16"]:
            url = f"{GOES_ARCHIVE_BASE}{sat}_GOES-{'18' if sat=='G18' else '16'}_intflux_1min_{yyyymm}.txt"
            try:
                resp = requests.get(url, timeout=60)
                if resp.status_code == 200:
                    from io import StringIO
                    # NOAA archive files are space-delimited with comment lines starting with #
                    text = "\n".join(
                        line for line in resp.text.splitlines()
                        if not line.startswith("#") and line.strip()
                    )
                    chunk = pd.read_csv(
                        StringIO(text), sep=r"\s+", header=None,
                        names=["year","month","day","hour","minute","jday",
                               "flux_p1_gt1mev","flux_p5_gt5mev","flux_p6_gt10mev",
                               "flux_p7_gt30mev","flux_p8_gt50mev","flux_p9_gt100mev",
                               "flux_p11_gt500mev"],
                        na_values=["-9999.0", "-9999", "9999.0"]
                    )
                    chunk["time_tag"] = pd.to_datetime(
                        chunk[["year","month","day","hour","minute"]]
                        .rename(columns={"hour":"hour","minute":"minute"})
                    ).dt.tz_localize("UTC")
                    chunk["satellite"] = sat
                    dfs.append(chunk.drop(columns=["year","month","day","hour","minute","jday"]))
                    log.info(f"  Fetched {sat} {yyyymm}: {len(chunk)} rows")
                    break
            except Exception as e:
                log.warning(f"  Could not fetch {sat} {yyyymm}: {e}")
        # Advance one month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not dfs:
        raise RuntimeError("No archive proton data fetched. Check date range and network.")
    df = pd.concat(dfs, ignore_index=True).sort_values("time_tag").reset_index(drop=True)
    # Clip to requested window
    df = df[(df["time_tag"] >= pd.Timestamp(start, tz="UTC")) &
            (df["time_tag"] <= pd.Timestamp(end,   tz="UTC"))].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# GOES Electron Flux (7-day live)
# ---------------------------------------------------------------------------

def fetch_goes_electron_live() -> pd.DataFrame:
    raw = _fetch_json(GOES_ELECTRON_7DAY_URL)
    records = []
    for entry in raw:
        records.append({
            "time_tag":         pd.to_datetime(entry["time_tag"], utc=True),
            "satellite":        entry.get("satellite", "unknown"),
            "flux_e1_gt0_8mev": entry.get("e1"),   # >=0.8 MeV
            "flux_e2_gt2mev":   entry.get("e2"),   # >=2.0 MeV
            "flux_e3_gt4mev":   entry.get("e3"),   # >=4.0 MeV
        })
    df = pd.DataFrame(records).sort_values("time_tag").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# NOAA SWPC SEP Events
# ---------------------------------------------------------------------------

def fetch_sep_events() -> pd.DataFrame:
    """
    Fetch the NOAA SWPC Solar Proton Event list.
    Returns start/end times and peak flux for each SEP event.
    Used as auxiliary labels / event flags in preprocessing.
    """
    raw = _fetch_json(SEP_EVENTS_URL)
    records = []
    for ev in raw:
        records.append({
            "event_start":    pd.to_datetime(ev.get("start_time"), utc=True, errors="coerce"),
            "event_end":      pd.to_datetime(ev.get("end_time"),   utc=True, errors="coerce"),
            "peak_time":      pd.to_datetime(ev.get("max_time"),   utc=True, errors="coerce"),
            "peak_flux_pfu":  ev.get("max_pfu"),          # proton flux units (pfu = p cm⁻² s⁻¹ sr⁻¹)
            "threshold_pfu":  ev.get("threshold"),
            "energy_channel": ev.get("energy"),
        })
    df = pd.DataFrame(records).sort_values("event_start").reset_index(drop=True)
    log.info(f"Fetched {len(df)} SEP events")
    return df


# ---------------------------------------------------------------------------
# AE9/AP9 RADBELT (file-based)
# ---------------------------------------------------------------------------

def load_radbelt_netcdf(filepath: str | Path) -> pd.DataFrame:
    """
    Load a pre-downloaded AE9/AP9 NetCDF output file.

    AE9/AP9 is run offline via the NASA RADBELT tool (https://www.vdl.afrl.af.mil/programs/ae9ap9/)
    and outputs NetCDF. This function extracts flux profiles along an orbit trajectory.

    Expected variables (adjust to your specific run output):
      - time         : seconds from epoch
      - L_star       : McIlwain L* parameter
      - electron_flux: differential electron flux [MeV⁻¹ cm⁻² s⁻¹ sr⁻¹]
      - proton_flux  : differential proton flux
      - energy_bins  : energy bin centres (MeV)

    Returns a long-format DataFrame indexed by (time, energy_bin).
    """
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 is required for RADBELT loading: pip install netCDF4")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"RADBELT NetCDF not found: {filepath}\n"
            "Download AE9/AP9 from https://www.vdl.afrl.af.mil/programs/ae9ap9/ "
            "and run for your orbit parameters."
        )

    records = []
    with nc.Dataset(filepath) as ds:
        times = nc.num2date(ds["time"][:], ds["time"].units)
        energy_bins = ds["energy_bins"][:].data  # MeV

        for t_idx, t in enumerate(times):
            timestamp = pd.Timestamp(t.isoformat(), tz="UTC")
            for e_idx, energy in enumerate(energy_bins):
                records.append({
                    "time_tag":      timestamp,
                    "energy_mev":    float(energy),
                    "L_star":        float(ds["L_star"][t_idx]) if "L_star" in ds.variables else np.nan,
                    "electron_flux": float(ds["electron_flux"][t_idx, e_idx]),
                    "proton_flux":   float(ds["proton_flux"][t_idx, e_idx]),
                })

    df = pd.DataFrame(records)
    log.info(f"Loaded RADBELT: {len(df)} records from {filepath.name}")
    return df


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SEU project — data acquisition")
    p.add_argument("--start",     type=str, default="2021-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end",       type=str, default="2023-12-31", help="End date YYYY-MM-DD")
    p.add_argument("--cache-dir", type=str, default="./data/raw", help="Local cache directory")
    p.add_argument("--radbelt-nc",type=str, default=None,         help="Path to AE9/AP9 NetCDF file")
    p.add_argument("--force",     action="store_true",            help="Re-fetch even if cache exists")
    p.add_argument("--live-only", action="store_true",
                   help="Only fetch 7-day live feeds (useful for testing without archive access)")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end   = datetime.strptime(args.end,   "%Y-%m-%d")

    log.info(f"Date range: {start.date()} → {end.date()}")
    log.info(f"Cache dir : {cache_dir.resolve()}")

    # 1. Proton flux
    if args.live_only:
        df_proton = _load_or_fetch(cache_dir, "goes_proton_live", fetch_goes_proton_live, args.force)
    else:
        df_proton = _load_or_fetch(
            cache_dir, f"goes_proton_{args.start}_{args.end}",
            lambda: fetch_goes_proton_archive(start, end),
            args.force
        )
    log.info(f"Proton flux: {df_proton.shape} | {df_proton['time_tag'].min()} → {df_proton['time_tag'].max()}")

    # 2. Electron flux (live feed only — archive structure differs)
    df_electron = _load_or_fetch(cache_dir, "goes_electron_live", fetch_goes_electron_live, args.force)
    log.info(f"Electron flux: {df_electron.shape}")

    # 3. SEP events
    df_sep = _load_or_fetch(cache_dir, "sep_events", fetch_sep_events, args.force)
    log.info(f"SEP events: {df_sep.shape}")

    # 4. RADBELT (optional)
    if args.radbelt_nc:
        df_radbelt = load_radbelt_netcdf(args.radbelt_nc)
        out = _cache_path(cache_dir, "radbelt")
        df_radbelt.to_parquet(out, index=False)
        log.info(f"RADBELT saved → {out}")

    log.info("✓ Data acquisition complete. Run preprocess.py next.")


if __name__ == "__main__":
    main()
