"""
prepare_sample_data.py
──────────────────────
Downloads the UCI Hydraulic System Condition Monitoring dataset and
processes it into the 81-feature CSV format expected by HydrauProtect AI.

Usage:
    python prepare_sample_data.py

Output:
    data/sample_data.csv        — full dataset (2205 cycles, 81 features)
    data/sample_10_cycles.csv   — 10-cycle sample for quick testing

UCI Dataset:
    Helwig et al. (2018). Condition monitoring of hydraulic systems.
    https://archive.ics.uci.edu/dataset/447
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

UCI_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
OUTPUT_DIR = "data"
FULL_OUTPUT = os.path.join(OUTPUT_DIR, "sample_data.csv")
SMALL_OUTPUT = os.path.join(OUTPUT_DIR, "sample_10_cycles.csv")

# Sensors and their sampling rates (Hz) over a 60-second cycle
# columns per file = Hz × 60 seconds
SENSORS = {
    'PS1':  100,   # 6000 columns
    'PS2':  100,
    'PS3':  100,
    'PS4':  100,
    'PS5':  100,
    'PS6':  100,
    'EPS1': 100,
    'FS1':   10,   # 600 columns
    'FS2':   10,
    'TS1':    1,   # 60 columns
    'TS2':    1,
    'TS3':    1,
    'TS4':    1,
    'VS1':    1,
    'CE':     1,
    'SE':     1,
}

# profile.txt column indices (0-based)
# col 0: cooler condition
# col 1: valve condition
# col 2: pump leakage       ← used by HydrauProtect
# col 3: accumulator bar    ← used by HydrauProtect
# col 4: stable flag        ← used by HydrauProtect
PROFILE_COLS = ['cooler', 'valve', 'pump', 'accumulator', 'stable']


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_dataset(url: str) -> bytes:
    print("Downloading UCI dataset (~7MB)...")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    print(f"Downloaded {len(data) / 1e6:.1f} MB")
    return data


# ── Step 2: Extract files from zip ───────────────────────────────────────────

def extract_files(zip_bytes: bytes) -> dict:
    """
    Returns dict of {filename: file_bytes} for all .txt files in the zip.
    Handles nested zips (UCI sometimes wraps the data in a second zip).
    """
    files = {}

    def _read_zip(zb):
        with zipfile.ZipFile(io.BytesIO(zb)) as zf:
            for name in zf.namelist():
                if name.endswith('.zip'):
                    # nested zip — recurse into it
                    inner = zf.read(name)
                    files.update(_read_zip(inner))
                elif name.endswith('.txt'):
                    base = os.path.basename(name)
                    files[base] = zf.read(name)

    _read_zip(zip_bytes)
    print(f"Extracted {len(files)} .txt files: {sorted(files.keys())}")
    return files


# ── Step 3: Parse sensor files ────────────────────────────────────────────────

def parse_sensor(file_bytes: bytes, sensor_name: str) -> np.ndarray:
    """
    Each sensor file: rows = cycles, columns = time-steps.
    Returns shape (n_cycles, n_timesteps).
    """
    content = file_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep='\t', header=None)
    print(f"  {sensor_name}: {df.shape[0]} cycles × {df.shape[1]} time-steps")
    return df.values


def parse_profile(file_bytes: bytes) -> pd.DataFrame:
    """
    profile.txt: rows = cycles, columns = condition labels.
    """
    content = file_bytes.decode('utf-8')
    df = pd.read_csv(io.StringIO(content), sep='\t', header=None)
    df.columns = PROFILE_COLS
    return df


# ── Step 4: Compute per-cycle statistics ──────────────────────────────────────

def compute_features(sensor_arrays: dict) -> pd.DataFrame:
    """
    For each sensor, compute mean, std, min, max, range across time-steps.
    Returns DataFrame with 80 feature columns (16 sensors × 5 stats).
    """
    rows = []
    n_cycles = next(iter(sensor_arrays.values())).shape[0]

    for cycle_idx in range(n_cycles):
        row = {}
        for sensor_name, arr in sensor_arrays.items():
            cycle = arr[cycle_idx]
            row[f'{sensor_name}_mean']  = np.mean(cycle)
            row[f'{sensor_name}_std']   = np.std(cycle)
            row[f'{sensor_name}_min']   = np.min(cycle)
            row[f'{sensor_name}_max']   = np.max(cycle)
            row[f'{sensor_name}_range'] = np.max(cycle) - np.min(cycle)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Step 5: Build final CSV ───────────────────────────────────────────────────

def build_csv(features_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the 80 engineered features with the stable flag from profile.txt.
    The stable column comes from profile col 4 (1 = stable, 0 = transitional).
    """
    features_df = features_df.copy()
    features_df['stable'] = profile_df['stable'].values
    return features_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Download
    zip_bytes = download_dataset(UCI_URL)

    # 2. Extract
    files = extract_files(zip_bytes)

    # 3. Parse sensors
    print("\nParsing sensor files...")
    sensor_arrays = {}
    for sensor in SENSORS:
        filename = f"{sensor}.txt"
        if filename not in files:
            raise FileNotFoundError(
                f"Expected '{filename}' in dataset zip — not found. "
                f"Available files: {sorted(files.keys())}"
            )
        sensor_arrays[sensor] = parse_sensor(files[filename], sensor)

    # 4. Parse profile (labels)
    if 'profile.txt' not in files:
        raise FileNotFoundError("profile.txt not found in dataset zip.")
    profile_df = parse_profile(files['profile.txt'])
    print(f"\nProfile loaded: {len(profile_df)} cycles")
    print("Pump leakage class distribution:")
    print(profile_df['pump'].value_counts().sort_index().to_string())
    print("\nAccumulator pressure class distribution:")
    print(profile_df['accumulator'].value_counts().sort_index().to_string())

    # 5. Compute features
    print("\nComputing per-cycle statistics (this takes ~30 seconds)...")
    features_df = compute_features(sensor_arrays)
    print(f"Feature matrix: {features_df.shape}")

    # 6. Build and save full dataset
    full_df = build_csv(features_df, profile_df)
    full_df.to_csv(FULL_OUTPUT, index=False)
    print(f"\n✅ Full dataset saved → {FULL_OUTPUT}")
    print(f"   Shape: {full_df.shape[0]} cycles × {full_df.shape[1]} columns")

    # 7. Save 10-cycle sample with variety
    # Pick cycles that represent different fault conditions for a useful demo
    sample_indices = []

    # 2 healthy cycles (pump=0, accum=130)
    healthy = profile_df[
        (profile_df['pump'] == 0) & (profile_df['accumulator'] == 130)
    ].index[:2].tolist()
    sample_indices.extend(healthy)

    # 2 weak leakage cycles (pump=1)
    weak = profile_df[profile_df['pump'] == 1].index[:2].tolist()
    sample_indices.extend(weak)

    # 2 severe leakage cycles (pump=2)
    severe = profile_df[profile_df['pump'] == 2].index[:2].tolist()
    sample_indices.extend(severe)

    # 2 accumulator fault cycles (accum != 130)
    accum_fault = profile_df[profile_df['accumulator'] != 130].index[:2].tolist()
    sample_indices.extend(accum_fault)

    # 2 combined fault cycles
    combined = profile_df[
        (profile_df['pump'] > 0) & (profile_df['accumulator'] != 130)
    ].index[:2].tolist()
    sample_indices.extend(combined)

    # Deduplicate and take first 10
    seen = set()
    unique_indices = []
    for idx in sample_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)
        if len(unique_indices) == 10:
            break

    sample_df = full_df.iloc[unique_indices].reset_index(drop=True)
    sample_df.to_csv(SMALL_OUTPUT, index=False)
    print(f"\n✅ 10-cycle sample saved → {SMALL_OUTPUT}")
    print("   Sample composition:")
    sample_profile = profile_df.iloc[unique_indices][['pump', 'accumulator', 'stable']]
    pump_map = {0: 'No leakage', 1: 'Weak leakage', 2: 'Severe leakage'}
    accum_map = {90: 'Close to failure', 100: 'Severely reduced',
                 115: 'Slightly reduced', 130: 'Optimal pressure'}
    for i, (_, row) in enumerate(sample_profile.iterrows()):
        print(f"   Cycle {i+1}: Pump={pump_map.get(row['pump'], row['pump'])}, "
              f"Accum={accum_map.get(row['accumulator'], row['accumulator'])}, "
              f"Stable={row['stable']}")

    print("\n── How to use ────────────────────────────────────────────────────")
    print(f"Quick test:   Upload '{SMALL_OUTPUT}' to /analyse endpoint")
    print(f"Full test:    Upload '{FULL_OUTPUT}' to /analyse endpoint")
    print("Template:     GET /template to see the expected column format")
    print("──────────────────────────────────────────────────────────────────")


if __name__ == '__main__':
    main()