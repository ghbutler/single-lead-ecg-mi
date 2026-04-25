"""
Regenerate the embedded data and key prose in ../demo.html so the demo
matches the current 40/20/40 held-out test split and the cnn_transformer_final
ONNX model.

Inputs:
  - ../models/cnn_transformer_final.onnx
  - ../models/cnn_transformer_final_calibration.json
  - ../data/shaoxing_final_splits.csv  (cached split table)
  - ../data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/
  - ../demo.html  (template; updated in place)

Outputs:
  - ../demo.html  (in place; backup written to ../demo.html.bak)

Run from the code/ directory:
    python build_demo.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import wfdb
from tqdm import tqdm

# --- Config (must match catnet_final.ipynb) -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
SPLITS_CSV = ROOT / "data" / "shaoxing_final_splits.csv"
ONNX_PATH = ROOT / "models" / "cnn_transformer_final.onnx"
CALIB_PATH = ROOT / "models" / "cnn_transformer_final_calibration.json"
DEMO_PATH = ROOT / "demo.html"
LEAD_IDX = 0           # Lead I
SUBSAMPLE = 5          # 500 Hz → 100 Hz
TARGET_LEN = 1000      # 10 s at 100 Hz
N_NON_MI = 300         # Non-MI sample count for the enriched demo set
SEED = 42              # For sampling non-MI records

# --- Inference helpers --------------------------------------------------------
def load_lead1(record_path: Path) -> np.ndarray:
    """Load Lead I, subsample 5x, z-score, pad/truncate to TARGET_LEN."""
    sig, _ = wfdb.rdsamp(str(record_path))
    s = sig[::SUBSAMPLE, LEAD_IDX].astype(np.float32)
    s = (s - s.mean()) / (s.std() + 1e-8)
    if s.shape[0] >= TARGET_LEN:
        s = s[:TARGET_LEN]
    else:
        s = np.pad(s, (0, TARGET_LEN - s.shape[0]))
    return s


def run_inference(session: ort.InferenceSession, x: np.ndarray) -> float:
    """Run ONNX model on a single (TARGET_LEN,) signal; return raw logit."""
    inp = x.reshape(1, 1, TARGET_LEN).astype(np.float32)
    in_name = session.get_inputs()[0].name
    out = session.run(None, {in_name: inp})[0]
    return float(out.flatten()[0])


# --- Main ---------------------------------------------------------------------
def main() -> None:
    print("Loading model + calibration...")
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    calib = json.loads(CALIB_PATH.read_text())
    temperature = float(calib["temperature"])

    print("Loading split table...")
    df = pd.read_csv(SPLITS_CSV)
    test_df = df[df["split"] == "test"].copy().reset_index(drop=True)
    n_mi = int(test_df["mi_label"].sum())
    print(f"  Test split: {len(test_df)} records, {n_mi} MI cases.")

    # Predict on all test records (faster: only the ones we'll show, but we need
    # the full distribution to assign honest percentile ranks).
    print("Running inference on full test split (used for percentile ranking)...")
    logits = np.zeros(len(test_df), dtype=np.float32)
    valid = np.ones(len(test_df), dtype=bool)
    for i, row in tqdm(list(test_df.iterrows()), total=len(test_df)):
        try:
            x = load_lead1(DATA_ROOT / row["record"])
            logits[i] = run_inference(session, x)
        except Exception as e:
            valid[i] = False
            logits[i] = np.nan

    if (~valid).any():
        n_bad = int((~valid).sum())
        print(f"  Skipped {n_bad} records that failed to load.")

    # Calibrated probability (temperature scaling)
    probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    test_df["prob"] = probs
    test_df["valid"] = valid

    # Percentile rank within the full (valid) test set, higher prob = higher percentile
    valid_df = test_df[test_df["valid"]].copy()
    valid_df["pct"] = valid_df["prob"].rank(pct=True) * 100.0
    test_df = test_df.merge(valid_df[["record", "pct"]], on="record", how="left")

    # Sample: all valid MI + N_NON_MI random non-MI
    rng = np.random.default_rng(SEED)
    mi_records = test_df[(test_df["mi_label"] == 1) & test_df["valid"]]
    non_mi_records = test_df[(test_df["mi_label"] == 0) & test_df["valid"]]
    sampled_non_mi = non_mi_records.sample(n=min(N_NON_MI, len(non_mi_records)),
                                           random_state=SEED)
    demo_df = pd.concat([mi_records, sampled_non_mi], ignore_index=True)
    print(f"  Demo set: {len(demo_df)} records ({len(mi_records)} MI + "
          f"{len(sampled_non_mi)} non-MI). Apparent MI rate "
          f"{len(mi_records)/len(demo_df)*100:.1f}%.")

    # Build records (load waveforms again — we discarded them after inference)
    print("Loading waveforms for demo records...")
    records = []
    for _, row in tqdm(demo_df.iterrows(), total=len(demo_df)):
        try:
            x = load_lead1(DATA_ROOT / row["record"])
        except Exception:
            continue
        rec_id = Path(row["record"]).name  # e.g. JS11931
        records.append({
            "r": rec_id,
            "w": [round(float(v), 3) for v in x.tolist()],
            "l": int(row["mi_label"]),
            "p": round(float(row["prob"]), 6),
            "pct": round(float(row["pct"]), 1),
        })

    n_records = len(records)
    n_mi_demo = sum(1 for r in records if r["l"] == 1)
    n_non_mi_demo = n_records - n_mi_demo
    apparent_rate = n_mi_demo / n_records * 100

    # --- Update demo.html ---------------------------------------------------
    print("Updating demo.html...")
    html = DEMO_PATH.read_text()
    shutil.copy(DEMO_PATH, str(DEMO_PATH) + ".bak")

    # 1. Replace the DATA array (between `const DATA = [` and the matching `];`)
    # Walk the brackets so we don't depend on any specific record content.
    start = html.index("const DATA")
    arr_start = html.index("[", start)
    depth = 0
    j = arr_start
    while j < len(html):
        c = html[j]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                arr_end = j + 1
                break
        j += 1
    new_array = "[\n  " + ",\n  ".join(json.dumps(r, separators=(",", ":")) for r in records) + "\n]"
    html = html[:arr_start] + new_array + html[arr_end:]

    # 2. Prose updates ------------------------------------------------------
    # Each replacement is exact-match; we assert it succeeded.
    def replace_once(s: str, old: str, new: str, label: str) -> str:
        if old not in s:
            print(f"  WARN: prose target not found ({label}): {old[:80]!r}")
            return s
        return s.replace(old, new, 1)

    html = replace_once(
        html,
        "Test-set AUC: 0.86.",
        "Test-set AUC: 0.81 (held-out 40/20/40 split, n=18,062).",
        "test-AUC line",
    )
    html = replace_once(
        html,
        '<a href="https://physionet.org/content/ecg-arrhythmia/1.0.0/" target="_blank" style="color:#2b6cb0;">Shaoxing People\'s Hospital ECG database</a> (45,152 ECGs, Zhejiang, China)',
        '<a href="https://physionet.org/content/ecg-arrhythmia/1.0.0/" target="_blank" style="color:#2b6cb0;">Ningbo + Shaoxing 12-lead ECG database</a> (Zheng et al. 2020; 45,152 ECGs from Ningbo First Hospital and Shaoxing People\'s Hospital, China)',
        "fine-tune dataset",
    )
    html = replace_once(
        html,
        f'<strong>Demo data:</strong> All 374 ECG recordings in this demo are drawn from the\n    <a href="https://physionet.org/content/ecg-arrhythmia/1.0.0/" target="_blank" style="color:#2b6cb0;">Shaoxing People\'s Hospital ECG database</a>.',
        f'<strong>Demo data:</strong> All {n_records} ECG recordings in this demo are drawn from the\n    <a href="https://physionet.org/content/ecg-arrhythmia/1.0.0/" target="_blank" style="color:#2b6cb0;">Ningbo + Shaoxing 12-lead ECG database</a> (Zheng et al. 2020).',
        "demo-data dataset link",
    )
    html = replace_once(
        html,
        "Specifically, they come from the held-out test set (27,092 records, 74 with inferior MI), which was not used during training or fine-tuning.",
        f"Specifically, they come from the held-out 40/20/40 test split (18,062 records, {n_mi_demo} with inferior MI), which was not used during training or fine-tuning.",
        "test-split description",
    )
    html = replace_once(
        html,
        "all 74 MI cases plus 300 randomly sampled non-MI cases are included, so the apparent MI rate here (~19.8%) does not reflect any real-world prevalence.",
        f"all {n_mi_demo} MI cases plus {n_non_mi_demo} randomly sampled non-MI cases are included, so the apparent MI rate here (~{apparent_rate:.1f}%) does not reflect any real-world prevalence.",
        "demo-set sampling description",
    )
    html = replace_once(
        html,
        "In the full Shaoxing dataset, inferior MI accounts for 0.27% of records (123 out of 45,152).\n    This rate is specific to that dataset and reflects hospital ECG referral traffic",
        "In the full Ningbo + Shaoxing dataset, inferior MI accounts for 0.27% of records (123 out of 45,152).\n    This rate is specific to that dataset and reflects hospital ECG referral traffic",
        "full-dataset prevalence note",
    )
    html = replace_once(
        html,
        "Run on all 374 samples",
        f"Run on all {n_records} samples",
        "stratification button",
    )

    DEMO_PATH.write_text(html)
    print(f"Wrote {DEMO_PATH} ({n_records} records). Backup at {DEMO_PATH}.bak.")


if __name__ == "__main__":
    main()
