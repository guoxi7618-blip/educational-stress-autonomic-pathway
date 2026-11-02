"""Check processed tables before analysis: ranges, duplicates, retention, device validity. Exits non-zero on hard errors."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from common import load_config

RANGES = {"scale_scores": {"phq9_total": (0, 27), "gad7_total": (0, 21), "dass_stress": (0, 42), "psqi_total": (0, 21), "ders16_total": (16, 80), "academic_stress": (1, 5), "employment_anxiety": (1, 5), "interpersonal_stress": (1, 5)},
          "ecg_lab": {"ln_rmssd": (1.0, 6.0), "rest_hr_bpm": (35, 130), "corrected_beats_pct": (0, 100)},
          "eeg_lab": {"data_loss_pct": (0, 100), "median_impedance_kohm": (0, 30), "aperiodic_exponent": (0, 4), "iapf_hz": (7, 14)},
          "clinician_assessment": {"cgi_s": (1, 7), "cssrs_intensity": (0, 25)},
          "ema_prompts": {"stress": (1, 7), "neg_affect": (1, 7), "pos_affect": (1, 7), "ideation_item": (0, 6)}}
KEYS = {"scale_scores": ["pid", "wave"], "ecg_lab": ["pid", "wave"], "eeg_lab": ["pid", "wave"], "clinician_assessment": ["pid", "wave"], "wearable_nightly": ["pid", "wave", "burst_day"], "passive_daily": ["pid", "wave", "burst_day"], "ema_prompts": ["pid", "wave", "burst_day", "prompt"]}


def check(folder, cfg, prefix=""):
    errors, info = [], {}
    tables = {f.stem[len(prefix):]: pd.read_csv(f) for f in Path(folder).glob(f"{prefix}*.csv")}
    for name, key in KEYS.items():
        if name not in tables:
            errors.append(f"missing table: {name}")
            continue
        dup = int(tables[name].duplicated(key).sum())
        if dup:
            errors.append(f"{name}: {dup} duplicated keys")
    for name, cols in RANGES.items():
        if name in tables:
            for c, (lo, hi) in cols.items():
                if c in tables[name]:
                    bad = int(((tables[name][c] < lo) | (tables[name][c] > hi)).sum())
                    if bad:
                        errors.append(f"{name}.{c}: {bad} values outside [{lo}, {hi}]")
    if "scale_scores" in tables:
        info["n_by_wave"] = tables["scale_scores"].groupby("wave").pid.nunique().to_dict()
    if "ecg_lab" in tables:
        info["ecg_valid"] = round(float(tables["ecg_lab"].ln_rmssd.notna().mean()), 3)
        if "sync_source" in tables["ecg_lab"]:
            unsynced = int((tables["ecg_lab"].sync_source == "none").sum())
            if unsynced:
                errors.append(f"ecg_lab: {unsynced} sessions without a sync marker")
    if "eeg_lab" in tables:
        info["eeg_valid"] = round(float(tables["eeg_lab"].alpha_periodic_log.notna().mean()), 3)
    if "wearable_nightly" in tables:
        w = tables["wearable_nightly"]
        info["valid_nights"] = round(float(w.valid_night.mean()), 3)
        per = w[w.valid_night == 1].groupby(["pid", "wave"]).size()
        info["bursts_meeting_minimum"] = round(float((per >= cfg["wearable"]["min_valid_nights_per_burst"]).mean()), 3)
    if "ema_prompts" in tables:
        info["ema_answered"] = round(float(tables["ema_prompts"].answered.mean()), 3)
    return errors, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed")
    ap.add_argument("--prefix", default="", help="table-name prefix")
    ap.add_argument("--config")
    a = ap.parse_args()
    errors, info = check(a.data, load_config(a.config), a.prefix)
    print(json.dumps(info, indent=1, default=str))
    for e in errors:
        print("ERROR:", e)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
