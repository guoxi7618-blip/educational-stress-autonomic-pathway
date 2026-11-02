"""Score item-level questionnaires into scale_scores.csv.

Input: scale_items.csv (pid,wave, items named <prefix>_<k>, psqi_total, caffeine_cups_day, exercise_hours_week);
optional therapy_self (self-reported treatment); optional safety_events.csv, whose first referral switches therapy_use on from that wave.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

PASS_THROUGH = ["visit_date", "days_to_next_final", "psqi_total", "caffeine_cups_day", "exercise_hours_week"]


def apply_reverse(items, reverse):
    out = items.copy()
    for col, (lo, hi) in reverse.items():
        out[col] = lo + hi - out[col]
    return out


def score(items, scales):
    scales = dict(scales)
    reverse = scales.pop("reverse_items", {}) or {}
    x = apply_reverse(items, reverse)
    out = x[["pid", "wave"] + [c for c in PASS_THROUGH if c in x]].copy()
    for name, s in scales.items():
        cols = [f"{s['prefix']}_{i + 1}" for i in range(s["n"])]
        if s["summary"] == "mean":
            out[name] = x[cols].mean(axis=1).round(3)
        else:
            # prorate up to 20% missing items; otherwise the total is missing
            enough = x[cols].notna().mean(axis=1) >= 0.8
            total = (x[cols].mean(axis=1) * s["n"]).where(enough)
            out[name] = (total * s.get("multiplier", 1)).round(3)
    return out


def add_therapy(scores, safety, items=None):
    """Therapy starts at an external referral or a self-reported treatment. A safety contact alone is recorded apart: most contacts are low-level checks."""
    first = lambda d: d.groupby("pid")["wave"].min() if len(d) else pd.Series(dtype=float)
    since = lambda s: (scores["wave"] >= scores["pid"].map(s).fillna(np.inf)).astype(int)
    referred = since(first(safety[safety.referred_external == 1])) if len(safety) else 0
    reported = since(first(items[items["therapy_self"] == 1])) if items is not None and "therapy_self" in items else 0
    scores["therapy_use"] = np.maximum(referred, reported)
    scores["safety_contact"] = since(first(safety[safety.contacted == 1])) if len(safety) else 0
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="raw/scale_items.csv")
    ap.add_argument("--safety", default="raw/safety_events.csv")
    ap.add_argument("--out", default="processed/scale_scores.csv")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    items = pd.read_csv(a.items)
    s = score(items, cfg["scales"])
    safety = pd.read_csv(a.safety) if Path(a.safety).exists() else pd.DataFrame()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    add_therapy(s, safety, items).sort_values(["pid", "wave"]).to_csv(a.out, index=False)
    print(len(s), "scored rows ->", a.out)


if __name__ == "__main__":
    main()
