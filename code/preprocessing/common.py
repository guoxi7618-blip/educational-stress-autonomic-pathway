"""Shared helpers for preprocessing."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT =Path(__file__).resolve().parents[1]


def load_config(path=None):
    return yaml.safe_load(Path(path or ROOT / "config" / "study_config.yaml").read_text(encoding="utf-8"))


def correct_rr(rr_ms, window=11, max_rel_dev=0.20, lo=300, hi=2000):
    """Flag RR intervals outside a physiological range or >20% from the local median, and interpolate them.

    Returns corrected RR and the percentage of intervals corrected.
    """
    rr = np.asarray(rr_ms, dtype=float)
    med = pd.Series(rr).rolling(window, center=True, min_periods=3).median().to_numpy()
    flagged = (rr < lo) | (rr > hi) | (np.abs(rr - med) / med > max_rel_dev)
    good = np.where(~flagged)[0]
    out = rr.copy()
    if flagged.any() and len(good) >= 2:
        out[flagged] = np.interp(np.where(flagged)[0], good, rr[good])
    return out, float(100 * flagged.mean())


def rmssd_ms(rr_ms):
    return float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))


def windowed_rmssd_ms(rr_ms, seconds=300, min_beats=100):
    """Mean of RMSSD over consecutive windows, the way consumer devices summarize a night."""
    rr = np.asarray(rr_ms, dtype=float)
    win = np.floor(np.cumsum(rr) / 1000 / seconds).astype(int)
    vals = [rmssd_ms(rr[win == w]) for w in np.unique(win) if (win == w).sum() >= min_beats]
    return float(np.mean(vals)) if vals else float("nan")


def parse_session(name):
    """'P0001_w2...' -> ('P0001', 2)."""
    m = re.match(r"(?P<pid>[^_]+)_w(?P<wave>\d+)", name)
    if not m:
        raise ValueError(f"file name does not follow <pid>_w<wave>: {name}")
    return m["pid"], int(m["wave"])
