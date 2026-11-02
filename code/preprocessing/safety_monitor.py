"""Risk-trigger list required by the safety protocol (framework 4.8). Run after each data pull; it only lists people to contact.

Sources: PHQ-9 item 9, C-SSRS, EMA ideation item. Thresholds are in config section safety and wait for the clinical team.
Usage: python safety_monitor.py --data processed --out processed/safety_alerts.csv [--prefix PREFIX] [--open-only safety_events.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

from common import load_config


def alerts(items, clinician, ema, cfg):
    s = cfg["safety"]
    rows = []
    hit = items[items["phq_9"] >= s["phq9_item9_min"]]
    rows += [dict(pid=r.pid, wave=r.wave, trigger_source="phq9_item9", risk_level="low", value=r.phq_9) for r in hit.itertuples()]
    hit = clinician[(clinician.cssrs_active_ideation == 1) | (clinician.cssrs_intensity >= s["cssrs_intensity_min"])]
    for r in hit.itertuples():
        high = r.cssrs_active_ideation == 1 and r.cssrs_intensity >= s["cssrs_intensity_min"] + 2
        rows.append(dict(pid=r.pid, wave=r.wave, trigger_source="cssrs", risk_level="high" if high else "moderate", value=r.cssrs_intensity))
    answered = ema[(ema.answered == 1) & (ema.ideation_item >= s["ema_ideation_min"])]
    peak = answered.groupby(["pid", "wave"]).ideation_item.max().reset_index()
    rows += [dict(pid=r.pid, wave=r.wave, trigger_source="ema_ideation", risk_level="moderate" if r.ideation_item > s["ema_ideation_min"] else "low", value=r.ideation_item) for r in peak.itertuples()]
    return pd.DataFrame(rows, columns=["pid", "wave", "trigger_source", "risk_level", "value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed")
    ap.add_argument("--out", default="processed/safety_alerts.csv")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--config")
    a = ap.parse_args()
    d = Path(a.data)
    rd = lambda f: pd.read_csv(d / f"{a.prefix}{f}.csv")
    T = alerts(rd("scale_items"), rd("clinician_assessment"), rd("ema_prompts"), load_config(a.config))
    T.to_csv(a.out, index=False)
    print(len(T), "alerts,", T.pid.nunique(), "people ->", a.out)
    print(T.groupby(["trigger_source", "risk_level"]).size().to_string())


if __name__ == "__main__":
    main()
