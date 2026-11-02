"""Can the labs finish one wave of one cohort inside the protocol's test window? Usage: python lab_capacity.py <out_csv>"""
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
lab = yaml.safe_load((ROOT / "config" / "study_config.yaml").read_text(encoding="utf-8"))["lab"]
N_CORE, N_EXT = 150, 150  # one cohort: half of the core and extended samples
EXT_MIN = 40              # extended visit: ECG only


def demand(wave_has_ext, n_labs, hours, weekdays, ext_room):
    """Weekdays needed for one wave, full sessions in whole slots; the extended visits share the labs unless a separate ECG room exists."""
    per_lab = int(hours * 60 // lab["session_minutes"])
    core_days = N_CORE / (per_lab * n_labs)
    ext_slots = int(hours * 60 // EXT_MIN)
    ext_days = 0 if not wave_has_ext else (N_EXT / (ext_slots * n_labs) if not ext_room else N_EXT / ext_slots)
    days = core_days + (0 if ext_room else ext_days)
    if ext_room:
        days = max(core_days, ext_days)
    return dict(sessions_per_lab_day=per_lab, weekdays_needed=round(days, 1), fits=days <= weekdays)


rows = []
win = lab["window_hours"][1] - lab["window_hours"][0]
for label, n, h, room in (("方案现状：2 间，14–18 时", lab["n_labs"], win, False), ("2 间，另设心电室", lab["n_labs"], win, True), ("3 间，14–18 时", 3, win, False), ("3 间，另设心电室", 3, win, True),
                          ("2 间，13–19 时，时段作协变量", lab["n_labs"], 6, False), ("2 间，13–19 时，另设心电室", lab["n_labs"], 6, True)):
    for wave, ext in (("T0/T3", True), ("T1/T2", False)):
        rows.append(dict(方案=label, 波次=wave, **demand(ext, n, h, lab["weekdays_per_wave"], room)))
T = pd.DataFrame(rows)
T.to_csv(sys.argv[1] if len(sys.argv) > 1 else "lab_capacity.csv", index=False, encoding="utf-8-sig")
print(T.to_string())
