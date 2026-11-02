"""Aggregate mindLAMP exports into ema_prompts.csv and passive_daily.csv.

raw/ema_raw.csv: pid,wave,burst_day,prompt,date,stress,neg_affect,pos_affect,ideation_item (answered prompts only)
raw/sensor_raw.csv: pid,timestamp,screen_on_seconds,steps,lat,lon (records at any resolution)
raw/burst_schedule.csv: pid,wave,start_date; raw/exam_days.csv (optional): pid,date
Column names from the export are mapped to these in the platform's export step.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config

EARTH_M_PER_DEG = 111_320.0


def full_prompt_grid(schedule, burst_days, per_day=3):
    rows = [(s.pid, s.wave, d + 1, k + 1, (pd.Timestamp(s.start_date) + pd.Timedelta(days=d)).date().isoformat())
            for s in schedule.itertuples() for d in range(burst_days) for k in range(per_day)]
    return pd.DataFrame(rows, columns=["pid", "wave", "burst_day", "prompt", "date"])


def ema_table(ema, schedule, burst_days):
    grid = full_prompt_grid(schedule, burst_days)
    cols = ["pid", "wave", "burst_day", "prompt", "stress", "neg_affect", "pos_affect", "ideation_item"]
    m = grid.merge(ema[cols], on=["pid", "wave", "burst_day", "prompt"], how="left")
    m.insert(5, "answered", m["stress"].notna().astype(int))
    return m


def location_cells(lat, lon, grid_m):
    y = np.floor(lat * EARTH_M_PER_DEG / grid_m).astype("Int64")
    x = np.floor(lon * EARTH_M_PER_DEG * np.cos(np.radians(np.nanmean(lat))) / grid_m).astype("Int64")
    return y.astype(str) + "_" + x.astype(str)


def shannon(counts):
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def passive_table(sensor, schedule, exams, burst_days, cfg):
    s = sensor.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"])
    s["date"] = s["timestamp"].dt.date.astype(str)
    s["hour"] = s["timestamp"].dt.hour
    lo, hi = cfg["passive"]["night_hours"]
    s["night"] = (s["hour"] >= lo) & (s["hour"] < hi)
    has_gps = s["lat"].notna() & s["lon"].notna()
    s.loc[has_gps, "cell"] = location_cells(s.loc[has_gps, "lat"], s.loc[has_gps, "lon"], cfg["passive"]["location_grid_m"]).to_numpy()
    home = s[s["night"] & has_gps].groupby("pid")["cell"].agg(lambda c: c.mode().iat[0]).to_dict()
    s["home"] = s["cell"].eq(s["pid"].map(home))
    rows = []
    for sc in schedule.itertuples():
        person = s[s.pid == sc.pid]
        for d in range(burst_days):
            date = (pd.Timestamp(sc.start_date) + pd.Timedelta(days=d)).date().isoformat()
            x = person[person.date == date]
            gps = x[x["cell"].notna()]
            # share of located records at home scaled to a full day, so uneven sampling does not inflate hours
            home_h = 24 * gps["home"].mean() if len(gps) else np.nan
            rows.append(dict(pid=sc.pid, wave=sc.wave, burst_day=d + 1, date=date, screen_minutes=round(x["screen_on_seconds"].sum() / 60) if len(x) else np.nan,
                             night_screen_minutes=round(x.loc[x["night"], "screen_on_seconds"].sum() / 60) if len(x) else np.nan, steps=int(x["steps"].sum()) if len(x) else np.nan,
                             home_hours=round(home_h, 2) if len(gps) else np.nan, location_entropy=round(shannon(gps["cell"].value_counts()), 3) if len(gps) else np.nan,
                             exam_day=int(((exams.pid == sc.pid) & (exams.date == date)).any()) if len(exams) else 0))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = pd.read_csv(raw / "burst_schedule.csv")
    exams = pd.read_csv(raw / "exam_days.csv") if (raw / "exam_days.csv").exists() else pd.DataFrame(columns=["pid", "date"])
    ema_table(pd.read_csv(raw / "ema_raw.csv"), sched, cfg["burst_days"]).to_csv(out / "ema_prompts.csv", index=False)
    passive_table(pd.read_csv(raw / "sensor_raw.csv"), sched, exams, cfg["burst_days"], cfg).to_csv(out / "passive_daily.csv", index=False)
    print("EMA and passive tables ->", out)


if __name__ == "__main__":
    main()
