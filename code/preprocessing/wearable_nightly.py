"""Build wearable_nightly.csv from Oura sleep documents and overnight_device_agreement.csv from Polar H10 RR files.

Oura input: raw/oura/<pid>.json holding the API response, or fetch with --fetch and raw/oura/tokens.csv (pid,token).
Burst schedule: raw/burst_schedule.csv with pid,wave,start_date (first night of the 14-day burst).
Polar input: raw/polar/<pid>_w<wave>_d<burst_day>.txt with one RR interval in ms per line.
"""
import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import correct_rr, load_config, windowed_rmssd_ms


def fetch_oura(token, url, start, end):
    docs, next_token = [], None
    while True:
        q = {"start_date": start, "end_date": end}
        if next_token:
            q["next_token"] = next_token
        req = urllib.request.Request(url + "?" + urllib.parse.urlencode(q), headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        docs += page.get("data", [])
        next_token = page.get("next_token")
        if not next_token:
            return docs


def nights_from_docs(docs, pid, schedule, cfg):
    """One row per scheduled night; the main sleep period supplies the values."""
    w, f = cfg["wearable"], cfg["wearable"]["oura_fields"]
    days = {}
    for d in docs:  # a day can hold more than one long sleep; the longest is the main period
        if d.get(f["type"]) == w["oura_main_sleep_type"] and (d[f["day"]] not in days or (d.get(f["duration_s"]) or 0) > (days[d[f["day"]]].get(f["duration_s"]) or 0)):
            days[d[f["day"]]] = d
    rows = []
    for s in schedule[schedule.pid == pid].itertuples():
        for k in range(cfg["burst_days"]):
            date = (pd.Timestamp(s.start_date) + pd.Timedelta(days=k)).date().isoformat()
            doc = days.get(date)
            hrv = doc.get(f["hrv"]) if doc else None
            mins = (doc.get(f["duration_s"]) or 0) / 60 if doc else 0
            valid = bool(doc and hrv and hrv > 0 and mins >= w["min_sleep_minutes"])
            rows.append(dict(pid=pid, wave=s.wave, burst_day=k + 1, date=date, night_rmssd_ms=hrv if valid else np.nan, ln_night_rmssd=round(float(np.log(hrv)), 4) if valid else np.nan,
                             night_hr_bpm=doc.get(f["hr"]) if valid else np.nan, sleep_minutes=round(mins) if valid else np.nan, valid_night=int(valid)))
    return rows


def polar_rows(folder, wearable):
    ring = wearable.set_index(["pid", "wave", "burst_day"])["night_rmssd_ms"].to_dict()
    rows = []
    for f in sorted(Path(folder).glob("*_w*_d*.txt")):
        m = re.match(r"([^_]+)_w(\d+)_d(\d+)", f.stem)
        pid, wave, day = m[1], int(m[2]), int(m[3])
        rr, pct = correct_rr(np.loadtxt(f))
        rows.append(dict(pid=pid, wave=wave, burst_day=day, h10_night_rmssd_ms=round(windowed_rmssd_ms(rr), 2), ring_night_rmssd_ms=ring.get((pid, wave, day), np.nan), h10_corrected_beats_pct=round(pct, 2)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--fetch", action="store_true", help="pull documents from the Oura API first")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sched = pd.read_csv(raw / "burst_schedule.csv")
    if a.fetch:
        tokens = pd.read_csv(raw / "oura" / "tokens.csv").set_index("pid")["token"].to_dict()
        for pid, tok in tokens.items():
            s = sched[sched.pid == pid]
            end = (pd.Timestamp(s.start_date.max()) + pd.Timedelta(days=cfg["burst_days"])).date().isoformat()
            docs = fetch_oura(tok, cfg["wearable"]["oura_api_url"], s.start_date.min(), end)
            (raw / "oura" / f"{pid}.json").write_text(json.dumps(docs), encoding="utf-8")
    rows = []
    for pid in sched.pid.unique():
        f = raw / "oura" / f"{pid}.json"
        docs = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
        rows += nights_from_docs(docs.get("data", docs) if isinstance(docs, dict) else docs, pid, sched, cfg)
    wear = pd.DataFrame(rows)
    wear.to_csv(out / "wearable_nightly.csv", index=False)
    print(len(wear), "nights ->", out / "wearable_nightly.csv")
    if (raw / "polar").exists():
        cols = ["pid", "wave", "burst_day", "h10_night_rmssd_ms", "ring_night_rmssd_ms", "h10_corrected_beats_pct"]
        pd.DataFrame(polar_rows(raw / "polar", wear), columns=cols).to_csv(out / "overnight_device_agreement.csv", index=False)


if __name__ == "__main__":
    main()
