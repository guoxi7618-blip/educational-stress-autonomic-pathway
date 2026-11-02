"""Convert BIOPAC recordings into ecg_lab.csv.

Input in raw/ecg: <pid>_w<wave>.acq (or .txt with columns ecg,resp,sync) and <pid>_w<wave>_segments.csv from stress_task.py.
Segment times count from the key press, so the recording must carry that moment: an AcqKnowledge event marker whose text contains
"sync", a non-zero sample in the sync column, or a sync_offset_s column in the segment log. Without one the session gets no HRV values.
"""
import argparse
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import welch

from common import correct_rr, load_config, parse_session, rmssd_ms


def load_recording(path, srate):
    path = Path(path)
    if path.suffix.lower() == ".acq":
        import bioread
        d = bioread.read_file(str(path))
        chans = {c.name.lower(): c for c in d.channels}
        ecg = next(c for n, c in chans.items() if "ecg" in n)
        resp = next((c for n, c in chans.items() if "resp" in n or "rsp" in n), None)
        marks = [m for m in d.event_markers if "sync" in (m.text or "").lower()]
        sync = marks[0].sample_index / float(d.samples_per_second) if marks else None
        return ecg.data, (resp.data if resp is not None else None), float(ecg.samples_per_second), sync
    df = pd.read_csv(path)
    sync = None
    if "sync" in df and (df["sync"] != 0).any():
        sync = int(np.argmax(df["sync"].to_numpy() != 0)) / srate
    return df["ecg"].to_numpy(), (df["resp"].to_numpy() if "resp" in df else None), srate, sync


def rr_metrics(ecg, srate, hf_band):
    """R peaks, local-median artefact correction; RMSSD on corrected intervals, HF power by Welch on the resampled tachogram."""
    clean = nk.ecg_clean(ecg, sampling_rate=srate, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=srate, method="neurokit")
    peaks = np.asarray(info["ECG_R_Peaks"])
    if len(peaks) < 10:
        return None
    # nk.signal_fixpeaks is avoided: it moves normal beats and under-reports corrections, which biases RMSSD downward
    rr, corrected_pct = correct_rr(np.diff(peaks) / srate * 1000)
    rmssd = rmssd_ms(rr)
    t = np.cumsum(rr) / 1000
    fs = 4.0
    grid = np.arange(t[0], t[-1], 1 / fs)
    rr_i = interp1d(t, rr, kind="cubic")(grid)
    f, p = welch(rr_i - rr_i.mean(), fs=fs, nperseg=min(256, len(rr_i)))
    band = (f >= hf_band[0]) & (f < hf_band[1])
    return dict(hr=float(60000 / rr.mean()), rmssd=rmssd, hf=float(np.trapezoid(p[band], f[band])), corrected_pct=corrected_pct)


def resp_rate(resp, srate):
    if resp is None:
        return np.nan
    rate = nk.rsp_rate(nk.rsp_clean(resp, sampling_rate=srate), sampling_rate=srate, method="xcorr")
    return float(np.nanmean(rate))


def process_session(rec_path, seg_path, cfg, tier):
    srate0 = cfg["ecg"]["sampling_rate_hz"]
    ecg, resp, srate, sync = load_recording(rec_path, srate0)
    segs = pd.read_csv(seg_path).set_index("segment")
    hf = cfg["ecg"]["hf_band_hz"]
    pid, wave = parse_session(Path(rec_path).name)
    out = dict(pid=pid, wave=wave, tier=tier, sampling_rate_hz=int(srate), sync_source="marker" if sync is not None else "none")
    if "session_start" in segs:
        start = pd.Timestamp(segs["session_start"].iloc[0])
        out["test_hour"] = round(start.hour + start.minute / 60, 2)
    if sync is None and "sync_offset_s" in segs:
        sync, out["sync_source"] = float(segs["sync_offset_s"].iloc[0]), "offset"
    if sync is None:  # cutting by an unknown offset would silently mix rest and task, so no values are produced
        for c in ("corrected_beats_pct", "rest_hr_bpm", "rmssd_ms", "ln_rmssd", "hf_power_ms2", "ln_rmssd_task", "ln_rmssd_recovery", "resp_rate_bpm"):
            out[c] = np.nan
        return out
    cut = lambda x, s: None if x is None else x[int((segs.loc[s, "onset_s"] + sync) * srate):int((segs.loc[s, "offset_s"] + sync) * srate)]
    rest = rr_metrics(cut(ecg, "rest"), srate, hf)
    max_c = cfg["ecg"]["max_corrected_beats_pct"]
    valid = rest is not None and rest["corrected_pct"] <= max_c
    out["corrected_beats_pct"] = np.nan if rest is None else round(rest["corrected_pct"], 2)
    out["rest_hr_bpm"] = np.nan if rest is None else round(rest["hr"], 1)
    out["rmssd_ms"] = round(rest["rmssd"], 2) if valid else np.nan
    out["ln_rmssd"] = round(np.log(rest["rmssd"]), 4) if valid else np.nan
    out["hf_power_ms2"] = round(rest["hf"], 1) if valid else np.nan
    for s, col in (("task", "ln_rmssd_task"), ("recovery", "ln_rmssd_recovery")):
        m = rr_metrics(cut(ecg, s), srate, hf)
        out[col] = round(np.log(m["rmssd"]), 4) if m and m["corrected_pct"] <= max_c else np.nan
    out["resp_rate_bpm"] = round(resp_rate(cut(resp, "rest"), srate), 1) if resp is not None else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/ecg")
    ap.add_argument("--participants", default="raw/participants.csv")
    ap.add_argument("--out", default="processed/ecg_lab.csv")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tiers = pd.read_csv(a.participants).set_index("pid")["tier"].to_dict()
    rows = []
    for rec in sorted(p for p in Path(a.raw).iterdir() if p.suffix.lower() in (".acq", ".txt") and p.stem.count("_") == 1):
        seg = rec.with_name(rec.stem + "_segments.csv")
        if not seg.exists():
            print("no segment log, skipped:", rec.name)
            continue
        pid, _ = parse_session(rec.name)
        rows.append(process_session(rec, seg, cfg, tiers.get(pid, "")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["pid", "wave"])
    table.to_csv(a.out, index=False)
    # sessions for manual review of R peaks (framework 4.6)
    r = cfg["ecg"]["review"]
    flag = (table.sync_source == "none") | (table.corrected_beats_pct > r["corrected_beats_pct"]) | ~table.rest_hr_bpm.between(*r["hr_bpm"]) | ~table.rmssd_ms.between(*r["rmssd_ms"])
    table[flag].to_csv(Path(a.out).with_name("ecg_review.csv"), index=False)
    print(len(rows), "ECG sessions ->", a.out, "|", int(flag.sum()), "for manual review")


if __name__ == "__main__":
    main()
