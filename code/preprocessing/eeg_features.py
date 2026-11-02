"""Convert BrainVision recordings into eeg_lab.csv (posterior alpha, aperiodic slope, FAA).

Input in raw/eeg: <pid>_w<wave>.vhdr (with .eeg and .vmrk) and <pid>_w<wave>.json from rest_eeg_protocol.py.
Primary mediator: aperiodic-adjusted posterior alpha power in the eyes-closed block.
"""
import argparse
import json
import zlib
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from fooof import FOOOF

from common import load_config, parse_session

mne.set_log_level("ERROR")


def load_recording(vhdr, e):
    """Read the recording and restore the online reference as a channel; average referencing waits until bad channels are known."""
    raw = mne.io.read_raw_brainvision(vhdr, preload=True)
    impedances = {c: v["imp"] for c, v in (getattr(raw, "impedances", None) or {}).items() if v.get("imp") is not None}
    srate = raw.info["sfreq"]
    names = [c for c in raw.ch_names if raw.get_channel_types([c])[0] == "eeg"]
    raw.pick(names)
    if e["online_reference"] not in names:
        raw = mne.add_reference_channels(raw, e["online_reference"])
    raw.set_montage("standard_1020", on_missing="warn")
    return raw, srate, impedances, len(names)


def block_times(raw, order, seconds, codes):
    """Block start and stop in seconds from TriggerBox markers; falls back to the logged order and duration."""
    times = {}
    for label in order:
        tag = f"Stimulus/S{codes[label]:>3}"
        hit = [o for o, d in zip(raw.annotations.onset, raw.annotations.description) if d == tag]
        if hit:
            times[label] = (hit[0], hit[0] + seconds)
    for i, label in enumerate(order):
        times.setdefault(label, (i * seconds, (i + 1) * seconds))
    return times


def find_bad_channels(raw, reference, flat_uv=0.5, max_uv=100.0, z_thresh=5.0, min_mad=0.3):
    """Flat or extreme channels, plus noise outliers judged in 20-45 Hz so that strong alpha is not mistaken for noise.

    Runs before average referencing, which would hide flat channels and spread bad ones; the online reference is all zeros here and is skipped.
    """
    names = [c for c in raw.ch_names if c != reference]
    sd = raw.get_data(picks=names).std(axis=1) * 1e6
    high = raw.copy().pick(names).filter(20.0, 45.0, fir_design="firwin").get_data()
    v = np.log(np.var(high, axis=1) + 1e-30)
    med = np.median(v)
    mad = max(np.median(np.abs(v - med)) * 1.4826, min_mad)
    return [c for c, s, x in zip(names, sd, v) if s < flat_uv or s > max_uv or (x - med) / mad > z_thresh]


def remove_ocular(raw, seed, max_components):
    """ICA on the 1 Hz high-passed recording, removing components correlated with frontal channels."""
    frontal = [c for c in ("Fp1", "Fp2") if c in raw.ch_names and c not in raw.info["bads"]]
    n = min(len(raw.ch_names) - len(raw.info["bads"]) - 1, max_components)
    ica = mne.preprocessing.ICA(n_components=n, method="fastica", random_state=seed, max_iter=1000)
    ica.fit(raw)
    bad = set()
    for ch in frontal:
        hits, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=3.0)
        bad.update(hits)
    ica.exclude = sorted(bad)
    return ica.apply(raw.copy()), len(bad)


def clean_epochs(raw, tmin, tmax, epoch_s, reject_uv):
    seg = raw.copy().crop(tmin, min(tmax, raw.times[-1]))
    ep = mne.make_fixed_length_epochs(seg, duration=epoch_s, preload=True)
    ptp = np.ptp(ep.get_data(), axis=2).max(axis=1) * 1e6
    return ep[ptp <= reject_uv]


def alpha_features(ep, cfg):
    srate = ep.info["sfreq"]
    roi = [c for c in cfg["posterior_roi"] if c in ep.ch_names]
    n_fft = int(srate * cfg["epoch_seconds"])
    psd = ep.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_fft, n_per_seg=n_fft)
    p = psd.get_data().mean(0)
    f = psd.freqs
    lo, hi = cfg["alpha_band_hz"]
    band = (f >= lo) & (f <= hi)
    spec = p[[ep.ch_names.index(c) for c in roi]].mean(0) * 1e12  # V^2/Hz -> uV^2/Hz
    fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, spec, cfg["fooof_freq_range"])
    peaks = fm.get_params("peak_params")
    peaks = np.atleast_2d(peaks)
    in_alpha = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)] if peaks.size else np.empty((0, 3))
    if len(in_alpha):
        best = in_alpha[np.argmax(in_alpha[:, 1])]
        iapf, periodic = best[0], best[1]
    else:
        iapf, periodic = np.nan, 0.0
    r, l = cfg["faa_pair"]
    faa = np.nan
    if r in ep.ch_names and l in ep.ch_names:
        faa = np.log(p[ep.ch_names.index(r)][band].mean()) - np.log(p[ep.ch_names.index(l)][band].mean())
    return dict(alpha_periodic_log=float(periodic), alpha_total_log=float(np.log10(spec[band].mean())), aperiodic_exponent=float(fm.get_params("aperiodic_params", "exponent")),
                iapf_hz=float(iapf), faa_f4_f3=float(faa), fit_r2=float(fm.r_squared_))


def channel_table(ep, cfg, pid, wave, condition, impedances, interpolated):
    """One spectral parameterization per channel, in the same units as the session features."""
    srate = ep.info["sfreq"]
    n_fft = int(srate * cfg["epoch_seconds"])
    psd = ep.compute_psd(method="welch", fmin=1, fmax=45, n_fft=n_fft, n_per_seg=n_fft)
    p, f = psd.get_data().mean(0) * 1e12, psd.freqs
    lo, hi = cfg["alpha_band_hz"]
    band = (f >= lo) & (f <= hi)
    rows = []
    for name, spec in zip(ep.ch_names, p):
        fm = FOOOF(peak_width_limits=(1, 8), max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
        fm.fit(f, spec, cfg["fooof_freq_range"])
        peaks = np.atleast_2d(fm.get_params("peak_params"))
        alpha = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)] if peaks.size else np.empty((0, 3))
        best = alpha[np.argmax(alpha[:, 1])] if len(alpha) else (np.nan, 0.0, np.nan)
        rows.append(dict(pid=pid, wave=wave, condition=condition, channel=name, alpha_periodic_log=float(best[1]), alpha_total_log=float(np.log10(spec[band].mean())), aperiodic_exponent=float(fm.get_params("aperiodic_params", "exponent")),
                         aperiodic_offset=float(fm.get_params("aperiodic_params", "offset")), iapf_hz=float(best[0]), alpha_bandwidth_hz=float(best[2]), fit_r2=float(fm.r_squared_), impedance_kohm=impedances.get(name, np.nan), interpolated=int(name in interpolated)))
    return rows


def process_session(vhdr, cfg):
    e = cfg["eeg"]
    vhdr = Path(vhdr)
    meta = json.loads(vhdr.with_suffix(".json").read_text(encoding="utf-8"))
    pid, wave = parse_session(vhdr.name)
    raw, srate, impedances, n_rec = load_recording(vhdr, e)
    impedance = float(np.median(list(impedances.values()))) if impedances else np.nan
    seconds = meta["condition_seconds"]
    times = block_times(raw, meta["conditions"], seconds, e["trigger_codes"])
    # share of the eyes-closed block missing from the file (recording stopped early or gaps)
    start, stop = times["eyes_closed"]
    loss = 100 * max(0.0, (stop - min(stop, raw.times[-1])) / seconds)
    raw.filter(*e["bandpass_hz"], fir_design="firwin")
    raw.resample(e["analysis_sampling_rate_hz"])
    raw.info["bads"] = find_bad_channels(raw, e["online_reference"])
    n_interp = len(raw.info["bads"])
    interpolated = list(raw.info["bads"])
    out = dict(pid=pid, wave=wave, n_channels_recorded=n_rec, sampling_rate_hz=round(srate), n_channels_interpolated=n_interp, median_impedance_kohm=impedance, data_loss_pct=round(loss, 2))
    empty = dict(alpha_periodic_log=np.nan, alpha_total_log=np.nan, aperiodic_exponent=np.nan, iapf_hz=np.nan, faa_f4_f3=np.nan, clean_seconds=0, n_ica_removed=np.nan)
    if n_interp > e["max_interpolated_channels"] or loss > e["max_data_loss_pct"]:
        return {**out, **empty}, []
    raw.set_eeg_reference("average", projection=False)  # bad channels excluded from the average
    cleaned, n_ica = remove_ocular(raw, zlib.crc32(f"{pid}-{wave}".encode()) % (2**31), e["ica_max_components"])
    if cleaned.info["bads"]:
        cleaned.interpolate_bads(reset_bads=True)
    cleaned.set_eeg_reference("average", projection=False)
    ep = clean_epochs(cleaned, start, stop, e["epoch_seconds"], e["reject_peak_to_peak_uv"])
    clean_s = len(ep) * e["epoch_seconds"]
    out.update(clean_seconds=clean_s, n_ica_removed=n_ica)
    if clean_s < e["min_clean_seconds"]:
        return {**out, **{k: v for k, v in empty.items() if k not in ("clean_seconds", "n_ica_removed")}}, []
    rows = channel_table(ep, e, pid, wave, "eyes_closed", impedances, interpolated)
    open_start, open_stop = times["eyes_open"]
    ep_open = clean_epochs(cleaned, open_start, open_stop, e["epoch_seconds"], e["reject_peak_to_peak_uv"])
    features = alpha_features(ep, e)
    if len(ep_open):
        rows += channel_table(ep_open, e, pid, wave, "eyes_open", impedances, interpolated)
        features["alpha_periodic_log_eo"] = alpha_features(ep_open, e)["alpha_periodic_log"]
    return {**out, **features}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/eeg")
    ap.add_argument("--out", default="processed/eeg_lab.csv")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rows, channels = [], []
    for f in sorted(Path(a.raw).glob("*_w*.vhdr")):
        try:
            session, per_channel = process_session(f, cfg)
            rows.append(session)
            channels += per_channel
        except Exception as err:
            print("failed:", f.name, err)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["pid", "wave"]).round(4).to_csv(out, index=False)
    pd.DataFrame(channels).round(4).to_csv(out.with_name("eeg_channel_level.csv"), index=False)
    print(len(rows), "EEG sessions ->", out)


if __name__ == "__main__":
    main()
