"""Copy BrainVision recordings into a BIDS-EEG tree (framework 4.6). Usage: python export_bids.py --raw raw/eeg --out bids"""
import argparse
import json
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids

from common import load_config, parse_session

mne.set_log_level("ERROR")


def export_session(vhdr, root, cfg):
    pid, wave = parse_session(Path(vhdr).name)
    meta = json.loads(Path(vhdr).with_suffix(".json").read_text(encoding="utf-8"))
    raw = mne.io.read_raw_brainvision(vhdr, preload=False)
    raw.info["line_freq"] = 50
    path = BIDSPath(subject=pid.replace("_", ""), session=f"w{wave}", task="rest", datatype="eeg", root=root)
    write_raw_bids(raw, path, overwrite=True, allow_preload=False, verbose=False)
    return path.fpath, meta["conditions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw/eeg")
    ap.add_argument("--out", default="bids")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    files = sorted(Path(a.raw).glob("*_w*.vhdr"))
    for f in files:
        export_session(f, a.out, cfg)
    print(len(files), "EEG sessions ->", a.out)


if __name__ == "__main__":
    main()
