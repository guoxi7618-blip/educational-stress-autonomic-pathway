"""Run the resting EEG blocks while BrainVision Recorder saves the actiCHamp Plus recording.

Before running: check impedances in Recorder, then start saving to <pid>_w<wave> in the raw/eeg folder.
Markers go through a Brain Products TriggerBox (--trigger-port); the session log records the block order either way.
Usage: python rest_eeg_protocol.py --pid P0001 --wave 0 --out raw/eeg --trigger-port COM5
"""
import argparse
import json
import random
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def condition_order(pid, wave):
    """Eyes-open/closed order, randomized but reproducible per session."""
    rng = random.Random(f"{pid}-{wave}")
    order = ["eyes_open", "eyes_closed"]
    rng.shuffle(order)
    return order


class Trigger:
    """Pulse an 8-bit code on the TriggerBox's virtual serial port; logs only when no port is given."""

    def __init__(self, port=None):
        self.port = None
        if port:
            import serial
            self.port = serial.Serial(port)
            self.port.write(bytes([0]))

    def send(self, code):
        if self.port:
            self.port.write(bytes([code]))
            time.sleep(0.01)
            self.port.write(bytes([0]))

    def close(self):
        if self.port:
            self.port.close()


INSTRUCTIONS = {"eyes_open": "请睁眼注视前方十字，保持放松，尽量不要眨眼或移动。", "eyes_closed": "请闭上眼睛，保持清醒和放松，不要移动。"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--out", default="raw/eeg")
    ap.add_argument("--trigger-port")
    ap.add_argument("--config", default=str(ROOT / "config" / "study_config.yaml"))
    ap.add_argument("--no-prompt", action="store_true", help="run blocks without waiting for Enter")
    ap.add_argument("--block-seconds", type=int)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))["eeg"]
    codes = cfg["trigger_codes"]
    seconds = a.block_seconds or cfg["rest_condition_seconds"]
    order = condition_order(a.pid, a.wave)
    wait = (lambda _: None) if a.no_prompt else input

    print(f"文件名应为 {a.pid}_w{a.wave}；阻抗须低于 {cfg['max_impedance_kohm']} kΩ。")
    wait("确认 Recorder 已开始保存后按回车：")
    trig = Trigger(a.trigger_port)
    blocks = []
    try:
        for label in order:
            print(INSTRUCTIONS[label])
            wait(f"{label} {seconds} 秒，被试准备好后按回车：")
            trig.send(codes[label])
            t0 = time.time()
            time.sleep(seconds)
            trig.send(codes["block_end"])
            blocks.append(dict(condition=label, started_unix=t0, seconds=seconds))
    finally:
        trig.close()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(pid=a.pid, wave=a.wave, conditions=order, condition_seconds=seconds, blocks=blocks, trigger_port=a.trigger_port)
    (out / f"{a.pid}_w{a.wave}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("block log saved:", out / f"{a.pid}_w{a.wave}.json")


if __name__ == "__main__":
    main()
