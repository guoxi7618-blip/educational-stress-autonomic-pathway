"""Session timeline for the ECG protocol: adaptation, rest, serial-subtraction stressor, recovery.

Sends LSL markers and writes a segment log used by preprocessing/ecg_hrv.py.
Start the BIOPAC recording first, then press the key shown on screen and mark the same moment in AcqKnowledge.
Usage: python stress_task.py --pid P0001 --wave 0 --out raw/ecg
"""
import argparse
import csv
import random
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SUBTRAHENDS = [7, 13, 17]


def session_plan(segments):
    """Protocol order (framework 4.5). The EEG block has no fixed length: the experimenter ends it after rest_eeg_protocol.py finishes."""
    return [dict(segment=name, seconds=None if name == "eeg_block" else segments[name]) for name in ("adaptation", "rest", "eeg_block", "task", "recovery")]


class Subtraction:
    """Adaptive serial subtraction: an error restarts the sequence and shortens the time allowed."""

    def __init__(self, seed, start_limit=5.0, min_limit=2.0):
        rng = random.Random(seed)
        self.step = rng.choice(SUBTRAHENDS)
        self.start = rng.randint(1500, 2500)
        self.current = self.start
        self.limit = start_limit
        self.min_limit = min_limit
        self.correct = self.errors = 0

    def respond(self, answer):
        if answer is not None and answer == self.current - self.step:
            self.current = answer
            self.correct += 1
            self.limit = min(self.limit + 0.05, 8.0)
            return True
        self.current = self.start
        self.errors += 1
        self.limit = max(self.limit - 0.25, self.min_limit)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--wave", type=int, required=True)
    ap.add_argument("--out", default="raw/ecg")
    ap.add_argument("--config", default=str(ROOT / "config" / "study_config.yaml"))
    a = ap.parse_args()
    from psychopy import core, event, visual  # imported here so the logic above stays importable without a display
    from pylsl import StreamInfo, StreamOutlet

    segs = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))["segments_seconds"]
    outlet = StreamOutlet(StreamInfo("StudyMarkers", "Markers", 1, 0, "string", f"task-{a.pid}-{a.wave}"))
    win = visual.Window(fullscr=True, color="black", units="height")
    msg = visual.TextStim(win, height=0.05, wrapWidth=1.2)

    def show(text):
        msg.text = text
        msg.draw()
        win.flip()

    show("实验员确认心电记录已开始后，按空格键开始；按键的同时在 AcqKnowledge 插入名为 sync 的事件标记。")
    event.waitKeys(keyList=["space"])
    clock = core.Clock()
    session_start = datetime.now().isoformat(timespec="seconds")
    plan = session_plan(segs)
    log, trials = [], []
    task = Subtraction(f"{a.pid}-{a.wave}")
    for seg in plan:
        outlet.push_sample([f"{seg['segment']}_start"])
        actual_on = clock.getTime()
        if seg["segment"] == "eeg_block":
            show("请保持坐姿。实验员运行静息脑电流程，结束后按空格键继续。")
            event.waitKeys(keyList=["space"])
        elif seg["segment"] == "task":
            typed, deadline = "", clock.getTime() + task.limit
            while clock.getTime() < actual_on + seg["seconds"]:
                show(f"从 {task.current} 连续减 {task.step}\n\n{typed}\n\n回车提交，答错或超时从头开始")
                for key in event.getKeys():
                    if key in list("0123456789"):
                        typed += key
                    elif key in ("num_0", "num_1", "num_2", "num_3", "num_4", "num_5", "num_6", "num_7", "num_8", "num_9"):
                        typed += key[-1]
                    elif key == "backspace":
                        typed = typed[:-1]
                    elif key == "return" and typed:
                        ok = task.respond(int(typed))
                        trials.append((round(clock.getTime(), 3), typed, int(ok), round(task.limit, 2)))
                        typed, deadline = "", clock.getTime() + task.limit
                if clock.getTime() > deadline:
                    task.respond(None)
                    trials.append((round(clock.getTime(), 3), "", 0, round(task.limit, 2)))
                    typed, deadline = "", clock.getTime() + task.limit
        else:
            show("+\n\n请保持安静坐姿，自然呼吸")
            core.wait(max(0.0, actual_on + seg["seconds"] - clock.getTime()))
        outlet.push_sample([f"{seg['segment']}_end"])
        log.append(dict(pid=a.pid, wave=a.wave, segment=seg["segment"], onset_s=round(actual_on, 3), offset_s=round(clock.getTime(), 3), session_start=session_start))
    show("本环节结束，请等待实验员。")
    core.wait(3)
    win.close()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.pid}_w{a.wave}_segments.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    with open(out / f"{a.pid}_w{a.wave}_task_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "answer", "correct", "limit_s"])
        w.writerows(trials)


if __name__ == "__main__":
    main()
