"""Run preprocessing, quality control and analysis in order.

  python run_pipeline.py --raw raw --processed processed --results results
  python run_pipeline.py --processed processed --results results --steps qc analysis
Set RSCRIPT to the Rscript executable if it is not on PATH, and BAYES_PYTHON to the Python of the environment holding JAX and NumPyro if it is not this one.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRE = ROOT / "preprocessing"
ANA = ROOT / "analysis"
COPY_THROUGH = ["participants.csv", "screening.csv", "scale_items.csv", "clinician_assessment.csv", "safety_events.csv", "academic_job_logs.csv"]


def run(cmd, cwd=None):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--processed", default="processed")
    ap.add_argument("--results", default="results")
    ap.add_argument("--steps", nargs="+", default=["preprocess", "qc", "analysis"], choices=["preprocess", "qc", "analysis"])
    ap.add_argument("--fetch-oura", action="store_true")
    ap.add_argument("--prefix", default="", help="table-name prefix for the qc and analysis steps")
    a = ap.parse_args()
    raw, proc, res = Path(a.raw).resolve(), Path(a.processed).resolve(), Path(a.results).resolve()
    py = sys.executable
    if "preprocess" in a.steps:
        proc.mkdir(parents=True, exist_ok=True)
        for f in COPY_THROUGH:
            if (raw / f).exists():
                shutil.copy2(raw / f, proc / f)
        run([py, PRE / "score_scales.py", "--items", raw / "scale_items.csv", "--safety", raw / "safety_events.csv", "--out", proc / "scale_scores.csv"])
        run([py, PRE / "ecg_hrv.py", "--raw", raw / "ecg", "--participants", raw / "participants.csv", "--out", proc / "ecg_lab.csv"])
        run([py, PRE / "eeg_features.py", "--raw", raw / "eeg", "--out", proc / "eeg_lab.csv"])
        run([py, PRE / "wearable_nightly.py", "--raw", raw, "--out", proc] + (["--fetch"] if a.fetch_oura else []))
        run([py, PRE / "ema_passive.py", "--raw", raw, "--out", proc])
        run([py, PRE / "export_bids.py", "--raw", raw / "eeg", "--out", proc / "bids"])
    if "qc" in a.steps:
        run([py, PRE / "qc_report.py", "--data", proc, "--prefix", a.prefix])
    if "analysis" in a.steps:
        res.mkdir(parents=True, exist_ok=True)
        run([py, PRE / "safety_monitor.py", "--data", proc, "--out", res / "safety_alerts.csv", "--prefix", a.prefix])
        run([os.environ.get("RSCRIPT", "Rscript"), ANA / "main_models.R", proc, res, a.prefix])
        run([py, ANA / "prediction_agreement.py", proc, res, a.prefix])
        run([os.environ.get("RSCRIPT", "Rscript"), ANA / "supplement_models.R", proc, res, a.prefix])
        run([py, ANA / "supplement_prediction.py", proc, res, a.prefix])
        run([py, ANA / "lab_capacity.py", res / "lab_capacity.csv"])
        bayes = os.environ.get("BAYES_PYTHON", py)
        run([bayes, ANA / "ri_lta.py", proc, res, a.prefix])
        run([bayes, ANA / "dsem_burst.py", proc, res, a.prefix])


if __name__ == "__main__":
    main()
