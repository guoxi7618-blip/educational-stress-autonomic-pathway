"""RQ4: wearable agreement and out-of-sample incremental prediction. Usage: python prediction_agreement.py <processed_dir> <out_dir> [table_prefix]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("processed")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else D / "results"
OUT.mkdir(exist_ok=True, parents=True)
PREFIX = sys.argv[3] if len(sys.argv) > 3 else ""
rd = lambda f: pd.read_csv(D / f"{PREFIX}{f}.csv")
S, E, W, Pa, EMA, P, C = rd("scale_scores"), rd("ecg_lab"), rd("wearable_nightly"), rd("passive_daily"), rd("ema_prompts"), rd("participants"), rd("clinician_assessment")
CAL = rd("overnight_device_agreement")


def ccc(x, y):
    return 2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)


res = {}
# ---- agreement: ring vs chest-strap on the same nights ----
c = CAL.dropna()
c = c[c.h10_corrected_beats_pct < 5]
res["ring_vs_h10"] = dict(n_nights=len(c), n_people=int(c.pid.nunique()), ccc=float(ccc(c.ring_night_rmssd_ms, c.h10_night_rmssd_ms)),
                          mape=float((abs(c.ring_night_rmssd_ms - c.h10_night_rmssd_ms) / c.h10_night_rmssd_ms).mean() * 100))
# ---- agreement: burst-mean nocturnal ring vs daytime resting lab ECG (different physiological states) ----
wb = W[W.valid_night == 1].groupby(["pid", "wave"]).agg(n=("ln_night_rmssd", "size"), ln_ring=("ln_night_rmssd", "mean")).reset_index()
wb = wb[wb.n >= 5]
m = wb.merge(E[["pid", "wave", "ln_rmssd"]], on=["pid", "wave"]).dropna()
diff = m.ln_ring - m.ln_rmssd
res["ring_vs_lab_ecg"] = dict(n=len(m), ccc=float(ccc(m.ln_ring, m.ln_rmssd)), r=float(np.corrcoef(m.ln_ring, m.ln_rmssd)[0, 1]), bias=float(diff.mean()), loa_low=float(diff.mean() - 1.96 * diff.std()), loa_high=float(diff.mean() + 1.96 * diff.std()))

# ---- prediction of next-wave subclinical state ----
S["state"] = ((S.phq9_total >= 10) | (S.gad7_total >= 10)).astype(int)
base_cols = ["academic_stress", "employment_anxiety", "interpersonal_stress", "phq9_total", "gad7_total", "dass_stress", "psqi_total", "sam_challenge", "stress_mindset", "erq_reappraisal", "ders16_total"]
X = S[["pid", "wave", "state"] + base_cols].merge(P[["pid", "sex", "age", "ctq_total"]], on="pid")
X["female"] = (X.sex == "F").astype(int)
X = X.merge(E[["pid", "wave", "ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery"]], on=["pid", "wave"], how="left").merge(wb[["pid", "wave", "ln_ring"]], on=["pid", "wave"], how="left")
pas = Pa.groupby(["pid", "wave"]).agg(screen=("screen_minutes", "mean"), night_screen=("night_screen_minutes", "mean"), steps=("steps", "mean"), home=("home_hours", "mean"), entropy=("location_entropy", "mean")).reset_index()
em = EMA[EMA.answered == 1].groupby(["pid", "wave"]).agg(ema_stress=("stress", "mean"), ema_na=("neg_affect", "mean"), ema_na_sd=("neg_affect", "std"), ema_idea=("ideation_item", "mean")).reset_index()
X = X.merge(pas, on=["pid", "wave"], how="left").merge(em, on=["pid", "wave"], how="left")
nxt = S[["pid", "wave", "state"]].copy(); nxt["wave"] -= 1; nxt = nxt.rename(columns={"state": "y"})
Rows = X.merge(nxt, on=["pid", "wave"])
# framework 4.4: the state is subclinical, so next-wave assessments that met the immediate-referral standard are left out
Saf = rd("safety_events")
referred = set(zip(Saf[Saf.referred_external == 1].pid, Saf[Saf.referred_external == 1].wave - 1))
Rows = Rows[[k not in referred for k in zip(Rows.pid, Rows.wave)]].reset_index(drop=True)
SETS = {"selfreport": base_cols + ["female", "age", "ctq_total"]}
SETS["selfreport_physio"] = SETS["selfreport"] + ["ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery", "ln_ring"]
SETS["selfreport_physio_digital"] = SETS["selfreport_physio"] + ["screen", "night_screen", "steps", "home", "entropy", "ema_stress", "ema_na", "ema_na_sd", "ema_idea"]


def cv_pred(d, cols, seed=0):
    pids = d.pid.unique()
    rng = np.random.default_rng(seed)
    grp = dict(zip(rng.permutation(pids), np.arange(len(pids)) % 5))
    f = d.pid.map(grp).to_numpy()
    p = np.zeros(len(d))
    for k in range(5):
        tr, te = f != k, f == k
        mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        mdl.fit(d.loc[tr, cols], d.loc[tr, "y"])
        p[te] = mdl.predict_proba(d.loc[te, cols])[:, 1]
    return p


def table(d, label):
    y = d.y.to_numpy()
    preds = {k: cv_pred(d, v) for k, v in SETS.items()}
    rows = [dict(population=label, model=k, n=len(d), events=int(y.sum()), auc=roc_auc_score(y, p), brier=brier_score_loss(y, p)) for k, p in preds.items()]
    # person-level bootstrap of AUC increments
    pid = d.pid.to_numpy()
    uniq = np.unique(pid)
    idx = {u: np.where(pid == u)[0] for u in uniq}
    rng = np.random.default_rng(11)
    dl = []
    for _ in range(1000):
        s = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        if y[s].min() == y[s].max():
            continue
        a0 = roc_auc_score(y[s], preds["selfreport"][s])
        dl.append((roc_auc_score(y[s], preds["selfreport_physio"][s]) - a0, roc_auc_score(y[s], preds["selfreport_physio_digital"][s]) - a0))
    dl = np.array(dl)
    for r, j in zip(rows[1:], (0, 1)):
        r.update(delta_auc=dl[:, j].mean(), delta_lo=np.quantile(dl[:, j], 0.025), delta_hi=np.quantile(dl[:, j], 0.975))
    return rows


rows = table(Rows, "all_transitions") + table(Rows[Rows.state == 0].reset_index(drop=True), "new_onset_only")
T = pd.DataFrame(rows)
T.round(3).to_csv(OUT / "t7_prediction.csv", index=False)
(OUT / "t7_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
print(T.round(3).to_string())
