"""Internal-external validation of the next-wave prediction models, holding out one school or one start cohort at a time.

Features, exclusions and the model match supplement_prediction.py.
Usage: python internal_external_validation.py <processed_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
rd = lambda f: pd.read_csv(D / f"{f}.csv")
S, E, W, Pa, EMA, P, SAF = rd("scale_scores"), rd("ecg_lab"), rd("wearable_nightly"), rd("passive_daily"), rd("ema_prompts"), rd("participants"), rd("safety_events")

wb = W[W.valid_night == 1].groupby(["pid", "wave"]).agg(n=("ln_night_rmssd", "size"), ln_ring=("ln_night_rmssd", "mean")).reset_index()
wb = wb[wb.n >= 5]
S["state"] = ((S.phq9_total >= 10) | (S.gad7_total >= 10)).astype(int)
base = ["academic_stress", "employment_anxiety", "interpersonal_stress", "phq9_total", "gad7_total", "dass_stress", "psqi_total", "sam_challenge", "stress_mindset", "erq_reappraisal", "ders16_total"]
X = S[["pid", "wave", "state"] + base].merge(P[["pid", "sex", "age", "ctq_total", "school", "cohort"]], on="pid")
X["female"] = (X.sex == "F").astype(int)
X = X.merge(E[["pid", "wave", "ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery"]], on=["pid", "wave"], how="left").merge(wb[["pid", "wave", "ln_ring"]], on=["pid", "wave"], how="left")
pas = Pa.groupby(["pid", "wave"]).agg(screen=("screen_minutes", "mean"), night_screen=("night_screen_minutes", "mean"), steps=("steps", "mean"), home=("home_hours", "mean"), entropy=("location_entropy", "mean")).reset_index()
em = EMA[EMA.answered == 1].groupby(["pid", "wave"]).agg(ema_stress=("stress", "mean"), ema_na=("neg_affect", "mean"), ema_na_sd=("neg_affect", "std"), ema_idea=("ideation_item", "mean")).reset_index()
X = X.merge(pas, on=["pid", "wave"], how="left").merge(em, on=["pid", "wave"], how="left")
nxt = S[["pid", "wave", "state"]].copy()
nxt["wave"] -= 1
R = X.merge(nxt.rename(columns={"state": "y"}), on=["pid", "wave"])
ref = set(zip(SAF[SAF.referred_external == 1].pid, SAF[SAF.referred_external == 1].wave - 1))
R = R[[k not in ref for k in zip(R.pid, R.wave)]].reset_index(drop=True)
SELF = base + ["female", "age", "ctq_total"]
PHY = ["ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery", "ln_ring"]
PAS = ["screen", "night_screen", "steps", "home", "entropy"]
EM = ["ema_stress", "ema_na", "ema_na_sd", "ema_idea"]
SETS = {"Self-report": SELF, "Self-report + physiology": SELF + PHY, "Self-report + passive sensing": SELF + PAS, "Self-report + momentary assessment": SELF + EM, "All features": SELF + PHY + PAS + EM}


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def slope(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    return float(sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1])


def holdout(d, by, cols):
    p = np.zeros(len(d))
    for g in d[by].unique():
        te = (d[by] == g).to_numpy()
        p[te] = model().fit(d.loc[~te, cols], d.loc[~te, "y"]).predict_proba(d.loc[te, cols])[:, 1]
    return p


rows = []
for pop, d in (("All transitions", R), ("New onset", R[R.state == 0].reset_index(drop=True))):
    y, pid = d.y.to_numpy(), d.pid.to_numpy()
    uniq = np.unique(pid)
    idx = {u: np.where(pid == u)[0] for u in uniq}
    for by, scheme in (("school", "Leave one school out"), ("cohort", "Leave one cohort out")):
        pred = {k: holdout(d, by, v) for k, v in SETS.items()}
        for k, p in pred.items():
            rng = np.random.default_rng(11)
            dl = []
            for _ in range(1000):
                b = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
                dl.append(roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], pred["Self-report"][b]))
            rows.append(dict(population=pop, scheme=scheme, held_out="Pooled", model=k, n=len(d), events=int(y.sum()), auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), calib_slope=slope(y, p), delta_auc=float(np.mean(dl)), delta_lo=float(np.quantile(dl, .025)), delta_hi=float(np.quantile(dl, .975))))
            for g in sorted(d[by].unique()):
                te = (d[by] == g).to_numpy()
                rows.append(dict(population=pop, scheme=scheme, held_out=g, model=k, n=int(te.sum()), events=int(y[te].sum()), auc=roc_auc_score(y[te], p[te]), brier=brier_score_loss(y[te], p[te]), calib_slope=slope(y[te], p[te]), delta_auc=roc_auc_score(y[te], p[te]) - roc_auc_score(y[te], pred["Self-report"][te])))
T = pd.DataFrame(rows)
T.round(4).to_csv(OUT / "s37_internal_external.csv", index=False, encoding="utf-8-sig")
print(len(R), R.y.sum())
print(T[T.held_out == "Pooled"].drop(columns=["held_out"]).round(3).to_string())
print(T[(T.held_out != "Pooled") & T.model.isin(["Self-report", "Self-report + passive sensing"])].drop(columns=["brier"]).round(3).to_string())
