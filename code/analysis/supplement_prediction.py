"""RQ4 items the plan requires beyond prediction_agreement.py: agreement by state, calibration, net benefit, exact linear SHAP, repeated CV, sample flow, secondary outcomes.

Usage: python supplement_prediction.py <processed_dir> <out_dir> [table_prefix]
"""
import json
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

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(exist_ok=True, parents=True)
PREFIX = sys.argv[3] if len(sys.argv) > 3 else ""
rd = lambda f: pd.read_csv(D / f"{PREFIX}{f}.csv")
S, E, W, Pa, EMA, P, CAL, SAF, C, SCR = rd("scale_scores"), rd("ecg_lab"), rd("wearable_nightly"), rd("passive_daily"), rd("ema_prompts"), rd("participants"), rd("overnight_device_agreement"), rd("safety_events"), rd("clinician_assessment"), rd("screening")
OUTPUT = {}


def icc21(a, b):
    """Two-way random, absolute agreement, single measure."""
    x = np.c_[a, b]
    n, k = x.shape
    grand, row, col = x.mean(), x.mean(1), x.mean(0)
    msr = k * ((row - grand) ** 2).sum() / (n - 1)
    msc = n * ((col - grand) ** 2).sum() / (k - 1)
    mse = ((x - row[:, None] - col[None, :] + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def ccc(x, y):
    return float(2 * np.cov(x, y, ddof=0)[0, 1] / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def agreement(x, y, label):
    ok = np.isfinite(x) & np.isfinite(y)  # rest and task validity are judged separately, so a session can have one and not the other
    x, y = x[ok], y[ok]
    d = x - y
    # proportional bias: slope of the difference on the mean
    slope = np.polyfit((x + y) / 2, d, 1)[0]
    return dict(comparison=label, n=len(x), r=float(np.corrcoef(x, y)[0, 1]), ccc=ccc(x, y), icc21=icc21(x, y), bias=float(d.mean()), loa_low=float(d.mean() - 1.96 * d.std()), loa_high=float(d.mean() + 1.96 * d.std()), prop_slope=float(slope))


# ---- 3.7 agreement, split by physiological state ----
c = CAL.dropna()
c = c[c.h10_corrected_beats_pct < 5]
wb = W[W.valid_night == 1].groupby(["pid", "wave"]).agg(n=("ln_night_rmssd", "size"), ln_ring=("ln_night_rmssd", "mean"), hr_ring=("night_hr_bpm", "mean")).reset_index()
wb = wb[wb.n >= 5]
m = wb.merge(E, on=["pid", "wave"]).dropna(subset=["ln_ring", "ln_rmssd"])
AG = pd.DataFrame([agreement(np.log(c.ring_night_rmssd_ms.to_numpy()), np.log(c.h10_night_rmssd_ms.to_numpy()), "夜间：戒指 vs H10（同夜，ln）"),
                   agreement(m.ln_ring.to_numpy(), m.ln_rmssd.to_numpy(), "跨状态：戒指 burst 均值 vs 实验室静息（ln）"),
                   agreement(m.ln_ring.to_numpy(), m.ln_rmssd_task.to_numpy(), "跨状态：戒指 burst 均值 vs 实验室任务期（ln）")])
AG.round(3).to_csv(OUT / "s37_agreement.csv", index=False, encoding="utf-8-sig")
# within-person tracking: does the ring follow the lab measure over waves?
mw = m[["pid", "wave", "ln_ring", "ln_rmssd"]].copy()
for k in ("ln_ring", "ln_rmssd"):
    mw[k] = mw[k] - mw.groupby("pid")[k].transform("mean")
OUTPUT["ring_lab_within_person_r"] = float(mw[["ln_ring", "ln_rmssd"]].corr().iloc[0, 1])

# ---- 3.7 prediction with repeated person-grouped CV, calibration, net benefit, SHAP ----
S["state"] = ((S.phq9_total >= 10) | (S.gad7_total >= 10)).astype(int)
base = ["academic_stress", "employment_anxiety", "interpersonal_stress", "phq9_total", "gad7_total", "dass_stress", "psqi_total", "sam_challenge", "stress_mindset", "erq_reappraisal", "ders16_total"]
X = S[["pid", "wave", "state"] + base].merge(P[["pid", "sex", "age", "ctq_total"]], on="pid")
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
SETS = {"自评": SELF, "自评+生理": SELF + PHY, "自评+被动": SELF + PAS, "自评+EMA": SELF + EM, "自评+生理+数字": SELF + PHY + PAS + EM}


def model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def cv_pred(d, cols, repeats=10):
    out = np.zeros((repeats, len(d)))
    pids = d.pid.unique()
    for s in range(repeats):
        g = dict(zip(np.random.default_rng(s).permutation(pids), np.arange(len(pids)) % 5))
        f = d.pid.map(g).to_numpy()
        for k in range(5):
            mdl = model().fit(d.loc[f != k, cols], d.loc[f != k, "y"])
            out[s, f == k] = mdl.predict_proba(d.loc[f == k, cols])[:, 1]
    return out


def calib(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    off = sm.GLM(y, np.ones(len(y)), family=sm.families.Binomial(), offset=lp).fit()
    return float(off.params.iloc[0] if hasattr(off.params, "iloc") else off.params[0]), float(fit.params[1])


def net_benefit(y, p, t):
    tp, fp = ((p >= t) & (y == 1)).sum(), ((p >= t) & (y == 0)).sum()
    return float(tp / len(y) - fp / len(y) * t / (1 - t))


rows, nb = [], []
for pop, d in (("全部转移", R), ("新发（当前未达状态）", R[R.state == 0].reset_index(drop=True))):
    y = d.y.to_numpy()
    preds = {k: cv_pred(d, v) for k, v in SETS.items()}
    pid = d.pid.to_numpy()
    uniq = np.unique(pid)
    idx = {u: np.where(pid == u)[0] for u in uniq}
    mean_self = preds["自评"].mean(0)
    for k, pr in preds.items():
        pm = pr.mean(0)
        a, s_ = calib(y, pm)
        row = dict(population=pop, model=k, n=len(d), events=int(y.sum()), auc=np.mean([roc_auc_score(y, q) for q in pr]), brier=brier_score_loss(y, pm), calib_intercept=a, calib_slope=s_)
        rng = np.random.default_rng(11)
        dl = []
        for _ in range(1000):
            b = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
            dl.append(roc_auc_score(y[b], pm[b]) - roc_auc_score(y[b], mean_self[b]))
        row.update(delta_auc=float(np.mean(dl)), delta_lo=float(np.quantile(dl, .025)), delta_hi=float(np.quantile(dl, .975)))
        rows.append(row)
        for t in (0.1, 0.2, 0.3, 0.4, 0.5):
            nb.append(dict(population=pop, model=k, threshold=t, net_benefit=net_benefit(y, pm, t), treat_all=float(y.mean() - (1 - y.mean()) * t / (1 - t))))
pd.DataFrame(rows).round(4).to_csv(OUT / "s37_prediction.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(nb).round(4).to_csv(OUT / "s37_net_benefit.csv", index=False, encoding="utf-8-sig")

# exact SHAP for a linear model with an independence masker: coefficient times deviation from the mean, on the scaled features
mdl = model().fit(R[SETS["自评+生理+数字"]], R.y)
Z = mdl[:-1].transform(R[SETS["自评+生理+数字"]])
phi = mdl[-1].coef_[0] * (Z - Z.mean(0))
pd.DataFrame({"feature": SETS["自评+生理+数字"], "mean_abs_shap": np.abs(phi).mean(0), "coef": mdl[-1].coef_[0]}).sort_values("mean_abs_shap", ascending=False).round(4).to_csv(OUT / "s37_shap.csv", index=False, encoding="utf-8-sig")
import shap  # noqa: E402
check = shap.LinearExplainer(mdl[-1], shap.maskers.Independent(Z, max_samples=Z.shape[0])).shap_values(Z)
(OUT / "s37_shap_check.json").write_text(json.dumps(dict(max_abs_diff_exact_vs_shap_package=float(np.abs(check - phi).max()), n=int(Z.shape[0]), features=int(Z.shape[1]))), encoding="utf-8")

# ---- 3.1 flow, retention, compliance, missingness ----
Sc = S.groupby("wave").pid.nunique().to_dict()
valid_nights = W[W.valid_night == 1].groupby(["pid", "wave"]).size()
OUTPUT["flow"] = dict(screened=int(len(SCR)), eligible=int(SCR.eligible.sum()), consented=int((SCR.eligible & SCR.consented).sum()), enrolled=int(SCR.enrolled.sum()), by_wave={str(k): int(v) for k, v in Sc.items()})
OUTPUT["tier"] = P.tier.value_counts().to_dict()
OUTPUT["compliance"] = dict(ecg_sessions=int(len(E)), ecg_valid=float(E.ln_rmssd.notna().mean()), eeg_sessions=int(len(rd("eeg_lab"))), eeg_valid=float(rd("eeg_lab").alpha_periodic_log.notna().mean()), ring_valid_nights=float(W.valid_night.mean()), bursts_ge5=float((valid_nights >= 5).mean()), ema_answered=float(EMA.answered.mean()), passive_steps_days=float(Pa.steps.notna().mean()), passive_screen_days=float(Pa.screen_minutes.notna().mean()))
OUTPUT["safety"] = dict(rows=int(len(SAF)), people=int(SAF.pid.nunique()), share_of_sample=float(SAF.pid.nunique() / len(P)), by_source=SAF.trigger_source.value_counts().to_dict(), by_level=SAF.risk_level.value_counts().to_dict(), referred=int(SAF.referred_external.sum()), median_response_h=float(SAF.response_hours.median()), therapy_t3=float(S[S.wave == 3].therapy_use.mean()))

# ---- secondary outcomes: state, passive and active ideation at the next wave, within-person predictors, cluster-robust GEE ----
Lg = S[["pid", "wave", "academic_stress", "employment_anxiety", "interpersonal_stress"]].merge(E[["pid", "wave", "ln_rmssd"]], on=["pid", "wave"], how="left")
Lg["X"] = Lg[["academic_stress", "employment_anxiety", "interpersonal_stress"]].mean(axis=1)
for k in ("X", "ln_rmssd"):
    Lg[k + "_w"] = Lg[k] - Lg.groupby("pid")[k].transform("mean")
    Lg[k + "_b"] = Lg.groupby("pid")[k].transform("mean")
out_y = S[["pid", "wave", "state"]].merge(C[["pid", "wave", "cssrs_passive_ideation", "cssrs_active_ideation"]], on=["pid", "wave"])
out_y["wave"] -= 1
G = Lg.merge(out_y, on=["pid", "wave"]).dropna(subset=["X_w", "ln_rmssd_w"])
sec = []
for y, lab in (("state", "次要结局：亚临床状态"), ("cssrs_passive_ideation", "被动意念"), ("cssrs_active_ideation", "主动意念")):
    xs = sm.add_constant(G[["X_w", "ln_rmssd_w", "X_b", "ln_rmssd_b"]])
    fit = sm.GEE(G[y], xs, groups=G.pid, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    for term in ("X_w", "ln_rmssd_w"):
        sec.append(dict(outcome=lab, predictor=term, events=int(G[y].sum()), n=len(G), or_=float(np.exp(fit.params[term])), lo=float(np.exp(fit.conf_int().loc[term, 0])), hi=float(np.exp(fit.conf_int().loc[term, 1])), p=float(fit.pvalues[term])))
pd.DataFrame(sec).round(4).to_csv(OUT / "s38_secondary_outcomes.csv", index=False, encoding="utf-8-sig")
(OUT / "s31_flow_quality.json").write_text(json.dumps(OUTPUT, indent=1, ensure_ascii=False), encoding="utf-8")
print("supplement prediction done")
