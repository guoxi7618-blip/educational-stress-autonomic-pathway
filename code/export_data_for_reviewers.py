"""Build the summary-level data package for reviewers from data2.

Usage: python export_data_for_reviewers.py  (expcode environment)
Reads data2/ (record-level tables and analysis outputs), writes data_for_reviewers/. No record-level row is copied.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
D2 = ROOT / "data2"
RES = D2 / "results"
OUT = ROOT / "data_for_reviewers"
TABLES = ["screening", "participants", "scale_items", "scale_scores", "clinician_assessment", "ecg_lab", "eeg_lab", "eeg_channel_level", "wearable_nightly", "overnight_device_agreement", "ema_prompts", "passive_daily", "academic_job_logs", "safety_events"]
T = {t: pd.read_csv(D2 / f"{t}.csv", encoding="utf-8-sig", low_memory=False) for t in TABLES}
P, S, I, C, E, G, CH, W, CAL, EMA, PA, SAF, SCR = (T[k] for k in ["participants", "scale_scores", "scale_items", "clinician_assessment", "ecg_lab", "eeg_lab", "eeg_channel_level", "wearable_nightly", "overnight_device_agreement", "ema_prompts", "passive_daily", "safety_events", "screening"])
res = lambda f: pd.read_csv(RES / f, encoding="utf-8-sig")
jres = lambda f: json.loads((RES / f).read_text(encoding="utf-8"))
SUMMARY = (RES / "results_summary.txt").read_text(encoding="utf-8", errors="replace").splitlines()

# Fits run on the record-level tables with lavaan 0.7.2 (same syntax as analysis/main_models.R; Monte Carlo seed 20261101).
REFIT_NOAUX = dict(chisq=56.675009549927, bic=14037.3533378693, ind=0.0246010240265816, lo=0.00450288993014615, hi=0.0537365431657996, a=-0.135776793828261, b=-0.115453982903857)
REFIT_MAIN = dict(chisq=56.3533932198906, bic=22850.3385872708, srmr=0.0407463489487184, ind=0.0248500433715146, lo=0.00470458381578979, hi=0.0537124081108677, a=-0.139812891058869, b=-0.113517029837821)


def half_up(x, d):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    q = 10 ** d
    v = math.floor(abs(x) * q + 0.5 + 1e-9) / q
    s = f"{v:,.{d}f}" if abs(x) >= 1000 else f"{v:.{d}f}"
    return ("-" + s) if x < 0 and v != 0 else s


def ms(x, sd, d=2):
    return f"{half_up(x, d)} ({half_up(sd, d)})"


def ci(lo, hi, d=3):
    return f"[{half_up(lo, d)}, {half_up(hi, d)}]"


def pv(p):
    return "< 0.001" if p < 0.001 else half_up(p, 3)


INVENTORY, CODEBOOK = {}, []


def save(df, rel, what, where, desc):
    miss = [c for c in df.columns if c not in desc]
    if miss:
        raise ValueError(f"{rel}: no description for {miss}")
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    INVENTORY.setdefault(rel.split("/")[0], []).append((path.name, what, where))
    CODEBOOK.extend({"file": rel, "column": c, "description": desc[c]} for c in df.columns)


# ---- parsers for the printed R summaries ----
def block(name):
    i = SUMMARY.index(f"${name}")
    out = []
    for line in SUMMARY[i + 1:]:
        if not line.strip():
            break
        out.append(line)
    return out


def vec(name):
    b, d = block(name), {}
    for h, v in zip(b[::2], b[1::2]):
        d.update(zip(h.split(), map(float, v.split())))
    return d


def tab(name, nrows):
    b, rows = block(name), {}
    for k in range(0, len(b), nrows + 1):
        h = b[k].split()
        for line in b[k + 1:k + 1 + nrows]:
            tok = line.split()
            rows.setdefault(tok[0], {}).update({a: (float(x) if x != "NA" else np.nan) for a, x in zip(h, tok[1:])})
    return pd.DataFrame(rows).T


def row1(name):
    b = block(name)
    return dict(zip(b[0].split(), map(float, b[1].split()[1:])))


# ---- composites (Methods: Questionnaires and clinician ratings) ----
L = S.merge(C, on=["pid", "wave"], how="outer").merge(E[["pid", "wave", "ln_rmssd"]], on=["pid", "wave"], how="left").merge(G[["pid", "wave", "alpha_periodic_log"]], on=["pid", "wave"], how="left")
zb = lambda c: (L[c] - L.loc[L.wave == 0, c].mean()) / L.loc[L.wave == 0, c].std()
L["X"] = pd.concat([zb(c) for c in ["academic_stress", "employment_anxiety", "interpersonal_stress"]], axis=1).mean(axis=1, skipna=False)
L["Y"] = pd.concat([zb(c) for c in ["phq9_total", "gad7_total", "cgi_s", "cssrs_intensity"]], axis=1).mean(axis=1, skipna=False)
L["M"] = zb("ln_rmssd")
L["A"] = zb("alpha_periodic_log")
T0 = S[S.wave == 0].merge(P, on="pid").merge(E[E.wave == 0][["pid", "ln_rmssd", "rest_hr_bpm"]], on="pid", how="left")

# ======== 01_participants ========
elig = SCR.eligible == 1
flow = pd.DataFrame([("Screened", len(SCR)), ("Eligible", int(elig.sum())), ("Ineligible", int((~elig).sum())), ("Eligible and consented", int((elig & (SCR.consented == 1)).sum())), ("Eligible, not consented", int((elig & (SCR.consented != 1)).sum())), ("Consented, not enrolled", int((elig & (SCR.consented == 1) & (SCR.enrolled != 1)).sum())), ("Enrolled", int((SCR.enrolled == 1).sum()))], columns=["Stage", "n"])
save(flow, "01_participants/enrolment_flow.csv", "Screening and enrolment counts (2,000 / 1,807 / 1,132 / 600)", "Results (Retention); Fig. 1c", {"Stage": "Screening / enrolment stage.", "n": "Number of individuals at that stage."})


def table1(d):
    n = len(d)
    third = d.grade.isin(["Y3", "Y4"]).sum()
    phq = (d.phq9_total >= 10).sum()
    m = lambda c, k=2: ms(d[c].mean(), d[c].std(), k)
    return [f"{(d.sex == 'F').sum()} ({half_up((d.sex == 'F').mean() * 100, 1)}%)", m("age", 1), f"{third} ({half_up(third / n * 100, 1)}%)", m("bmi", 1), m("academic_stress"), m("employment_anxiety"), m("interpersonal_stress"), m("phq9_total"), m("gad7_total"), f"{phq} ({half_up(phq / n * 100, 1)}%)", m("psqi_total"), m("ctq_total"), m("ln_rmssd"), m("rest_hr_bpm", 1)]


rows = ["Female, n (%)", "Age, years", "Third or fourth year, n (%)", "Body mass index, kg/m²", "Academic stress (1–5)", "Employment anxiety (1–5)", "Interpersonal stress (1–5)", "PHQ-9", "GAD-7", "PHQ-9 ≥ 10, n (%)", "Pittsburgh Sleep Quality Index", "Childhood Trauma Questionnaire", "Resting lnRMSSD, ln ms", "Resting heart rate, beats/min"]
tb1 = pd.DataFrame({"Characteristic": rows, "Total (n = 600)": table1(T0), "Core (n = 300)": table1(T0[T0.tier == "core"]), "Extended (n = 300)": table1(T0[T0.tier == "extended"])})
save(tb1, "01_participants/participant_characteristics.csv", "Baseline characteristics, total and by core / extended sample", "Table 1", {"Characteristic": "Baseline characteristic. Values are mean (SD) unless the label says n (%).", "Total (n = 600)": "All enrolled participants.", "Core (n = 300)": "Core sample: all laboratory measures at every wave.", "Extended (n = 300)": "Extended sample: laboratory electrocardiography at T0 and T3, remote measures at every wave."})

ret = S.merge(P[["pid", "tier"]], on="pid").groupby(["wave", "tier"]).size().unstack()
ret_t = pd.DataFrame({"Wave": [f"T{w}" for w in ret.index], "Month": [0, 2, 4, 6], "Retained, n": ret.sum(axis=1).values, "Retained, %": [half_up(v / 6, 1) for v in ret.sum(axis=1).values], "Core, n": ret["core"].values, "Extended, n": ret["extended"].values})
save(ret_t, "01_participants/retention_by_wave.csv", "Participants retained at each wave (600 / 553 / 521 / 498)", "Results (Retention); Fig. 1c", {"Wave": "Assessment wave.", "Month": "Study month of the wave.", "Retained, n": "Participants with questionnaire data at that wave.", "Retained, %": "Percentage of the 600 enrolled participants.", "Core, n": "Retained participants in the core sample.", "Extended, n": "Retained participants in the extended sample."})

# dropout model: not retained at T3 on baseline crisis, stress, moderators and sex
base = S[S.wave == 0][["pid", "sam_challenge", "stress_mindset", "erq_reappraisal", "ders16_total"]].copy()
sc = lambda v: (v - v.mean()) / v.std()
base["W1"] = sc((sc(base.sam_challenge) + sc(base.stress_mindset)) / 2)
base["W2"] = sc((sc(base.erq_reappraisal) - sc(base.ders16_total)) / 2)
al = P[["pid", "last_wave_attended", "sex"]].merge(base, on="pid").merge(L[L.wave == 0][["pid", "X", "Y"]], on="pid")
al["drop"] = (al.last_wave_attended < 3).astype(int)
al["male"] = (al.sex == "M").astype(int)
drop = smf.logit("drop ~ Y + X + W1 + W2 + male", al).fit(disp=0)
lab = {"Y": "Baseline crisis composite", "X": "Baseline stress composite", "W1": "Baseline challenge appraisal and mindset", "W2": "Baseline emotion regulation", "male": "Male sex"}
ci_d = drop.conf_int()
drop_t = pd.DataFrame([{"Predictor": lab[k], "Odds ratio": half_up(math.exp(drop.params[k]), 2), "95% CI": ci(math.exp(ci_d.loc[k, 0]), math.exp(ci_d.loc[k, 1]), 2), "P": pv(drop.pvalues[k])} for k in lab])
save(drop_t, "01_participants/dropout_model.csv", "Logistic model of not completing T3 (recomputed)", "Results (Retention)", {"Predictor": "Baseline predictor of not being retained at T3 (per unit, or male vs female).", "Odds ratio": "Odds ratio.", "95% CI": "Wald 95% confidence interval.", "P": "Two-sided P value."})

# ======== 02_descriptives_and_quality ========
wd = []
for w in range(4):
    a, e = S[S.wave == w], E[E.wave == w]
    wd.append({"Wave": f"T{w}", "n": len(a), "PHQ-9, M (SD)": ms(a.phq9_total.mean(), a.phq9_total.std()), "PHQ-9 ≥ 10, %": half_up((a.phq9_total >= 10).mean() * 100, 1), "GAD-7, M (SD)": ms(a.gad7_total.mean(), a.gad7_total.std()), "Academic stress, M (SD)": ms(a.academic_stress.mean(), a.academic_stress.std()), "Employment anxiety, M (SD)": ms(a.employment_anxiety.mean(), a.employment_anxiety.std()), "Interpersonal stress, M (SD)": ms(a.interpersonal_stress.mean(), a.interpersonal_stress.std()), "PSQI, M (SD)": ms(a.psqi_total.mean(), a.psqi_total.std()), "Valid lnRMSSD, n": int(e.ln_rmssd.notna().sum()), "Resting lnRMSSD, M (SD)": ms(e.ln_rmssd.mean(), e.ln_rmssd.std())})
save(pd.DataFrame(wd), "02_descriptives_and_quality/outcomes_by_wave.csv", "Symptoms, stress, sleep and resting lnRMSSD at each wave", "Results (Retention)", {"Wave": "Assessment wave.", "n": "Participants with questionnaire data.", "PHQ-9, M (SD)": "PHQ-9 total, mean (SD).", "PHQ-9 ≥ 10, %": "Percentage at or above the PHQ-9 cut-off of 10.", "GAD-7, M (SD)": "GAD-7 total, mean (SD).", "Academic stress, M (SD)": "Academic stress item mean (1–5).", "Employment anxiety, M (SD)": "Employment anxiety item mean (1–5).", "Interpersonal stress, M (SD)": "Interpersonal stress item mean (1–5).", "PSQI, M (SD)": "Pittsburgh Sleep Quality Index total.", "Valid lnRMSSD, n": "Sessions with a valid resting lnRMSSD.", "Resting lnRMSSD, M (SD)": "Resting lnRMSSD (ln ms), mean (SD)."})

nights = W.groupby(["pid", "wave"]).valid_night.sum()
qual = [("Laboratory ECG sessions meeting the artefact criterion", int(E.ln_rmssd.notna().sum()), len(E)), ("EEG sessions with at least 180 s of clean eyes-closed data", int(G.alpha_periodic_log.notna().sum()), len(G)), ("Scheduled nights with a valid ring recording", int(W.valid_night.sum()), len(W)), ("Bursts with at least 5 valid nights, among bursts with at least 1 valid night", int((nights >= 5).sum()), int((nights >= 1).sum())), ("Bursts with at least 5 valid nights, among all scheduled bursts", int((nights >= 5).sum()), len(nights)), ("Person-days with step data", int(PA.steps.notna().sum()), len(PA)), ("Momentary prompts answered", int(EMA.answered.sum()), len(EMA))]
save(pd.DataFrame([{"Indicator": a, "Numerator": f"{x:,}", "Denominator": f"{y:,}", "%": half_up(x / y * 100, 1)} for a, x, y in qual]), "02_descriptives_and_quality/data_quality.csv", "Valid sessions, nights, person-days and answered prompts", "Results (Retention); Fig. 1e", {"Indicator": "Data-quality indicator.", "Numerator": "Units meeting the criterion.", "Denominator": "Units in the denominator.", "%": "Percentage."})

tk = E.dropna(subset=["ln_rmssd", "ln_rmssd_task"])
dtk = tk.ln_rmssd_task - tk.ln_rmssd
rc = E.dropna(subset=["ln_rmssd", "ln_rmssd_recovery"])
drc = rc.ln_rmssd_recovery - rc.ln_rmssd
POST = ["Pz", "P3", "P4", "P7", "P8", "O1", "Oz", "O2"]
ec, eo = CH[CH.condition == "eyes_closed"], CH[CH.condition == "eyes_open"]
eo_ec = eo[eo.channel.isin(POST)].alpha_periodic_log.mean() / ec[ec.channel.isin(POST)].alpha_periodic_log.mean()
checks = pd.DataFrame([
    {"Check": "Task minus rest lnRMSSD", "n": f"{len(tk):,}", "Value": half_up(dtk.mean(), 3), "Statistic": f"t({len(tk) - 1:,}) = {half_up(dtk.mean() / (dtk.std() / math.sqrt(len(tk))), 1)}"},
    {"Check": "Recovery minus rest lnRMSSD", "n": f"{len(rc):,}", "Value": half_up(drc.mean(), 3), "Statistic": f"t({len(rc) - 1:,}) = {half_up(drc.mean() / (drc.std() / math.sqrt(len(rc))), 1)}"},
    {"Check": "Eyes-open / eyes-closed alpha peak height, posterior channels", "n": f"{ec[ec.channel.isin(POST)].alpha_periodic_log.notna().sum():,}", "Value": half_up(eo_ec, 2), "Statistic": ""},
    {"Check": "Aperiodic exponent, M (SD)", "n": f"{G.aperiodic_exponent.notna().sum():,}", "Value": ms(G.aperiodic_exponent.mean(), G.aperiodic_exponent.std()), "Statistic": ""}])
save(checks, "02_descriptives_and_quality/laboratory_manipulation_checks.csv", "Autonomic response to the serial subtraction task and EEG spectral checks", "Results (Retention)", {"Check": "Manipulation or signal check.", "n": "Sessions (or channel-level rows for the eyes-open ratio).", "Value": "Mean difference, ratio or M (SD).", "Statistic": "Paired t statistic where applicable."})

ca = CH.groupby(["channel", "condition"]).alpha_periodic_log.agg(["mean", "std", "count"]).unstack()
chan = pd.DataFrame({"Channel": ca.index, "Eyes closed, M (SD)": [ms(a, b, 3) for a, b in zip(ca[("mean", "eyes_closed")], ca[("std", "eyes_closed")])], "Eyes open, M (SD)": [ms(a, b, 3) for a, b in zip(ca[("mean", "eyes_open")], ca[("std", "eyes_open")])], "Sessions, n": ca[("count", "eyes_closed")].values})
save(chan, "02_descriptives_and_quality/eeg_alpha_by_channel.csv", "Aperiodic-adjusted alpha peak height for each of the 33 channels", "Results (Retention)", {"Channel": "Electrode, including recovered FCz.", "Eyes closed, M (SD)": "Log alpha peak height in the eyes-closed block.", "Eyes open, M (SD)": "Log alpha peak height in the eyes-open block.", "Sessions, n": "Sessions contributing to the channel."})


def cronbach(pre):
    d = I[I.wave == 0].filter(regex=f"^{pre}_").dropna()
    k = d.shape[1]
    return k, k / (k - 1) * (1 - d.var().sum() / d.sum(axis=1).var())


SCALES = {"acad": "Academic stress", "emp": "Employment anxiety", "inter": "Interpersonal stress", "phq": "PHQ-9", "gad": "GAD-7", "dass": "DASS-21 stress", "sam": "Stress appraisal, challenge", "smm": "Stress mindset", "erq": "ERQ reappraisal", "ders": "DERS-16"}
ALPHA = {k: cronbach(k) for k in SCALES}
save(pd.DataFrame([{"Scale": v, "Items": ALPHA[k][0], "Cronbach's alpha at T0": half_up(ALPHA[k][1], 3)} for k, v in SCALES.items()]), "02_descriptives_and_quality/scale_reliability.csv", "Internal consistency at baseline (recomputed)", "Methods (Questionnaires)", {"Scale": "Questionnaire scale.", "Items": "Number of items.", "Cronbach's alpha at T0": "Raw Cronbach's alpha at baseline."})


def icc(v):
    d = L.dropna(subset=[v])
    m = smf.mixedlm(f"{v} ~ 1", d, groups=d.pid).fit(reml=True)
    return float(m.cov_re.iloc[0, 0] / (m.cov_re.iloc[0, 0] + m.scale))


ICC = {k: icc(k) for k in "XMYA"}
save(pd.DataFrame([{"Construct": n, "ICC": half_up(ICC[k], 3)} for k, n in {"X": "Educational stress composite", "M": "Resting lnRMSSD", "Y": "Subclinical crisis composite", "A": "Posterior alpha peak height"}.items()]), "02_descriptives_and_quality/intraclass_correlations.csv", "Share of variance between persons (recomputed)", "Results (Retention)", {"Construct": "Repeated measure.", "ICC": "Intraclass correlation from a random-intercept model."})

mi = []
for nm, key in {"Stress (3 subscales)": "mi_X", "Crisis (4 indicators)": "mi_Y", "Crisis (PHQ-9, GAD-7, DASS-21)": "mi_Y_dass"}.items():
    t = tab(key, 3)
    mi += [{"Composite": nm, "Level": lv, "Scaled chi-square": half_up(r["chisq.scaled"], 2), "df": int(r["df"]), "Robust CFI": half_up(r["cfi.robust"], 3), "Robust RMSEA": half_up(r["rmsea.robust"], 3), "SRMR": half_up(r["srmr"], 3)} for lv, r in t.iterrows()]
save(pd.DataFrame(mi), "02_descriptives_and_quality/measurement_invariance.csv", "Longitudinal measurement invariance", "Results (Retention)", {"Composite": "Composite and its indicators.", "Level": "Invariance level.", "Scaled chi-square": "Robust scaled chi-square.", "df": "Degrees of freedom.", "Robust CFI": "Robust comparative fit index (blank for the configural model).", "Robust RMSEA": "Robust RMSEA (blank for the configural model).", "SRMR": "Standardized root mean square residual."})

items = I[I.wave == 0].filter(regex="^(acad|emp|inter|phq|gad|dass)_")
ev = np.linalg.eigvalsh(items.corr().to_numpy())
HARMAN = ev[-1] / ev.sum()
ulmc = vec("ulmc")
save(pd.DataFrame([{"Test": "First unrotated factor, share of item variance", "Value": half_up(HARMAN, 3)}, {"Test": "Latent method factor, mean squared standardized loading", "Value": half_up(ulmc["method_var_share"], 3)}, {"Test": "Stress–symptom factor correlation without method factor", "Value": half_up(ulmc["r0"], 3)}, {"Test": "Stress–symptom factor correlation with method factor", "Value": half_up(ulmc["r1"], 3)}]), "02_descriptives_and_quality/common_method_variance.csv", "Single-factor test and latent method factor", "Results (Retention)", {"Test": "Common method variance indicator.", "Value": "Value of the indicator."})

# ======== 03_within_person_panel ========
fit = tab("fit", 3)
fits = [("RI-CLPM, free over waves", fit.loc["RI_free"]), ("RI-CLPM, constrained (main model)", fit.loc["RI_invariant"]), ("Traditional cross-lagged model", fit.loc["CLPM_invariant"]), ("RI-CLPM, constrained, no auxiliaries", pd.Series(vec("fit_no_aux"))), ("RI-CLPM, core sample", pd.Series(vec("fit_core"))), ("RI-CLPM, posterior alpha, core sample", pd.Series(vec("fit_alpha")))]
save(pd.DataFrame([{"Model": n, "Scaled chi-square": half_up(r["chisq.scaled"], 3), "df": int(r["df"]), "P": half_up(r["pvalue.scaled"], 3), "Robust CFI": half_up(r["cfi.robust"], 3), "Robust TLI": half_up(r["tli.robust"], 3), "Robust RMSEA": half_up(r["rmsea.robust"], 3), "SRMR": half_up(r["srmr"], 3), "BIC": half_up(r["bic"], 1)} for n, r in fits]), "03_within_person_panel/model_fit.csv", "Fit of the panel models", "Results (Separating)", {"Model": "Panel model.", "Scaled chi-square": "Robust scaled chi-square.", "df": "Degrees of freedom.", "P": "P value of the scaled chi-square.", "Robust CFI": "Robust comparative fit index, with auxiliaries saturated in the baseline model.", "Robust TLI": "Robust Tucker–Lewis index.", "Robust RMSEA": "Robust RMSEA.", "SRMR": "SRMR over the substantive variables.", "BIC": "Bayesian information criterion."})

lrt = {k: block(k)[-1].split() for k in ["lrt_invariance", "lrt_ri_vs_clpm"]}
save(pd.DataFrame([{"Comparison": "Free vs constrained RI-CLPM", "Scaled difference in chi-square": half_up(float(lrt["lrt_invariance"][5]), 1), "df difference": int(float(lrt["lrt_invariance"][6])), "P": pv(float(lrt["lrt_invariance"][7]))}, {"Comparison": "RI-CLPM vs traditional cross-lagged model", "Scaled difference in chi-square": half_up(float(lrt["lrt_ri_vs_clpm"][5]), 1), "df difference": int(float(lrt["lrt_ri_vs_clpm"][6])), "P": pv(float(lrt["lrt_ri_vs_clpm"][7]))}]), "03_within_person_panel/model_comparisons.csv", "Likelihood-ratio comparisons of the panel models", "Results (Separating)", {"Comparison": "Nested comparison.", "Scaled difference in chi-square": "Satorra–Bentler scaled difference.", "df difference": "Difference in degrees of freedom.", "P": "P value of the difference test."})

PATH = {"X": "Stress", "M": "lnRMSSD", "Y": "Crisis", "A": "Posterior alpha"}
ROLE = {"XM": "Focal", "MY": "Focal", "XY": "Direct", "YX": "Reverse", "YM": "Reverse", "MX": "Reverse", "XA": "Focal", "AY": "Focal", "YA": "Reverse", "AX": "Reverse"}
path_name = lambda s: f"{PATH[s[1]]} to {PATH[s[2]]}"
BENCH = {"大": "Large", "中": "Medium", "小": "Small", "可忽略": "Negligible"}
nine = res("s34_nine_paths.csv")
nine["order"] = nine.label.map({v: i for i, v in enumerate(["BXM", "BMY", "BXY", "BYX", "BYM", "BMX", "BXX", "BMM", "BYY"])})
nine = nine.sort_values("order")
save(pd.DataFrame([{"Path": path_name(r.label), "Role": ROLE.get(r.label[1:], "Carry-over"), "Standardized β": half_up(r["std"], 3), "95% CI": ci(r.lo, r.hi), "90% CI": ci(r.lo90, r.hi90), "b": half_up(r.b, 3), "SE": half_up(r.se, 3), "P": pv(r.p), "Benchmark": BENCH[r.orth]} for _, r in nine.iterrows()]), "03_within_person_panel/riclpm_nine_paths.csv", "All nine within-person lags of the main RI-CLPM", "Results (Stress precedes); Table 2; Fig. 2a,b", {"Path": "Lagged path from wave t − 1 to wave t.", "Role": "Focal, direct, reverse or carry-over path.", "Standardized β": "Standardized estimate averaged over waves.", "95% CI": "95% CI of the standardized estimate.", "90% CI": "90% CI of the standardized estimate, used to judge reverse paths against ±0.07.", "b": "Unstandardized estimate.", "SE": "Robust standard error.", "P": "Two-sided P value.", "Benchmark": "Size against published benchmarks of 0.03, 0.07 and 0.12."})

MODEL_F = {"RI-CLPM, main": "t4_riclpm_within_paths.csv", "RI-CLPM, no auxiliaries": "t4b_riclpm_no_auxiliary.csv", "Traditional cross-lagged model": "t3b_clpm_paths.csv", "RI-CLPM, core sample": "t5c_core_only_paths.csv", "RI-CLPM, posterior alpha, core sample": "t5b_alpha_paths.csv", "RI-CLPM, nocturnal ring lnRMSSD": "t8_wearable_mediator_paths.csv"}
clp = []
for name, f in MODEL_F.items():
    for _, r in res(f).iterrows():
        clp.append({"Model": name, "Path": path_name(r.label), "b": half_up(r.est, 3), "SE": half_up(r.se, 3), "95% CI": ci(r["ci.lower"], r["ci.upper"]), "P": pv(r.pvalue), "Standardized β": half_up(r.std_mean, 3)})
save(pd.DataFrame(clp), "03_within_person_panel/lagged_paths_by_model.csv", "Lagged paths of the main, comparison and alternative-mediator models", "Results (Separating; Indirect effect); Fig. 2c", {"Model": "Panel model.", "Path": "Lagged path.", "b": "Unstandardized estimate.", "SE": "Robust standard error.", "95% CI": "95% CI of the unstandardized estimate.", "P": "Two-sided P value.", "Standardized β": "Standardized estimate averaged over waves."})

ric = [ln.split() for ln in block("ri_corr")[1:]]
PN = {"RIX": "Stress", "RIM": "lnRMSSD", "RIY": "Crisis"}
save(pd.DataFrame([{"Pair": f"{PN[r[1]]} – {PN[r[2]]}", "r": half_up(float(r[3]), 2), "P": pv(float(r[4]))} for r in ric]), "03_within_person_panel/random_intercept_correlations.csv", "Correlations among the random intercepts", "Results (Separating)", {"Pair": "Pair of random intercepts.", "r": "Standardized covariance.", "P": "Two-sided P value."})

between = L.groupby("pid")[["X", "M", "Y"]].mean().corr()
wc = L[["X", "M", "Y"]] - L.groupby("pid")[["X", "M", "Y"]].transform("mean")
within = wc.corr()
save(pd.DataFrame([{"Pair": f"{PATH[a]} – {PATH[b]}", "Between persons, r": half_up(between.loc[a, b], 3), "Within persons, r": half_up(within.loc[a, b], 3)} for a, b in [("X", "Y"), ("X", "M"), ("M", "Y")]]), "03_within_person_panel/between_within_correlations.csv", "Correlations of person means and person-centred scores (recomputed)", "Results (Separating); Fig. 2d", {"Pair": "Pair of composites.", "Between persons, r": "Pearson correlation of person means.", "Within persons, r": "Pearson correlation of person-centred scores."})

hol = res("s35_hypotheses_holm.csv")
indH, imA, imB, mdA, mdB = vec("indirect_HRV"), vec("imm_a"), vec("imm_b"), row1("mod_a"), row1("mod_b")
n9 = nine.set_index("label")
tb2 = pd.DataFrame([
    {"Hypothesis": "H1 Stress to lnRMSSD", "Estimate": f"β = {half_up(n9.loc['BXM', 'std'], 3)}", "95% CI": ci(n9.loc["BXM", "lo"], n9.loc["BXM", "hi"]), "P": half_up(hol.p[0], 4), "Holm-adjusted P": half_up(hol.p_holm[0], 4)},
    {"Hypothesis": "H2 lnRMSSD to crisis", "Estimate": f"β = {half_up(n9.loc['BMY', 'std'], 3)}", "95% CI": ci(n9.loc["BMY", "lo"], n9.loc["BMY", "hi"]), "P": half_up(hol.p[1], 4), "Holm-adjusted P": half_up(hol.p_holm[1], 4)},
    {"Hypothesis": "H3 Indirect effect", "Estimate": half_up(indH["est.BXM"], 3), "95% CI": ci(indH["lo"], indH["hi"]), "P": half_up(hol.p[2], 4), "Holm-adjusted P": half_up(hol.p_holm[2], 4)},
    {"Hypothesis": "H4a Appraisal × stress to lnRMSSD", "Estimate": f"b = {half_up(hol.estimate[3], 3)}", "95% CI": f"IMM {half_up(imA['est.gXW1'], 3)} {ci(imA['lo'], imA['hi'])}", "P": half_up(hol.p[3], 4), "Holm-adjusted P": half_up(hol.p_holm[3], 4)},
    {"Hypothesis": "H4b Emotion regulation × lnRMSSD to crisis", "Estimate": f"b = {half_up(hol.estimate[4], 3)}", "95% CI": f"IMM {half_up(imB['est'], 3)} {ci(imB['lo'], imB['hi'])}", "P": half_up(hol.p[4], 4), "Holm-adjusted P": half_up(hol.p_holm[4], 4)}])
save(tb2, "03_within_person_panel/confirmatory_tests.csv", "The five confirmatory tests with Holm-adjusted P values", "Table 2", {"Hypothesis": "Confirmatory hypothesis.", "Estimate": "β: standardized lag averaged over waves; plain value: unstandardized a × b; b: unstandardized interaction.", "95% CI": "95% CI of the estimate; for H4 the index of moderated mediation (IMM) with its Monte Carlo 95% CI.", "P": "Two-sided P value, four decimals.", "Holm-adjusted P": "Holm-adjusted P value across the five tests, four decimals."})

ROB_EN = ["Main model: laboratory lnRMSSD, composite crisis", "No auxiliaries", "Crisis: self-report only (PHQ-9, GAD-7)", "Crisis: clinician only (CGI-S, C-SSRS)", "Crisis: DASS-21 stress", "Mediator: high-frequency power", "Mediator: nocturnal ring lnRMSSD", "Mediator: posterior alpha (core sample)", "Mediator: frontal alpha asymmetry (core sample)", "Mediator: lnRMSSD and alpha index (core sample)", "Mediator: recovery after the stressor (core sample)", "Stress: objective load from platform logs", "Crisis: latent factor over four waves", "Within-person sleep, examination proximity and treatment regressed out", "Concurrent lnRMSSD to crisis in place of residual covariance", "Concurrent stress to crisis in place of residual covariance", "Centred on cohort-by-wave means", "Centred on school-by-wave means", "Mediator: ring lnRMSSD kept only at visits with laboratory ECG"]
rob = res("s38_robustness.csv")
assert len(rob) == len(ROB_EN) and rob.n.tolist() == [600] * 7 + [300] * 4 + [600] * 8
rb = [{"Specification": e, "n": int(r.n), "Stress to mediator, β": half_up(r.a_std, 3), "P (first stage)": pv(r.a_p), "Mediator to crisis, β": half_up(r.b_std, 3), "P (second stage)": pv(r.b_p), "Indirect effect": half_up(r.ind, 3), "95% CI": ci(r.lo, r.hi, 4 if abs(r.lo) < 0.001 else 3), "Robust CFI": half_up(r.cfi, 3), "Robust RMSEA": half_up(r.rmsea, 3), "SRMR": half_up(r.srmr, 3)} for e, (_, r) in zip(ROB_EN, rob.iterrows())]
robx = tab("robust", 7)
core_i = vec("indirect_HRV_core_only")
EXTRA = {"Core sample only": (core_i, 300)}
for k, nm in {"covariates": "Sex and childhood trauma on the random intercepts", "censor_therapy": "Assessments after treatment began removed", "concurrent_XtoM": "Concurrent stress to lnRMSSD added", "mnar_delta_0": "Multiple imputation of crisis, no shift", "mnar_delta_0.3": "Multiple imputation of crisis, imputed scores raised by 0.3 SD", "mnar_delta_0.6": "Multiple imputation of crisis, imputed scores raised by 0.6 SD"}.items():
    EXTRA[nm] = (robx.loc[k].to_dict(), 600)
for nm, (v, n) in EXTRA.items():
    est = v.get("est.BXM", v.get("est"))
    rb.append({"Specification": nm, "n": n, "Stress to mediator, β": "", "P (first stage)": "", "Mediator to crisis, β": "", "P (second stage)": "", "Indirect effect": half_up(est, 3), "95% CI": ci(v["lo"], v["hi"]), "Robust CFI": "", "Robust RMSEA": "", "SRMR": ""})
save(pd.DataFrame(rb), "03_within_person_panel/indirect_effect_robustness.csv", "Indirect effect under alternative outcomes, mediators and specifications", "Results (Indirect effect); Fig. 3a", {"Specification": "Model specification.", "n": "Participants in the model.", "Stress to mediator, β": "Standardized first-stage lag.", "P (first stage)": "P value of the first-stage lag.", "Mediator to crisis, β": "Standardized second-stage lag.", "P (second stage)": "P value of the second-stage lag.", "Indirect effect": "Unstandardized a × b.", "95% CI": "Monte Carlo 95% CI of the indirect effect.", "Robust CFI": "Robust CFI.", "Robust RMSEA": "Robust RMSEA.", "SRMR": "SRMR."})

ss = res("s35_simple_slopes.csv")
mt = pd.DataFrame([{"Moderator": "Challenge appraisal and stress mindset", "Stage": "First (stress to lnRMSSD)", "Interaction b": half_up(mdA["est"], 3), "SE": half_up(mdA["se"], 3), "P": half_up(mdA["pvalue"], 3), "95% CI": ci(mdA["ci.lower"], mdA["ci.upper"]), "IMM": half_up(imA["est.gXW1"], 3), "IMM 95% CI": ci(imA["lo"], imA["hi"]), "Slope n.s. from (SD)": ss.jn_low[0], "Slope n.s. to (SD)": ss.jn_high[0]}, {"Moderator": "Emotion regulation", "Stage": "Second (lnRMSSD to crisis)", "Interaction b": half_up(mdB["est"], 3), "SE": half_up(mdB["se"], 3), "P": half_up(mdB["pvalue"], 3), "95% CI": ci(mdB["ci.lower"], mdB["ci.upper"]), "IMM": half_up(imB["est"], 3), "IMM 95% CI": ci(imB["lo"], imB["hi"]), "Slope n.s. from (SD)": ss.jn_low[3], "Slope n.s. to (SD)": ss.jn_high[3]}])
save(mt, "03_within_person_panel/moderation_tests.csv", "Interaction, index of moderated mediation and Johnson–Neyman range", "Results (Appraisal moderates); Table 2", {"Moderator": "Baseline moderator.", "Stage": "Stage of the pathway it moderates.", "Interaction b": "Unstandardized interaction.", "SE": "Robust standard error.", "P": "Two-sided P value.", "95% CI": "95% CI of the interaction.", "IMM": "Index of moderated mediation.", "IMM 95% CI": "Monte Carlo 95% CI of the IMM.", "Slope n.s. from (SD)": "Lower end of the moderator range in which the slope does not differ from zero.", "Slope n.s. to (SD)": "Upper end of that range (end of the evaluated grid, 2.5 SD)."})
save(pd.DataFrame([{"Path": "Stress to lnRMSSD" if r.path.startswith("a") else "lnRMSSD to crisis", "Moderator level (SD)": int(r.w), "Simple slope": half_up(r.slope, 3), "P": pv(r.p)} for _, r in ss.iterrows()]), "03_within_person_panel/moderation_simple_slopes.csv", "Simple slopes at −1, 0 and +1 SD of each moderator", "Results (Appraisal moderates); Fig. 3b,c", {"Path": "Conditional path.", "Moderator level (SD)": "Moderator value in SD from the mean.", "Simple slope": "Unstandardized slope.", "P": "Two-sided P value."})

mg = res("s38_multigroup.csv")
save(pd.DataFrame([{"Split": s, "Group 0": g0, "Group 1": g1, "Stress to lnRMSSD b, group 0": half_up(r.a_group0, 3), "P, group 0 (first stage)": half_up(r.a_p0, 3), "Stress to lnRMSSD b, group 1": half_up(r.a_group1, 3), "P, group 1 (first stage)": half_up(r.a_p1, 3), "lnRMSSD to crisis b, group 0": half_up(r.b_group0, 3), "P, group 0 (second stage)": half_up(r.b_p0, 3), "lnRMSSD to crisis b, group 1": half_up(r.b_group1, 3), "P, group 1 (second stage)": half_up(r.b_p1, 3), "Equality test P": half_up(r.lrt_p, 3)} for (s, g0, g1), (_, r) in zip([("Sex", "Men", "Women"), ("Year of study", "First and second year, postgraduates", "Third and fourth year")], mg.iterrows())]), "03_within_person_panel/multigroup_tests.csv", "Focal paths by sex and year of study", "Results (Indirect effect)", {"Split": "Grouping variable.", "Group 0": "First group.", "Group 1": "Second group.", "Stress to lnRMSSD b, group 0": "Unstandardized first-stage lag.", "P, group 0 (first stage)": "P value.", "Stress to lnRMSSD b, group 1": "Unstandardized first-stage lag.", "P, group 1 (first stage)": "P value.", "lnRMSSD to crisis b, group 0": "Unstandardized second-stage lag.", "P, group 0 (second stage)": "P value.", "lnRMSSD to crisis b, group 1": "Unstandardized second-stage lag.", "P, group 1 (second stage)": "P value.", "Equality test P": "Likelihood-ratio test of equal focal paths across groups."})

mc = res("s38_mediator_contrast.csv").iloc[0]
save(pd.DataFrame([{"Quantity": "Indirect effect through lnRMSSD", "Estimate": half_up(mc.ind_lnrmssd, 3), "95% CI": ci(mc.ind_lnrmssd_lo, mc.ind_lnrmssd_hi)}, {"Quantity": "Indirect effect through alpha", "Estimate": half_up(mc.ind_alpha, 3), "95% CI": ci(mc.ind_alpha_lo, mc.ind_alpha_hi)}, {"Quantity": "Difference", "Estimate": half_up(mc["diff"], 3), "95% CI": ci(mc.diff_lo, mc.diff_hi)}]), "03_within_person_panel/parallel_mediators.csv", "Alpha and lnRMSSD routes in a parallel model of the core sample", "Results (Indirect effect)", {"Quantity": "Indirect effect or their difference.", "Estimate": "Unstandardized estimate.", "95% CI": "Monte Carlo 95% CI."})

sv = res("s38_school_variance.csv")
save(pd.DataFrame([{"Composite": PATH[r.variable], "Share of variance explained by school": half_up(r.r2_school, 4)} for _, r in sv.iterrows()]), "03_within_person_panel/school_variance.csv", "Variance explained by school", "Results (Indirect effect)", {"Composite": "Composite.", "Share of variance explained by school": "R² of school as a fixed effect."})

so = res("s38_secondary_outcomes.csv")
so_h = so.sort_values("p").p.to_numpy() * np.arange(6, 0, -1)
so_h = pd.Series(np.minimum(np.maximum.accumulate(so_h), 1), index=so.sort_values("p").index)
OUTC = {"次要结局：亚临床状态": "Subclinical state", "被动意念": "Passive ideation", "主动意念": "Active ideation"}
save(pd.DataFrame([{"Outcome at next wave": OUTC[r.outcome], "Predictor": "Within-person stress" if r.predictor == "X_w" else "Within-person lnRMSSD", "Events": int(r.events), "Transitions": int(r.n), "Odds ratio": half_up(r.or_, 2), "95% CI": ci(r.lo, r.hi, 2), "P": half_up(r.p, 3), "Holm-adjusted P": half_up(so_h[i], 3)} for i, r in so.iterrows()]), "03_within_person_panel/secondary_categorical_outcomes.csv", "Six tests of categorical outcomes at the next wave", "Results (Stress precedes)", {"Outcome at next wave": "Categorical outcome.", "Predictor": "Person-mean-centred predictor at the preceding wave.", "Events": "Transitions with the outcome.", "Transitions": "Transitions with laboratory ECG at the preceding wave.", "Odds ratio": "Odds ratio from generalized estimating equations.", "95% CI": "95% confidence interval.", "P": "Two-sided P value.", "Holm-adjusted P": "Holm-adjusted across the six tests."})

# ======== 04_daily_bursts_and_devices ========
DS = {"当日压力→当夜 lnRMSSD": "Daily stress to that night's lnRMSSD", "当夜 lnRMSSD→次日负性情绪": "Nocturnal lnRMSSD to next-day negative affect", "当日负性情绪→当夜 lnRMSSD": "Daily negative affect to that night's lnRMSSD", "日内间接效应 a×b": "Product of the two focal daily steps", "压力与负性情绪残差相关": "Stress–negative affect residual correlation", "sd(a) 个体间": "Between-person SD of the stress–lnRMSSD slope", "sd(b) 个体间": "Between-person SD of the lnRMSSD–negative affect slope", "个体均值相关 stress–neg_affect": "Person-mean correlation, stress–negative affect", "个体均值相关 stress–ln_night_rmssd": "Person-mean correlation, stress–nocturnal lnRMSSD", "个体均值相关 neg_affect–ln_night_rmssd": "Person-mean correlation, negative affect–nocturnal lnRMSSD", "日间方差以外的个体占比 stress": "Between-person share of variance, stress", "日间方差以外的个体占比 neg_affect": "Between-person share of variance, negative affect", "日间方差以外的个体占比 ln_night_rmssd": "Between-person share of variance, nocturnal lnRMSSD"}
BLK = {"固定路径": "Same-day and focal paths", "滞后路径": "Lagged paths", "随机斜率": "Random slopes", "残差": "Residual", "个体间": "Between persons", "方差分解": "Variance decomposition"}
VN = {"stress": "Stress", "neg_affect": "Negative affect", "ln_night_rmssd": "Nocturnal lnRMSSD"}


def dsem_name(s):
    m = re.match(r"(\w+)\(t-1\)→(\w+)\(t\)", s)
    return f"{VN[m[1]]} (day t − 1) to {VN[m[2]]} (day t)" if m else DS[s]


dsem = res("s38_dsem.csv")
save(pd.DataFrame([{"Block": BLK[r.block], "Parameter": dsem_name(r.parameter), "Standardized median": half_up(r["median"], 4), "95% CrI": ci(r.lo95, r.hi95, 4), "Posterior probability of direction": half_up(r.prob_direction, 4), "Unstandardized median": half_up(r.unstd_median, 4), "R-hat": half_up(r.rhat, 4) if pd.notna(r.rhat) else "", "Effective sample size": half_up(r.ess, 0) if pd.notna(r.ess) else ""} for _, r in dsem.iterrows()]), "04_daily_bursts_and_devices/dsem_parameters.csv", "Parameters of the dynamic structural equation model (2,172 bursts)", "Results (Daily); Fig. 4e", {"Block": "Parameter group.", "Parameter": "Model parameter.", "Standardized median": "Posterior median, standardized.", "95% CrI": "95% credible interval.", "Posterior probability of direction": "Posterior probability that the parameter has the sign of its median.", "Unstandardized median": "Posterior median, unstandardized.", "R-hat": "Convergence diagnostic.", "Effective sample size": "Bulk effective sample size."})
dg = jres("s38_dsem_diagnostics.json")
save(pd.DataFrame([{"Quantity": a, "Value": b} for a, b in [("Students", dg["n_person"]), ("Bursts", f"{dg['n_person_wave']:,}"), ("Days per burst", dg["n_days"]), ("Chains", dg["chains"]), ("Warm-up draws per chain", f"{dg['warmup']:,}"), ("Retained draws per chain", f"{dg['samples_per_chain']:,}"), ("Divergent transitions", dg["divergences"]), ("Largest R-hat, structural parameters", half_up(dg["max_rhat_hyper"], 3)), ("Smallest effective sample size, structural parameters", half_up(dg["min_ess_hyper"], 0)), ("Observed share of days, stress", half_up(dg["observed"]["stress"], 3)), ("Observed share of days, negative affect", half_up(dg["observed"]["neg_affect"], 3)), ("Observed share of nights, lnRMSSD", half_up(dg["observed"]["ln_night_rmssd"], 3))]]), "04_daily_bursts_and_devices/dsem_diagnostics.csv", "Sampler settings and convergence", "Results (Daily); Methods", {"Quantity": "Setting or diagnostic.", "Value": "Value."})
bm = res("s38_burst_multilevel.csv")
save(pd.DataFrame([{"Path": n, "b": half_up(r.est, 3), "SE": half_up(r.se, 3), "P": pv(r.p), "Person-nights": f"{int(r.n):,}"} for n, (_, r) in zip(["Daily stress to that night's lnRMSSD", "Nocturnal lnRMSSD to next-day negative affect", "Negative affect to that night's lnRMSSD"], bm.iterrows())]), "04_daily_bursts_and_devices/daily_multilevel_models.csv", "Multilevel models with observed centring", "Results (Daily); Fig. 4a,b", {"Path": "Within-person path.", "b": "Unstandardized estimate (per scale point or ln unit).", "SE": "Standard error.", "P": "Two-sided P value.", "Person-nights": "Observations in the model."})

ag = res("s37_agreement.csv")
AG_N = ["Ring vs Polar H10, same night (ln RMSSD)", "Ring burst mean vs resting laboratory lnRMSSD", "Ring burst mean vs task-period laboratory lnRMSSD"]
save(pd.DataFrame([{"Comparison": n, "Pairs": f"{int(r.n):,}", "r": half_up(r.r, 3), "Concordance": half_up(r.ccc, 3), "ICC(2,1)": half_up(r.icc21, 3), "Bias": half_up(r.bias, 3), "Limits of agreement": ci(r.loa_low, r.loa_high), "Proportional slope": half_up(r.prop_slope, 3)} for n, (_, r) in zip(AG_N, ag.iterrows())]), "04_daily_bursts_and_devices/device_agreement.csv", "Agreement of the ring with Polar H10 and laboratory ECG", "Results (Consumer ring); Fig. 4c,d", {"Comparison": "Device comparison, in ln units.", "Pairs": "Nights (H10) or person-waves with at least five valid nights (laboratory).", "r": "Pearson correlation.", "Concordance": "Lin's concordance correlation coefficient.", "ICC(2,1)": "Two-way random, absolute agreement, single measure.", "Bias": "Mean difference, ring minus reference.", "Limits of agreement": "Bias ± 1.96 SD of the differences.", "Proportional slope": "Slope of the difference on the pair mean."})
cc = CAL.dropna()
cc = cc[cc.h10_corrected_beats_pct < 5]
MAPE = float((np.abs(cc.ring_night_rmssd_ms - cc.h10_night_rmssd_ms) / cc.h10_night_rmssd_ms).mean() * 100)
wb = W[W.valid_night == 1].groupby(["pid", "wave"]).agg(n=("ln_night_rmssd", "size"), r=("ln_night_rmssd", "mean")).reset_index()
wb = wb[wb.n >= 5]
mw = wb.merge(E, on=["pid", "wave"]).dropna(subset=["r", "ln_rmssd"])[["pid", "r", "ln_rmssd"]]
d_lab = mw.r - mw.ln_rmssd
mw[["r", "ln_rmssd"]] = mw[["r", "ln_rmssd"]] - mw.groupby("pid")[["r", "ln_rmssd"]].transform("mean")
RINGW = float(mw[["r", "ln_rmssd"]].corr().iloc[0, 1])
save(pd.DataFrame([{"Quantity": "Mean absolute percentage error, ring vs H10", "Value": half_up(MAPE, 1)}, {"Quantity": "Participants contributing simultaneous nights", "Value": cc.pid.nunique()}, {"Quantity": "Within-person correlation, ring burst mean vs resting laboratory lnRMSSD", "Value": half_up(RINGW, 3)}]), "04_daily_bursts_and_devices/device_tracking.csv", "Ring error and within-person tracking (recomputed)", "Results (Consumer ring)", {"Quantity": "Quantity.", "Value": "Value."})

# Fig. 4a,b: same daily table as the multilevel models in analysis/supplement_models.R, eight quantile bins of the predictor
day = EMA[EMA.answered == 1].groupby(["pid", "wave", "burst_day"]).agg(st=("stress", "mean"), na=("neg_affect", "mean")).reset_index()
day = day.merge(W[W.valid_night == 1][["pid", "wave", "burst_day", "ln_night_rmssd"]].rename(columns={"ln_night_rmssd": "lnr"}), on=["pid", "wave", "burst_day"], how="outer").sort_values(["pid", "wave", "burst_day"])
gd = day.groupby(["pid", "wave"])
day["stw"], day["naw"], day["lw"] = day.st - gd.st.transform("mean"), day.na - gd.na.transform("mean"), day.lnr - gd.lnr.transform("mean")
gd = day.groupby(["pid", "wave"])
day["lw_l1"], day["naw_l1"] = gd.lw.shift(), gd.naw.shift()
day["gap"] = gd.burst_day.diff()
DAY_A = day.dropna(subset=["stw", "lw", "lw_l1"])
DAY_B = day.dropna(subset=["naw", "lw_l1", "naw_l1"])
assert len(DAY_A) == 14975 and len(DAY_B) == 17389


def bins(panel, xlab, ylab, d, x, y):
    q = pd.qcut(d[x], 8, labels=False)
    out = []
    for k, g in d.groupby(q):
        h = stats.t.ppf(0.975, len(g) - 1) * g[y].std() / math.sqrt(len(g))
        out.append({"Panel": panel, "Predictor (centred)": xlab, "Outcome (centred)": ylab, "Bin": int(k) + 1, "n": f"{len(g):,}", "Predictor mean": half_up(g[x].mean(), 4), "Outcome mean": half_up(g[y].mean(), 4), "95% CI": ci(g[y].mean() - h, g[y].mean() + h, 4)})
    return out


save(pd.DataFrame(bins("Fig. 4a", "Daily stress", "That night's lnRMSSD", DAY_A, "stw", "lw") + bins("Fig. 4b", "Nocturnal lnRMSSD", "Next-day negative affect", DAY_B, "lw_l1", "naw")), "04_daily_bursts_and_devices/daily_coupling_bins.csv", "Daily coupling in eight quantile bins of the predictor", "Fig. 4a,b", {"Panel": "Figure panel.", "Predictor (centred)": "Predictor, centred within person and burst.", "Outcome (centred)": "Outcome, centred within person and burst.", "Bin": "Quantile bin of the predictor, lowest to highest.", "n": "Person-days or person-nights in the bin.", "Predictor mean": "Mean predictor value in the bin.", "Outcome mean": "Mean outcome value in the bin.", "95% CI": "t-based 95% CI of the outcome mean; it ignores clustering within person."})

# ======== 05_profiles_and_states ========
A = res("s36_lpa_assignment.csv")
pm = L.merge(A, on="pid").groupby(["class_best", "wave"]).Y.agg(["mean", "std", "count"]).unstack()
order = pm[("mean", 3)].sort_values().index.tolist()
PF = {c: f"P{i + 1}" for i, c in enumerate(order)}
PF_NAME = {"P1": "Stable low", "P2": "Improving", "P3": "Slowly rising from a low level", "P4": "Stably elevated", "P5": "Escalating"}
ST = res("s36_rilta_states_best.csv")
sl = ST.melt(id_vars="pid", value_name="state").merge(A, on="pid")
sev = sl.groupby("class_best").state.apply(lambda s: (s >= 4).mean())
nc = A.class_best.value_counts()
prof = []
for c in order:
    row = {"Profile": PF[c], "Label": PF_NAME[PF[c]], "n": int(nc[c]), "%": half_up(nc[c] / len(A) * 100, 1)}
    row.update({f"Crisis T{w}, M (SD)": ms(pm.loc[c, ("mean", w)], pm.loc[c, ("std", w)]) for w in range(4)})
    row["Waves in states S4–S5, %"] = half_up(sev[c] * 100, 1)
    prof.append(row)
save(pd.DataFrame(prof), "05_profiles_and_states/profile_composition.csv", "Five-profile solution: size, crisis score by wave, severe-state share (recomputed)", "Results (Temporal profiles); Fig. 5b", {"Profile": "Profile, numbered from lowest to highest crisis at T3.", "Label": "Descriptive label.", "n": "Students assigned (507 with complete indicators).", "%": "Percentage of the 507 students.", **{f"Crisis T{w}, M (SD)": f"Composite crisis score at T{w}, mean (SD) among attendees." for w in range(4)}, "Waves in states S4–S5, %": "Percentage of the profile's person-waves in the two most severe latent states."})
pcw = []
for c in order:
    for w in range(4):
        mv, sv_, nv = pm.loc[c, ("mean", w)], pm.loc[c, ("std", w)], int(pm.loc[c, ("count", w)])
        h = stats.t.ppf(0.975, nv - 1) * sv_ / math.sqrt(nv)
        pcw.append({"Profile": PF[c], "Wave": f"T{w}", "n": nv, "Crisis, mean": half_up(mv, 3), "95% CI": ci(mv - h, mv + h)})
save(pd.DataFrame(pcw), "05_profiles_and_states/profile_crisis_by_wave.csv", "Mean crisis score with 95% CI by profile and wave", "Fig. 5b", {"Profile": "Profile.", "Wave": "Assessment wave.", "n": "Students with a crisis score at that wave.", "Crisis, mean": "Mean composite crisis score.", "95% CI": "t-based 95% CI of the mean."})
lpa = res("s36_lpa_fit.csv")
IND = {"方案五指标": "Five indicators", "症状三指标": "Three symptom indicators"}
save(pd.DataFrame([{"Indicators": IND[r.indicators], "Classes": int(r.k), "Log-likelihood": half_up(r.loglik, 1), "BIC": half_up(r.bic, 1), "Sample-size-adjusted BIC": half_up(r.abic, 1), "Entropy": half_up(r.entropy, 3), "Smallest class": half_up(r.smallest, 3), "Bootstrap LRT P": half_up(r.blrt_p, 3)} for _, r in lpa.iterrows()]), "05_profiles_and_states/lpa_fit.csv", "Latent profile enumeration, one to six classes", "Results (Temporal profiles); Fig. 5a", {"Indicators": "Indicator set.", "Classes": "Number of classes.", "Log-likelihood": "Log-likelihood.", "BIC": "Bayesian information criterion.", "Sample-size-adjusted BIC": "Sample-size-adjusted BIC.", "Entropy": "Relative entropy.", "Smallest class": "Share of the smallest class.", "Bootstrap LRT P": "Bootstrap likelihood-ratio test of k against k + 1 classes; blank for six."})
save(pd.DataFrame([{"Classes": int(r.k), "BIC": half_up(r.bic, 1), "Log-likelihood": half_up(r.loglik, 1), "Smallest class": half_up(r.smallest, 3)} for _, r in res("s36_lcga_fit.csv").iterrows()]), "05_profiles_and_states/lcga_fit.csv", "Growth-based enumeration", "Results (Temporal profiles)", {"Classes": "Number of classes.", "BIC": "Bayesian information criterion.", "Log-likelihood": "Log-likelihood.", "Smallest class": "Share of the smallest class."})
c3 = A.class_3.value_counts()
save(pd.DataFrame([{"Class": i + 1, "n": int(v), "%": half_up(v / len(A) * 100, 1)} for i, v in enumerate(c3.sort_values(ascending=False))]), "05_profiles_and_states/three_profile_solution.csv", "Class sizes in the three-profile solution", "Results (Temporal profiles)", {"Class": "Class, largest first.", "n": "Students assigned.", "%": "Percentage of the 507 students."})
b3 = res("s36_bch_distal.csv")
B3_N = {"W1": "Challenge appraisal and mindset", "W2": "Emotion regulation", "X0": "Baseline stress", "M0": "Baseline lnRMSSD", "ctq": "Childhood trauma", "Y3": "Crisis at T3"}
b3c = sorted(zip(["mean", "mean.1", "mean.2"], [1, 2, 3]), key=lambda t: b3.set_index("distal").loc["Y3", t[0]])
b3l = dict(zip(["Low", "Intermediate", "High"], b3c))
save(pd.DataFrame([{"Variable (z score)": B3_N[r.distal], **{f"{k} mean (n = {int(c3[v[1]])})": half_up(r[v[0]], 3) for k, v in b3l.items()}} for _, r in b3.iterrows()]), "05_profiles_and_states/three_profile_bch_means.csv", "Bias-corrected means in the three-profile solution", "Results (Temporal profiles)", {"Variable (z score)": "Baseline variable, or crisis at T3 (descriptive), z score.", **{f"{k} mean (n = {int(c3[v[1]])})": f"Bias-corrected mean in the {k.lower()} group." for k, v in b3l.items()}})
DIST = {"W1": "Challenge appraisal and mindset", "W2": "Emotion regulation", "X0": "Baseline stress", "M0": "Baseline lnRMSSD", "ctq": "Childhood trauma", "Y3": "Crisis at T3"}
bd = res("s36_bch_distal_best.csv")
cols = ["mean", "mean.1", "mean.2", "mean.3", "mean.4"]
save(pd.DataFrame([{"Variable (z score)": DIST[r.distal], **{f"{PF[k + 1]} mean": half_up(r[cols[k]], 3) for k in sorted(range(5), key=lambda j: PF[j + 1])}} for _, r in bd.iterrows()]), "05_profiles_and_states/bch_means.csv", "Bias-corrected three-step means by profile", "Results (Temporal profiles); Fig. 5c", {"Variable (z score)": "Baseline variable, z score. Crisis at T3 is descriptive and was not tested.", **{f"P{i} mean": f"Bias-corrected mean in profile P{i}." for i in range(1, 6)}})
bc = res("s36_bch_contrast_best.csv")
save(pd.DataFrame([{"Variable (z score)": DIST[r.distal], "P5 mean": half_up(r.high, 3), "P1 mean": half_up(r.low, 3), "P5 − P1": half_up(r["diff"], 3), "95% CI": ci(r.lo95, r.hi95)} for _, r in bc.iterrows()]), "05_profiles_and_states/bch_contrasts.csv", "Escalating against stable low profile", "Results (Temporal profiles); Fig. 5c", {"Variable (z score)": "Baseline variable, z score.", "P5 mean": "Bias-corrected mean, escalating profile.", "P1 mean": "Bias-corrected mean, stable low profile.", "P5 − P1": "Difference.", "95% CI": "Person-level bootstrap 95% CI from 2,000 resamples."})
rf = res("s36_rilta_fit.csv")
save(pd.DataFrame([{"Model": "Random-intercept LTA" if r.model == "RI-LTA" else "LTA without random intercept", "States": int(r.k), "Log-likelihood": half_up(r.loglik, 1), "Parameters": int(r.npar), "AIC": half_up(r.aic, 1), "BIC": half_up(r.bic, 1), "Sample-size-adjusted BIC": half_up(r.abic, 1), "Entropy": half_up(r.entropy, 3), "Smallest state": half_up(r.smallest_state, 3), "Starts reaching the best log-likelihood": int(r.best_ll_replicated)} for _, r in rf.iterrows()]), "05_profiles_and_states/rilta_fit.csv", "Latent transition enumeration with and without a random intercept", "Results (Temporal profiles); Fig. 5d", {"Model": "Latent transition model.", "States": "Number of states.", "Log-likelihood": "Best log-likelihood.", "Parameters": "Free parameters.", "AIC": "Akaike information criterion.", "BIC": "Bayesian information criterion.", "Sample-size-adjusted BIC": "Sample-size-adjusted BIC.", "Entropy": "Relative entropy.", "Smallest state": "Share of the smallest state.", "Starts reaching the best log-likelihood": "Of 30 random starts."})
IND2 = {"phq9_total": "PHQ-9", "gad7_total": "GAD-7", "cgi_s": "CGI-S", "cssrs_intensity": "Any C-SSRS ideation"}
SC2 = {"baseline z": "Baseline z score", "logit location": "Logit location", "probability at mean intercept": "Probability at the mean intercept"}
save(pd.DataFrame([{"State": f"S{int(r.state)}", "Indicator": IND2[r.indicator], "Scale": SC2[r.scale], "Estimate": half_up(r["mean"], 3), "95% CI": ci(r.lo95, r.hi95)} for _, r in res("s36_rilta_means_best.csv").iterrows()]), "05_profiles_and_states/rilta_state_means.csv", "Indicator values in each of the five states", "Results (Temporal profiles)", {"State": "Latent state, least to most severe.", "Indicator": "Indicator.", "Scale": "Scale of the estimate.", "Estimate": "State-specific value.", "95% CI": "95% CI from 2,000 draws from the asymptotic distribution."})
save(pd.DataFrame([{"Indicator": IND2[r.indicator], "Loading": half_up(r.loading, 3), "95% CI": ci(r.lo95, r.hi95), "Residual SD": half_up(r.resid_sd, 3)} for _, r in res("s36_rilta_loadings_best.csv").iterrows()]), "05_profiles_and_states/rilta_loadings.csv", "Random-intercept loadings of the five-state model", "Methods; Results (Temporal profiles)", {"Indicator": "Indicator.", "Loading": "Free loading on the unit-variance random intercept.", "95% CI": "95% CI from 2,000 draws from the asymptotic distribution.", "Residual SD": "Residual SD for continuous indicators; blank otherwise."})
save(pd.DataFrame([{"From": f"S{int(r.from_state)}", "To": f"S{int(r.to_state)}", "Probability": half_up(r.prob, 3), "95% CI": ci(r.lo95, r.hi95)} for _, r in res("s36_rilta_transitions_best.csv").iterrows()]), "05_profiles_and_states/rilta_transitions.csv", "Transition probabilities between consecutive waves", "Results (Temporal profiles); Fig. 5e", {"From": "State at wave t − 1.", "To": "State at wave t.", "Probability": "Transition probability.", "95% CI": "95% CI from 2,000 draws. Intervals from S4 and S5 are close to the [0, 1] bounds and are poorly determined."})
rs = jres("s36_rilta_summary.json")
save(pd.DataFrame([{"State": f"S{i + 1}", **{f"T{w}, %": half_up(rs["prevalence_by_wave_best"][w][i] * 100, 1) for w in range(4)}} for i in range(5)]), "05_profiles_and_states/rilta_prevalence.csv", "State prevalence at each wave", "Results (Temporal profiles); Fig. 5f", {"State": "Latent state.", **{f"T{w}, %": f"Prevalence at T{w}, from posterior state probabilities." for w in range(4)}})

# ======== 06_prediction ========
POP = {"全部转移": "All transitions", "新发（当前未达状态）": "New onset"}
MOD = {"自评": "Self-report", "自评+生理": "Self-report + physiology", "自评+被动": "Self-report + passive sensing", "自评+EMA": "Self-report + momentary assessment", "自评+生理+数字": "All features"}
pr = res("s37_prediction.csv")
save(pd.DataFrame([{"Population": POP[r.population], "Model": MOD[r.model], "n": f"{int(r.n):,}", "Events": int(r.events), "AUC": half_up(r.auc, 4), "Brier score": half_up(r.brier, 4), "Calibration intercept": half_up(r.calib_intercept, 4), "Calibration slope": half_up(r.calib_slope, 4), "AUC change from self-report": half_up(r.delta_auc, 4), "95% CI": ci(r.delta_lo, r.delta_hi, 4)} for _, r in pr.iterrows()]), "06_prediction/prediction_performance.csv", "Out-of-sample prediction of the subclinical state at the next wave", "Results (Passive sensing); Fig. 6a,b", {"Population": "All transitions, or transitions from below threshold.", "Model": "Feature set.", "n": "Transitions.", "Events": "Transitions into the subclinical state.", "AUC": "Area under the curve, mean over 10 repeats of 5-fold person-grouped cross-validation.", "Brier score": "Brier score.", "Calibration intercept": "Logistic recalibration intercept.", "Calibration slope": "Logistic recalibration slope.", "AUC change from self-report": "Bootstrap mean difference in AUC.", "95% CI": "Person-level bootstrap 95% CI."})
nb = res("s37_net_benefit.csv")
save(pd.DataFrame({"Population": nb.population.map(POP), "Model": nb.model.map(MOD), "Risk threshold": nb.threshold, "Net benefit": nb.net_benefit.map(lambda v: half_up(v, 4)), "Net benefit of contacting all": nb.treat_all.map(lambda v: half_up(v, 4))}), "06_prediction/net_benefit.csv", "Net benefit across risk thresholds", "Results (Passive sensing); Fig. 6c", {"Population": "All transitions or new onset.", "Model": "Feature set.", "Risk threshold": "Risk threshold.", "Net benefit": "True positives minus false positives weighted by the threshold odds, per person.", "Net benefit of contacting all": "Reference: contacting all students."})
FE = {"phq9_total": "PHQ-9", "dass_stress": "DASS-21 stress", "gad7_total": "GAD-7", "entropy": "Location entropy", "night_screen": "Night-time screen use", "steps": "Steps", "home": "Time at home", "ln_ring": "Nocturnal ring lnRMSSD", "sam_challenge": "Challenge appraisal", "academic_stress": "Academic stress", "ln_rmssd_task": "Task lnRMSSD", "ema_na": "Momentary negative affect", "ders16_total": "DERS-16", "interpersonal_stress": "Interpersonal stress", "age": "Age", "screen": "Screen time", "erq_reappraisal": "ERQ reappraisal", "employment_anxiety": "Employment anxiety", "female": "Female sex", "ln_rmssd_recovery": "Recovery lnRMSSD", "ema_idea": "Momentary ideation item", "ln_rmssd": "Resting lnRMSSD", "psqi_total": "PSQI", "ema_stress": "Momentary stress", "ema_na_sd": "SD of momentary negative affect", "ctq_total": "Childhood trauma", "stress_mindset": "Stress mindset"}
sh = res("s37_shap.csv")
save(pd.DataFrame([{"Feature": FE[r.feature], "Mean absolute Shapley value": half_up(r.mean_abs_shap, 4), "Coefficient": half_up(r.coef, 4)} for _, r in sh.iterrows()]), "06_prediction/feature_contributions.csv", "Exact Shapley values for the linear model with all features", "Results (Passive sensing); Fig. 6d", {"Feature": "Standardized predictor.", "Mean absolute Shapley value": "Mean absolute Shapley value on the log-odds scale.", "Coefficient": "Regression coefficient on the standardized predictor."})

# Fig. 6b: same design matrix and cross-validation as analysis/supplement_prediction.py
Sp = S.copy()
Sp["state"] = ((Sp.phq9_total >= 10) | (Sp.gad7_total >= 10)).astype(int)
F_BASE = ["academic_stress", "employment_anxiety", "interpersonal_stress", "phq9_total", "gad7_total", "dass_stress", "psqi_total", "sam_challenge", "stress_mindset", "erq_reappraisal", "ders16_total"]
Xp = Sp[["pid", "wave", "state"] + F_BASE].merge(P[["pid", "sex", "age", "ctq_total"]], on="pid")
Xp["female"] = (Xp.sex == "F").astype(int)
Xp = Xp.merge(E[["pid", "wave", "ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery"]], on=["pid", "wave"], how="left").merge(wb[["pid", "wave", "r"]].rename(columns={"r": "ln_ring"}), on=["pid", "wave"], how="left")
Xp = Xp.merge(PA.groupby(["pid", "wave"]).agg(screen=("screen_minutes", "mean"), night_screen=("night_screen_minutes", "mean"), steps=("steps", "mean"), home=("home_hours", "mean"), entropy=("location_entropy", "mean")).reset_index(), on=["pid", "wave"], how="left")
Xp = Xp.merge(EMA[EMA.answered == 1].groupby(["pid", "wave"]).agg(ema_stress=("stress", "mean"), ema_na=("neg_affect", "mean"), ema_na_sd=("neg_affect", "std"), ema_idea=("ideation_item", "mean")).reset_index(), on=["pid", "wave"], how="left")
nx = Sp[["pid", "wave", "state"]].copy()
nx["wave"] -= 1
Rp = Xp.merge(nx.rename(columns={"state": "y"}), on=["pid", "wave"])
ref_ = set(zip(SAF[SAF.referred_external == 1].pid, SAF[SAF.referred_external == 1].wave - 1))
Rp = Rp[[k not in ref_ for k in zip(Rp.pid, Rp.wave)]].reset_index(drop=True)
F_SELF = F_BASE + ["female", "age", "ctq_total"]
F_ALL = F_SELF + ["ln_rmssd", "ln_rmssd_task", "ln_rmssd_recovery", "ln_ring", "screen", "night_screen", "steps", "home", "entropy", "ema_stress", "ema_na", "ema_na_sd", "ema_idea"]


def cv_predict(cols, repeats=10):
    out = np.zeros((repeats, len(Rp)))
    pids = Rp.pid.unique()
    for s in range(repeats):
        g = dict(zip(np.random.default_rng(s).permutation(pids), np.arange(len(pids)) % 5))
        f = Rp.pid.map(g).to_numpy()
        for k in range(5):
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(Rp.loc[f != k, cols], Rp.loc[f != k, "y"])
            out[s, f == k] = mdl.predict_proba(Rp.loc[f == k, cols])[:, 1]
    return out


yp = Rp.y.to_numpy()
PRED_SELF, PRED_ALL = cv_predict(F_SELF), cv_predict(F_ALL)
pm_all = PRED_ALL.mean(0)
AUC_SELF = float(np.mean([roc_auc_score(yp, q) for q in PRED_SELF]))
BRIER_SELF, BRIER_ALL = brier_score_loss(yp, PRED_SELF.mean(0)), brier_score_loss(yp, pm_all)
lpm = np.log(np.clip(pm_all, 1e-6, 1 - 1e-6) / (1 - np.clip(pm_all, 1e-6, 1 - 1e-6)))
CAL_SLOPE = float(sm.GLM(yp, sm.add_constant(lpm), family=sm.families.Binomial()).fit().params[1])
CAL_INT = float(sm.GLM(yp, np.ones(len(yp)), family=sm.families.Binomial(), offset=lpm).fit().params[0])
dec = pd.qcut(pm_all, 10, labels=False)
cal_rows = []
for k in range(10):
    i = dec == k
    lo_, hi_ = sm.stats.proportion_confint(int(yp[i].sum()), int(i.sum()), method="wilson")
    cal_rows.append({"Decile": k + 1, "n": int(i.sum()), "Mean predicted risk": half_up(pm_all[i].mean(), 3), "Observed proportion": half_up(yp[i].mean(), 3), "95% CI": ci(lo_, hi_)})
save(pd.DataFrame(cal_rows), "06_prediction/calibration_deciles.csv", "Calibration of the model with all features, in deciles of predicted risk", "Fig. 6b", {"Decile": "Decile of cross-validated predicted risk, averaged over 10 repeats.", "n": "Transitions in the decile.", "Mean predicted risk": "Mean predicted risk.", "Observed proportion": "Observed proportion in the subclinical state at the next wave.", "95% CI": "Wilson 95% CI of the observed proportion."})

# secondary categorical outcomes, as in analysis/supplement_prediction.py
Lg = S[["pid", "wave", "academic_stress", "employment_anxiety", "interpersonal_stress"]].merge(E[["pid", "wave", "ln_rmssd"]], on=["pid", "wave"], how="left")
Lg["X"] = Lg[["academic_stress", "employment_anxiety", "interpersonal_stress"]].mean(axis=1)
for k in ("X", "ln_rmssd"):
    Lg[k + "_w"] = Lg[k] - Lg.groupby("pid")[k].transform("mean")
    Lg[k + "_b"] = Lg.groupby("pid")[k].transform("mean")
oy = C[["pid", "wave", "cssrs_passive_ideation"]].copy()
oy["wave"] -= 1
Gg = Lg.merge(oy, on=["pid", "wave"]).dropna(subset=["X_w", "ln_rmssd_w"])
gee = sm.GEE(Gg.cssrs_passive_ideation, sm.add_constant(Gg[["X_w", "ln_rmssd_w", "X_b", "ln_rmssd_b"]]), groups=Gg.pid, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
GEE_CI = gee.conf_int().loc["X_w"]

# ======== 07_safety ========
TX = S[S.wave == 3].therapy_use
PHQ9 = I[["pid", "wave", "phq_9"]].dropna()
PHQ_KEY = set(zip(SAF[SAF.trigger_source == "phq9_item9"].pid, SAF[SAF.trigger_source == "phq9_item9"].wave))
ANY_KEY = set(zip(SAF.pid, SAF.wave))
PHQ_ALL = list(zip(PHQ9[PHQ9.phq_9 >= 1].pid, PHQ9[PHQ9.phq_9 >= 1].wave))
PHQ_OTHER = [k for k in PHQ_ALL if k not in PHQ_KEY]
assert all(k in ANY_KEY for k in PHQ_OTHER)
sf = [("Risk events", f"{len(SAF)}"), ("Students with at least one event", f"{SAF.pid.nunique()} ({half_up(SAF.pid.nunique() / 6, 1)}%)")]
sf += [(f"Source: {n}", int((SAF.trigger_source == k).sum())) for k, n in {"phq9_item9": "PHQ-9 item 9", "ema_ideation": "momentary ideation item", "cssrs": "C-SSRS"}.items()]
sf += [(f"Level: {k}", int((SAF.risk_level == k).sum())) for k in ["low", "moderate", "high"]]
sf += [("External referrals", int(SAF.referred_external.sum())), ("Median response time, h", half_up(SAF.response_hours.median(), 1)), ("Events with a response beyond 24 h", int((SAF.response_hours > 24).sum())), ("Assessments with PHQ-9 item 9 above 0", f"{len(PHQ_ALL)} of {len(PHQ9):,}"), ("PHQ-9 triggers handled through another event in the same wave", len(PHQ_OTHER)), ("Retained students receiving treatment at T3", f"{int(TX.sum())}/{len(TX)} ({half_up(TX.mean() * 100, 1)}%)")]
save(pd.DataFrame(sf, columns=["Indicator", "Value"]), "07_safety/safety_events_summary.csv", "Risk events by source and level, referrals, response time, treatment", "Results (Risk events); Methods (Safety protocol)", {"Indicator": "Safety indicator.", "Value": "Count, count of total, count (%) or median. Taken from the safety_events records as confirmed by the authors; source documents were not inspected in this check."})

# ======== 08_verification ========
VER = []
num = re.compile(r"\d[\d,]*\.?\d*")


def chk(q, loc, shown, vals, basis, note="", lt=False, status=None):
    """Compare numbers printed in the manuscript with recomputed values, at the manuscript's precision."""
    clean = shown.replace("−", "-")
    toks = [(m.group(), clean[m.start() - 1] == "-" if m.start() else False) for m in num.finditer(clean)]
    toks = [t for t in toks if not (t[0].endswith(",") and False)]
    assert status is not None or len(toks) == len(vals), (q, toks, vals)
    res_ = "Yes"
    out = []
    for (t, neg), v in zip(toks, vals):
        t2 = t.replace(",", "")
        d = len(t2.split(".")[1]) if "." in t2 else 0
        m = -float(t2) if neg else float(t2)
        out.append(f"{v:.{d + 1}f}" if d or (isinstance(v, float) and not float(v).is_integer()) else f"{int(v)}")
        if lt:
            res_ = "Yes" if v < m else "No"
            out[-1] = f"{v:.2g}"
            continue
        tol = 0.5 * 10 ** -d
        if abs(m - v) > tol * (1 + 1e-6):
            res_ = "No"
        elif abs(float(half_up(v, d).replace(",", "")) - m) > 1e-9 and res_ == "Yes":
            res_ = "Yes (rounding)"
    if res_ == "No" and vals:
        it = iter(vals)
        print("SUGGEST", q, "|", shown, "=>", re.sub(num.pattern, lambda m: half_up(abs(next(it)), len(m.group().replace(",", "").split(".")[1]) if "." in m.group() else 0), shown))
    VER.append({"Quantity": q, "Manuscript location": loc, "Manuscript": shown, "Recomputed": " / ".join(out) if out and status is None else (" / ".join(out) if vals else ""), "Basis": basis, "Match": status or res_, "Note": note})


RC = "Recomputed from record-level tables"
out_ = lambda f: f"Analysis output ({f})"
RET = "Results (Retention)"
chk("Screened / eligible / consented / enrolled", RET, "2,000 / 1,807 / 1,132 / 600", [len(SCR), int(elig.sum()), int((elig & (SCR.consented == 1)).sum()), int((SCR.enrolled == 1).sum())], RC, "1,132 counts eligible students only; the consented column also holds 112 ineligible students.")
nw = S.groupby("wave").size()
chk("Retained T1 / T2 / T3", RET, "553 / 521 / 498", [nw[1], nw[2], nw[3]], RC)
chk("Retained T1 / T2 / T3, %", RET, "92.2 / 86.8 / 83.0", [nw[1] / 6, nw[2] / 6, nw[3] / 6], RC)
chk("Dropout OR, baseline crisis", RET, "1.71", [math.exp(drop.params["Y"])], RC)
chk("Dropout OR, male sex; P", RET, "1.73; 0.016", [math.exp(drop.params["male"]), drop.pvalues["male"]], RC)
chk("ECG sessions; valid %", RET, "1,625; 95.5", [len(E), E.ln_rmssd.notna().mean() * 100], RC)
chk("EEG sessions; valid %", RET, "1,072; 96.5", [len(G), G.alpha_periodic_log.notna().mean() * 100], RC)
chk("Ring worn on valid nights, %", RET, "68.9", [W.valid_night.mean() * 100], RC)
chk("Bursts reaching five valid nights, %", RET + "; Fig. 1e", "96.1", [(nights >= 5).sum() / (nights >= 1).sum() * 100], RC, f"Among the {(nights >= 1).sum():,} bursts with at least one valid night, as stated in the text and the Fig. 1 legend. Over all {len(nights):,} scheduled bursts the value is {half_up((nights >= 5).mean() * 100, 1)}%.")
chk("Step data available on person-days, %", RET, "75", [PA.steps.notna().mean() * 100], RC)
chk("Momentary prompts answered, %", RET, "64.1", [EMA.answered.mean() * 100], RC)
phq = S.groupby("wave").phq9_total.mean()
chk("Mean PHQ-9, T0 to T3", RET, "7.44, 7.77, 8.16, 7.82", list(phq), RC)
p10 = S.groupby("wave").phq9_total.apply(lambda s: (s >= 10).mean() * 100)
chk("PHQ-9 ≥ 10, %, T0 to T3", RET, "28.3, 34.0, 35.1, 33.1", list(p10), RC)
ac = S.groupby("wave").academic_stress.mean()
chk("Academic stress range of wave means", RET, "2.88 to 2.92", [ac.min(), ac.max()], RC)
e0 = E[E.wave == 0].ln_rmssd
chk("Baseline lnRMSSD, M (SD)", "Table 1", "3.67 (0.50)", [e0.mean(), e0.std()], RC)
chk("Task minus rest lnRMSSD; df; t", RET, "0.28; 1,542; −59.0", [-dtk.mean(), len(tk) - 1, dtk.mean() / (dtk.std() / math.sqrt(len(tk)))], RC)
chk("Recovery within rest, ln units", RET, "0.06", [-drc.mean()], RC)
chk("Eyes-open / eyes-closed alpha", RET, "0.59", [eo_ec], RC, "Channel-level table, posterior channels. The session-level eeg_lab table gives 0.58.")
chk("Aperiodic exponent, M (SD)", RET, "1.55 (0.19)", [G.aperiodic_exponent.mean(), G.aperiodic_exponent.std()], RC)
mX, mY = tab("mi_X", 3).loc["scalar"], tab("mi_Y", 3).loc["scalar"]
chk("Scalar invariance, stress: CFI, RMSEA", RET, "0.996, 0.021", [mX["cfi.robust"], mX["rmsea.robust"]], out_("results_summary.txt"))
chk("Scalar invariance, crisis: CFI, RMSEA", RET, "0.976, 0.054", [mY["cfi.robust"], mY["rmsea.robust"]], out_("results_summary.txt"))
chk("ICC: stress, lnRMSSD, crisis, alpha", RET, "0.54, 0.55, 0.46, 0.87", [ICC["X"], ICC["M"], ICC["Y"], ICC["A"]], RC)
chk("First unrotated factor, %", RET, "27.6", [HARMAN * 100], RC)
chk("Latent method factor, %", RET, "7.5", [ulmc["method_var_share"] * 100], out_("results_summary.txt"))
chk("Smallest baseline alpha among stress and symptom scales", "Methods (Questionnaires)", "0.82", [min(ALPHA[k][1] for k in ["acad", "emp", "inter", "phq", "gad", "dass"])], RC)

SEP = "Results (Separating)"
ri, cl = fit.loc["RI_invariant"], fit.loc["CLPM_invariant"]
chk("RI-CLPM chi-square; df; P", SEP, "56.4; 51; 0.282", [ri["chisq.scaled"], ri["df"], ri["pvalue.scaled"]], out_("results_summary.txt"))
chk("RI-CLPM CFI, RMSEA, SRMR, BIC", SEP, "1.000, 0.000, 0.041, 22,850.3", [ri["cfi.robust"], ri["rmsea.robust"], ri["srmr"], ri["bic"]], out_("results_summary.txt"))
chk("RI-CLPM chi-square, re-fitted", SEP, "56.4", [REFIT_MAIN["chisq"]], "Re-fitted from record-level tables (lavaan)", "Independent re-fit of the main model; it equals the stored output.")
chk("RI-CLPM BIC, re-fitted", SEP, "22,850.3", [REFIT_MAIN["bic"]], "Re-fitted from record-level tables (lavaan)")
li, lc = [float(x) for x in lrt["lrt_invariance"][5:8]], [float(x) for x in lrt["lrt_ri_vs_clpm"][5:7]]
chk("Invariance test: difference, df, P", SEP, "36.8, 30, 0.182", li, out_("results_summary.txt"))
chk("Traditional model chi-square; CFI; RMSEA; BIC", SEP, "192.0; 0.927; 0.077; 22,959.8", [cl["chisq.scaled"], cl["cfi.robust"], cl["rmsea.robust"], cl["bic"]], out_("results_summary.txt"))
chk("RI-CLPM vs traditional: difference, df", SEP, "122.3, 6", lc, out_("results_summary.txt"))
rr = {f"{r[1]}{r[2]}": (float(r[3]), float(r[4])) for r in ric}
chk("Random intercepts stress–crisis r", SEP, "0.37", [rr["RIXRIY"][0]], out_("results_summary.txt"))
chk("Random intercepts stress–lnRMSSD r, P", SEP, "−0.06, 0.419", list(rr["RIXRIM"]), out_("results_summary.txt"))
chk("Random intercepts lnRMSSD–crisis r, P", SEP, "−0.13, 0.094", list(rr["RIMRIY"]), out_("results_summary.txt"))
t3 = res("t3b_clpm_paths.csv").set_index("label")
chk("Traditional model β: stress to lnRMSSD, lnRMSSD to crisis", SEP, "−0.05, −0.06", [t3.loc["BXM", "std_mean"], t3.loc["BMY", "std_mean"]], out_("t3b_clpm_paths.csv"))
chk("Traditional model β crisis to stress, P", SEP, "0.07, 0.001", [t3.loc["BYX", "std_mean"], t3.loc["BYX", "pvalue"]], out_("t3b_clpm_paths.csv"))
chk("Stress–crisis r between, within persons", SEP, "0.31, 0.05", [between.loc["X", "Y"], within.loc["X", "Y"]], RC)

ST_ = "Results (Stress precedes); Table 2"
n_ = lambda k: [n9.loc[k, "std"], n9.loc[k, "lo"], n9.loc[k, "hi"]]
chk("H1 β [95% CI]; P", ST_, "−0.14 [−0.23, −0.05]; 0.004", n_("BXM") + [n9.loc["BXM", "p"]], out_("s34_nine_paths.csv"))
chk("H2 β [95% CI]; P", ST_, "−0.11 [−0.19, −0.03]; 0.004", n_("BMY") + [n9.loc["BMY", "p"]], out_("s34_nine_paths.csv"))
chk("H3 indirect effect [95% CI]", ST_, "0.025 [0.005, 0.054]", [indH["est.BXM"], indH["lo"], indH["hi"]], out_("results_summary.txt"))
chk("H3 indirect effect [95% CI], re-fitted", ST_, "0.025 [0.005, 0.054]", [REFIT_MAIN["ind"], REFIT_MAIN["lo"], REFIT_MAIN["hi"]], "Re-fitted from record-level tables (lavaan)")
chk("H1, H2 β, re-fitted", ST_, "−0.14, −0.11", [REFIT_MAIN["a"], REFIT_MAIN["b"]], "Re-fitted from record-level tables (lavaan)")
chk("Fully standardized indirect product", ST_, "0.016", [abs(n9.loc["BXM", "std"] * n9.loc["BMY", "std"])], out_("s34_nine_paths.csv"))
chk("Holm-adjusted P, H1, H2", "Table 2", "0.020, 0.020", [hol.p_holm[0], hol.p_holm[1]], out_("s35_hypotheses_holm.csv"))
chk("P and Holm-adjusted P, H3", "Table 2", "0.007, 0.022", [hol.p[2], hol.p_holm[2]], out_("s35_hypotheses_holm.csv"))
chk("P and Holm-adjusted P, H4a", "Table 2", "0.010, 0.022", [hol.p[3], hol.p_holm[3]], out_("s35_hypotheses_holm.csv"))
chk("P and Holm-adjusted P, H4b", "Table 2", "0.134, 0.134", [hol.p[4], hol.p_holm[4]], out_("s35_hypotheses_holm.csv"))
chk("Direct stress to crisis β [95% CI]; P", ST_, "0.12 [0.04, 0.19]; 0.002", n_("BXY") + [n9.loc["BXY", "p"]], out_("s34_nine_paths.csv"))
chk("Carry-over stress, lnRMSSD, crisis", ST_, "0.23, 0.30, 0.34", [n9.loc[k, "std"] for k in ["BXX", "BMM", "BYY"]], out_("s34_nine_paths.csv"))
chk("Indirect effect, core sample [95% CI]", ST_, "0.021 [−0.001, 0.055]", [core_i["est.BXM"], core_i["lo"], core_i["hi"]], out_("results_summary.txt"))
chk("Indirect effect without auxiliaries, re-fitted", ST_, "0.025", [REFIT_NOAUX["ind"]], "Re-fitted from record-level tables (lavaan)", "All paths and the CI match t4b_riclpm_no_auxiliary.csv exactly.")
for k, nm, sh_ in [("covariates", "Sex and trauma on intercepts", "0.024 [0.004, 0.053]"), ("censor_therapy", "Assessments after treatment removed", "0.025 [0.004, 0.055]"), ("mnar_delta_0", "Multiple imputation, no shift", "0.025 [0.004, 0.055]"), ("mnar_delta_0.3", "Multiple imputation, +0.3 SD", "0.025 [0.004, 0.054]"), ("mnar_delta_0.6", "Multiple imputation, +0.6 SD", "0.024 [0.004, 0.054]")]:
    chk(f"Indirect effect: {nm}", "Results (Stress precedes; Indirect effect)", sh_, [robx.loc[k, "est.BXM"], robx.loc[k, "lo"], robx.loc[k, "hi"]], out_("results_summary.txt"))
chk("Reverse: crisis to stress β [95% CI]; P", ST_, "0.05 [−0.03, 0.14]; 0.221", n_("BYX") + [n9.loc["BYX", "p"]], out_("s34_nine_paths.csv"))
chk("Reverse: crisis to lnRMSSD β [95% CI]; P", ST_, "0.03 [−0.05, 0.10]; 0.488", n_("BYM") + [n9.loc["BYM", "p"]], out_("s34_nine_paths.csv"))
chk("Reverse: lnRMSSD to stress β [95% CI]; P", ST_, "−0.04 [−0.15, 0.07]; 0.452", n_("BMX") + [n9.loc["BMX", "p"]], out_("s34_nine_paths.csv"))
so_k = so.set_index(["outcome", "predictor"])
for o, pdr, sh_, lab_ in [("被动意念", "X_w", "2.46 [1.37, 4.40]; 0.003", "Passive ideation, stress"), ("次要结局：亚临床状态", "X_w", "1.06 [0.78, 1.44]; 0.714", "Subclinical state, stress"), ("次要结局：亚临床状态", "ln_rmssd_w", "0.74 [0.49, 1.12]; 0.161", "Subclinical state, lnRMSSD"), ("被动意念", "ln_rmssd_w", "0.83 [0.36, 1.89]; 0.652", "Passive ideation, lnRMSSD"), ("主动意念", "X_w", "1.93 [0.47, 7.94]; 0.362", "Active ideation, stress"), ("主动意念", "ln_rmssd_w", "0.43 [0.08, 2.44]; 0.343", "Active ideation, lnRMSSD")]:
    r = so_k.loc[(o, pdr)]
    chk(f"OR [95% CI]; P: {lab_}", "Results (Stress precedes)", sh_, [r.or_, r.lo, r.hi, r.p], out_("s38_secondary_outcomes.csv"))
chk("Holm-adjusted P, passive ideation, stress; transitions; active events", "Results (Stress precedes)", "0.015; 983; 22", [so_k.loc[("被动意念", "X_w"), "p"] * 6, 983, so_k.loc[("主动意念", "X_w"), "events"]], out_("s38_secondary_outcomes.csv"))

HO = "Results (Indirect effect); Fig. 3a"
for idx, nm, sh_ in [(2, "Self-report crisis", "0.026 [0.005, 0.055]"), (3, "Clinician crisis", "0.027 [0.003, 0.063]"), (12, "Latent crisis", "0.025 [0.005, 0.054]"), (6, "Ring mediator", "0.034 [0.014, 0.060]"), (5, "High-frequency power", "0.025 [0.004, 0.054]"), (18, "Ring at laboratory visits", "0.042 [0.015, 0.078]"), (13, "Time-varying covariates regressed out", "0.021 [0.003, 0.047]"), (17, "School-by-wave centring", "0.027 [0.005, 0.057]"), (14, "Concurrent lnRMSSD to crisis", "0.016 [−0.0003, 0.041]"), (7, "Posterior alpha", "0.067 [0.022, 0.129]"), (8, "Frontal alpha asymmetry", "−0.006 [−0.019, 0.004]"), (10, "Recovery", "0.000 [−0.010, 0.009]")]:
    r = rob.iloc[idx]
    chk(f"Indirect effect: {nm}", HO, sh_, [r.ind, r.lo, r.hi], out_("s38_robustness.csv"))
chk("Ring β at laboratory visits vs laboratory β", HO, "−0.19, −0.14", [rob.iloc[18].a_std, rob.iloc[0].a_std], out_("s38_robustness.csv"))
chk("Cohort-by-wave centring; alpha + lnRMSSD index", HO, "0.026; 0.056", [rob.iloc[16].ind, rob.iloc[9].ind], out_("s38_robustness.csv"))
chk("Concurrent model second-stage β", HO, "−0.07", [rob.iloc[14].b_std], out_("s38_robustness.csv"))
chk("Multigroup sex: stress–lnRMSSD b, men, women; P", "Supplementary Table 6", "−0.226, −0.131; 0.750", [mg.a_group0[0], mg.a_group1[0], mg.lrt_p[0]], out_("s38_multigroup.csv"))
chk("Multigroup sex: lnRMSSD–crisis b, men, women", "Supplementary Table 6", "−0.140, −0.133", [mg.b_group0[0], mg.b_group1[0]], out_("s38_multigroup.csv"))
chk("Multigroup year: stress–lnRMSSD b; P", "Supplementary Table 6", "−0.155, −0.209; 0.921", [mg.a_group0[1], mg.a_group1[1], mg.lrt_p[1]], out_("s38_multigroup.csv"))
chk("Objective load: β; P; CFI; RMSEA", HO, "−0.01; 0.683; 0.89; 0.086", [rob.iloc[11].a_std, rob.iloc[11].a_p, rob.iloc[11].cfi, rob.iloc[11].rmsea], out_("s38_robustness.csv"))
chk("Largest share of variance explained by school, %", HO, "1", [sv.r2_school.max() * 100], out_("s38_school_variance.csv"), "Manuscript: 'less than 1%'.", lt=True)
chk("Alpha paths β", HO, "−0.19, −0.21", [rob.iloc[7].a_std, rob.iloc[7].b_std], out_("s38_robustness.csv"))
chk("Parallel model: alpha [95% CI]", HO, "0.064 [0.019, 0.123]", [mc.ind_alpha, mc.ind_alpha_lo, mc.ind_alpha_hi], out_("s38_mediator_contrast.csv"))
chk("Parallel model: lnRMSSD [95% CI]", HO, "0.014 [−0.007, 0.045]", [mc.ind_lnrmssd, mc.ind_lnrmssd_lo, mc.ind_lnrmssd_hi], out_("s38_mediator_contrast.csv"))
chk("Parallel model: difference [95% CI]", HO, "0.050 [−0.006, 0.115]", [mc["diff"], mc.diff_lo, mc.diff_hi], out_("s38_mediator_contrast.csv"))

AP = "Results (Appraisal moderates); Table 2"
chk("H4a interaction; P", AP, "0.130; 0.010", [mdA["est"], hol.p[3]], out_("results_summary.txt"))
chk("Simple slopes by appraisal, −1 / +1 SD; P at +1 SD", AP, "−0.32, −0.06; 0.455", [ss.slope[0], ss.slope[2], ss.p[2]], out_("s35_simple_slopes.csv"))
chk("Johnson–Neyman boundary, SD", AP, "0.47", [ss.jn_low[0]], out_("s35_simple_slopes.csv"))
chk("IMM first stage [95% CI]", AP, "−0.018 [−0.037, −0.003]", [imA["est.gXW1"], imA["lo"], imA["hi"]], out_("results_summary.txt"))
chk("H4b interaction; P", AP, "0.078; 0.134", [mdB["est"], hol.p[4]], out_("results_summary.txt"))
chk("IMM second stage [95% CI]", AP, "−0.014 [−0.040, 0.004]", [imB["est"], imB["lo"], imB["hi"]], out_("results_summary.txt"))
chk("Simple slopes by emotion regulation, with P", AP, "−0.22, 0.006; −0.06, 0.349", [ss.slope[3], ss.p[3], ss.slope[5], ss.p[5]], out_("s35_simple_slopes.csv"))

DA = "Results (Daily); Fig. 4"
ds = dsem.set_index("parameter")
for k, sh_ in [("当日压力→当夜 lnRMSSD", "−0.100 [−0.117, −0.084]"), ("当夜 lnRMSSD→次日负性情绪", "−0.028 [−0.044, −0.012]"), ("当日负性情绪→当夜 lnRMSSD", "−0.030 [−0.047, −0.014]"), ("日内间接效应 a×b", "0.003 [0.001, 0.005]")]:
    chk(f"DSEM: {DS[k]}", DA, sh_, [ds.loc[k, "median"], ds.loc[k, "lo95"], ds.loc[k, "hi95"]], out_("s38_dsem.csv"))
chk("Between-person SD of the two focal slopes", DA, "0.014, 0.016", [ds.loc["sd(a) 个体间", "median"], ds.loc["sd(b) 个体间", "median"]], out_("s38_dsem.csv"))
chk("Bursts; divergent transitions", DA, "2,172; 0", [dg["n_person_wave"], dg["divergences"]], out_("s38_dsem_diagnostics.json"))
chk("Largest R-hat, structural parameters", "Methods (Statistical analysis)", "1.01", [dg["max_rhat_hyper"]], out_("s38_dsem_diagnostics.json"), "Manuscript: 'below 1.01'.", lt=True)
chk("Multilevel: b, SE, person-nights", DA, "−0.021, 0.002, 14,975", [bm.est[0], bm.se[0], bm.n[0]], out_("s38_burst_multilevel.csv"))
chk("Multilevel: b, SE", DA, "−0.130, 0.032", [bm.est[1], bm.se[1]], out_("s38_burst_multilevel.csv"))

CO = "Results (Consumer ring); Fig. 4c,d"
chk("Ring vs H10: nights, concordance, bias", CO, "148, 0.978, 0.008", [ag.n[0], ag.ccc[0], ag.bias[0]], out_("s37_agreement.csv"))
d_ = np.log(cc.ring_night_rmssd_ms) - np.log(cc.h10_night_rmssd_ms)
chk("Ring vs H10: limits of agreement", CO, "−0.16, 0.18", [d_.mean() - 1.96 * d_.std(), d_.mean() + 1.96 * d_.std()], RC, "Sample SD of the differences.")
chk("Ring vs H10: nights, participants", CO, "148, 37", [len(cc), cc.pid.nunique()], RC)
chk("Ring vs H10: MAPE; proportional slope", CO, "6.9; −0.072", [MAPE, ag.prop_slope[0]], RC)
chk("Ring burst vs laboratory: r, concordance, bias", CO, "0.80, 0.78, −0.09", [ag.r[1], ag.ccc[1], ag.bias[1]], out_("s37_agreement.csv"))
chk("Ring burst vs laboratory: limits of agreement", CO, "−0.70, 0.51", [d_lab.mean() - 1.96 * d_lab.std(), d_lab.mean() + 1.96 * d_lab.std()], RC, "Sample SD of the differences.")
chk("Ring within-person r; task concordance; person-waves", CO, "0.61; 0.69; 1,455", [RINGW, ag.ccc[2], ag.n[1]], RC)

PR = "Results (Temporal profiles); Fig. 5"
pct = lambda c: nc[c] / len(A) * 100
lf = lpa[lpa.indicators == "方案五指标"].set_index("k")
chk("Students in the profile analysis; entropy; smallest class, %", PR, "507; 0.77; 14", [len(A), lf.loc[5, "entropy"], lf.loc[5, "smallest"] * 100], out_("s36_lpa_fit.csv"))
for lab_, sh_ in [("P1", "15.6; −0.68, −0.73"), ("P2", "14.2; 0.10, −0.59"), ("P3", "30.4; −0.44, −0.20"), ("P4", "25.6; 0.41, 0.56")]:
    c = order[int(lab_[1]) - 1]
    chk(f"{lab_} share, %; crisis T0, T3", PR, sh_, [pct(c), pm.loc[c, ("mean", 0)], pm.loc[c, ("mean", 3)]], RC)
c5 = order[4]
chk("P5 share, %; crisis T0, T2, T3", PR, "14.2; 0.48, 2.11, 1.57", [pct(c5), pm.loc[c5, ("mean", 0)], pm.loc[c5, ("mean", 2)], pm.loc[c5, ("mean", 3)]], RC)
bk = bc.set_index("distal")
for k, nm, sh_ in [("W1", "appraisal", "−0.60 [−0.98, −0.22]"), ("W2", "emotion regulation", "−1.18 [−1.52, −0.86]"), ("X0", "baseline stress", "0.65 [0.33, 0.96]"), ("M0", "baseline lnRMSSD", "−0.31 [−0.66, 0.05]"), ("ctq", "childhood trauma", "0.28 [−0.10, 0.66]")]:
    chk(f"P5 − P1, {nm}", PR, sh_, [bk.loc[k, "diff"], bk.loc[k, "lo95"], bk.loc[k, "hi95"]], out_("s36_bch_contrast_best.csv"))
lg = res("s36_lcga_fit.csv").set_index("k")
chk("Growth-based BIC, 3 / 2 / 4 classes; smallest, %", PR, "5,318.6, 5,393.6, 5,337.8; 7", [lg.loc[3, "bic"], lg.loc[2, "bic"], lg.loc[4, "bic"], lg.loc[3, "smallest"] * 100], out_("s36_lcga_fit.csv"))
c3s = c3.sort_values(ascending=False) / len(A) * 100
chk("Three profiles: entropy; shares, %", PR, "0.74; 41.8, 41.4, 16.8", [lf.loc[3, "entropy"]] + list(c3s), RC)
rk = rf.set_index(["model", "k"])
chk("RI-LTA BIC, five states vs no intercept", PR, "14,322.2, 14,425.5", [rk.loc[("RI-LTA", 5), "bic"], rk.loc[("LTA", 5), "bic"]], out_("s36_rilta_fit.csv"))
chk("RI-LTA: entropy; smallest state, %; BIC for 4 and 6", PR, "0.69; 8; 14,334.6, 14,345.2", [rk.loc[("RI-LTA", 5), "entropy"], rk.loc[("RI-LTA", 5), "smallest_state"] * 100, rk.loc[("RI-LTA", 4), "bic"], rk.loc[("RI-LTA", 6), "bic"]], out_("s36_rilta_fit.csv"))
rm = res("s36_rilta_means_best.csv").set_index(["state", "indicator"])
chk("S1 PHQ-9 z; ideation probability", PR, "−0.77; 0.002", [rm.loc[(1, "phq9_total"), "mean"], rm.loc[(1, "cssrs_intensity"), "mean"]], out_("s36_rilta_means_best.csv"))
chk("S5 PHQ-9 z; ideation probability", PR, "2.40; 0.66", [rm.loc[(5, "phq9_total"), "mean"], rm.loc[(5, "cssrs_intensity"), "mean"]], out_("s36_rilta_means_best.csv"))
rt = res("s36_rilta_transitions_best.csv").set_index(["from_state", "to_state"])
chk("S1 to S1 [95% CI]", PR, "0.69 [0.62, 0.73]", [rt.loc[(1, 1), "prob"], rt.loc[(1, 1), "lo95"], rt.loc[(1, 1), "hi95"]], out_("s36_rilta_transitions_best.csv"))
chk("S1 to S5; S5 to S5", PR, "0.015; 0.38", [rt.loc[(1, 5), "prob"], rt.loc[(5, 5), "prob"]], out_("s36_rilta_transitions_best.csv"))
chk("S5 prevalence, %, T0 to T3", PR, "2.8, 7.9, 10.1, 9.4", [rs["prevalence_by_wave_best"][w][4] * 100 for w in range(4)], out_("s36_rilta_summary.json"))
chk("Students in one state at all waves, %", PR, "19.7", [(ST[[f"state_w{w}" for w in range(4)]].nunique(axis=1) == 1).mean() * 100], RC)
chk("P5 waves in the two most severe states, %", PR, "59", [sev[c5] * 100], RC)
chk("P1 waves in the two most severe states, %", PR, "1", [sev[order[0]] * 100], RC, "Manuscript: 'less than 1%'.", lt=True)

PS = "Results (Passive sensing); Fig. 6"
pk = pr.set_index(["population", "model"])
A_, N_ = "全部转移", "新发（当前未达状态）"
chk("Self-report AUC, all transitions; n", PS, "0.808; 1,538", [pk.loc[(A_, "自评"), "auc"], pk.loc[(A_, "自评"), "n"]], out_("s37_prediction.csv"))
for m_, sh_ in [("自评+被动", "0.017 [0.007, 0.026]"), ("自评+生理", "0.000 [−0.004, 0.005]")]:
    chk(f"AUC change, {MOD[m_]}", PS, sh_, [pk.loc[(A_, m_), c] for c in ["delta_auc", "delta_lo", "delta_hi"]], out_("s37_prediction.csv"))
chk("AUC change, momentary assessment", PS, "−0.001", [pk.loc[(A_, "自评+EMA"), "delta_auc"]], out_("s37_prediction.csv"))
chk("New onset: AUC; n", PS, "0.685; 1,005", [pk.loc[(N_, "自评"), "auc"], pk.loc[(N_, "自评"), "n"]], out_("s37_prediction.csv"))
chk("New onset: AUC change, passive sensing", PS, "0.022 [−0.003, 0.047]", [pk.loc[(N_, "自评+被动"), c] for c in ["delta_auc", "delta_lo", "delta_hi"]], out_("s37_prediction.csv"))
chk("New onset: AUC change, all features", PS, "0.012 [−0.014, 0.037]", [pk.loc[(N_, "自评+生理+数字"), c] for c in ["delta_auc", "delta_lo", "delta_hi"]], out_("s37_prediction.csv"))
chk("Calibration, all features: slope, intercept", PS, "0.93, 0.003", [pk.loc[(A_, "自评+生理+数字"), "calib_slope"], pk.loc[(A_, "自评+生理+数字"), "calib_intercept"]], out_("s37_prediction.csv"))
chk("Calibration slope, new onset", PS, "0.79", [pk.loc[(N_, "自评+生理+数字"), "calib_slope"]], out_("s37_prediction.csv"))
chk("Brier, self-report to all features", PS, "0.168, 0.160", [pk.loc[(A_, "自评"), "brier"], pk.loc[(A_, "自评+生理+数字"), "brier"]], out_("s37_prediction.csv"))
nk = nb[nb.population == A_].set_index(["model", "threshold"]).net_benefit
chk("Net benefit at 0.3, self-report to all features", PS, "0.209, 0.216", [nk[("自评", 0.3)], nk[("自评+生理+数字", 0.3)]], out_("s37_net_benefit.csv"))
chk("Net benefit at 0.5, self-report to all features", PS, "0.133, 0.154", [nk[("自评", 0.5)], nk[("自评+生理+数字", 0.5)]], out_("s37_net_benefit.csv"))
sk = sh.set_index("feature").mean_abs_shap
chk("Shapley: PHQ-9, DASS-21, GAD-7", PS, "0.40, 0.33, 0.27", [sk["phq9_total"], sk["dass_stress"], sk["gad7_total"]], out_("s37_shap.csv"))
chk("Shapley: entropy, night screen, steps, home", PS, "0.21, 0.19, 0.15, 0.15", [sk["entropy"], sk["night_screen"], sk["steps"], sk["home"]], out_("s37_shap.csv"))

RERUN = "Re-run from record-level tables"
chk("Self-report AUC, all transitions, re-run", PS, "0.808", [AUC_SELF], RERUN)
chk("Calibration slope, intercept, all features, re-run", PS, "0.93, 0.003", [CAL_SLOPE, CAL_INT], RERUN)
chk("Brier, self-report to all features, re-run", PS, "0.168, 0.160", [BRIER_SELF, BRIER_ALL], RERUN)
chk("OR [95% CI]: passive ideation, stress, re-run", "Results (Stress precedes)", "2.46 [1.37, 4.40]", [math.exp(gee.params["X_w"]), math.exp(GEE_CI[0]), math.exp(GEE_CI[1])], RERUN)
chk("Daily multilevel person-nights, person-days, recounted", DA, "14,975", [len(DAY_A)], RC, f"Next-day negative affect model: {len(DAY_B):,} person-days. Its lag is row-based; {int((DAY_B.gap > 1).sum()):,} of these rows span more than one calendar day.")

SA = "Results (Risk events)"
NOSRC = "From safety_events as confirmed by the authors; source documents not inspected."
chk("Events; students; %", SA, "540; 328; 54.7", [len(SAF), SAF.pid.nunique(), SAF.pid.nunique() / 6], RC, NOSRC)
chk("By source: PHQ-9 item 9, momentary, C-SSRS", SA, "400, 78, 62", [(SAF.trigger_source == k).sum() for k in ["phq9_item9", "ema_ideation", "cssrs"]], RC, NOSRC)
chk("By level: low, moderate, high", SA, "473, 60, 7", [(SAF.risk_level == k).sum() for k in ["low", "moderate", "high"]], RC, NOSRC)
chk("External referrals; median response, h", SA, "7; 11.8", [SAF.referred_external.sum(), SAF.response_hours.median()], RC, NOSRC)
chk("Receiving treatment at T3, %", SA, "11.2", [TX.mean() * 100], RC)
VER.append({"Quantity": "PHQ-9 item 9 triggers against logged events", "Manuscript location": SA + "; Methods (Safety protocol)", "Manuscript": "400 PHQ-9 events", "Recomputed": f"{len(PHQ_ALL)} triggers; {len(PHQ_OTHER)} in a wave with another logged event", "Basis": RC, "Match": "Yes", "Note": "Every person-wave with a trigger has at least one logged event."})
VER.append({"Quantity": "Safety contact records against source documents", "Manuscript location": "Methods (Safety protocol)", "Manuscript": "Licensed clinicians contacted each participant within 24 h", "Recomputed": "", "Basis": "Not recomputed", "Match": "Source documents not inspected", "Note": "The authors confirm the monitoring rule and the safety_events records. Source documents for these contacts were not inspected in this check."})

T1 = "Table 1"
for label, f in [("Total", T0), ("Core", T0[T0.tier == "core"]), ("Extended", T0[T0.tier == "extended"])]:
    vals = table1(f)
    shown = dict(Total=["339 (56.5%)", "20.8 (1.5)", "369 (61.5%)", "21.9 (2.9)", "2.88 (0.86)", "3.20 (0.86)", "2.97 (0.83)", "7.44 (5.03)", "6.58 (4.45)", "170 (28.3%)", "5.13 (2.35)", "40.88 (9.46)", "3.67 (0.50)", "73.6 (8.9)"], Core=["171 (57.0%)", "20.8 (1.5)", "189 (63.0%)", "21.9 (2.9)", "2.88 (0.87)", "3.23 (0.85)", "2.99 (0.83)", "7.27 (4.81)", "6.64 (4.34)", "78 (26.0%)", "5.05 (2.44)", "41.06 (9.30)", "3.64 (0.47)", "74.3 (9.0)"], Extended=["168 (56.0%)", "20.8 (1.5)", "180 (60.0%)", "21.8 (2.8)", "2.89 (0.85)", "3.18 (0.86)", "2.95 (0.83)", "7.62 (5.25)", "6.52 (4.57)", "92 (30.7%)", "5.21 (2.25)", "40.69 (9.62)", "3.69 (0.52)", "73.0 (8.7)"])[label]
    for rw, a, b in zip(rows, shown, vals):
        VER.append({"Quantity": f"{rw}: {label}", "Manuscript location": T1, "Manuscript": a, "Recomputed": b, "Basis": RC, "Match": "Yes" if a == b else "No", "Note": ""})

for q, sh_, why in [("Monte Carlo power at lags 0.15, 0.11, 0.05", "0.85, 0.55, 0.03", "Power simulation outputs are not among the stored analysis outputs."), ("Interaction detection in design replications", "70% at 0.13; 42% at 0.09", "Power simulation outputs are not among the stored analysis outputs.")]:
    VER.append({"Quantity": q, "Manuscript location": "Methods (Sample size)", "Manuscript": sh_, "Recomputed": "", "Basis": "Not recomputed", "Match": "Not verifiable", "Note": why})
ver = pd.DataFrame(VER)
save(ver, "08_verification/manuscript_vs_recomputed.csv", "Manuscript values compared with the record-level tables and analysis outputs", "All", {"Quantity": "Reported quantity.", "Manuscript location": "Section, table or figure.", "Manuscript": "Value as printed in the manuscript.", "Recomputed": "Value obtained here, one decimal beyond the manuscript's precision.", "Basis": "Recomputed or re-run from the tables, re-fitted in lavaan, or read from the stored analysis output.", "Match": "Yes; Yes (rounding): equal within half a unit of the printed precision but half-up rounding gives another digit; No; Source documents not inspected; Not verifiable.", "Note": "Explanation where needed."})
dm = [("Screening and enrolment", "Results (Retention); Fig. 1c", "01_participants/enrolment_flow.csv", "screening", "eligible, consented, enrolled"),
      ("Baseline characteristics", "Table 1", "01_participants/participant_characteristics.csv", "participants + scale_scores + ecg_lab", "sex, age, grade, bmi, ctq_total, academic_stress, employment_anxiety, interpersonal_stress, phq9_total, gad7_total, psqi_total, ln_rmssd, rest_hr_bpm"),
      ("Retention and dropout", "Results (Retention)", "01_participants/retention_by_wave.csv; dropout_model.csv", "participants + scale_scores", "last_wave_attended, wave"),
      ("Data quality", "Results (Retention); Fig. 1e", "02_descriptives_and_quality/data_quality.csv", "ecg_lab, eeg_lab, wearable_nightly, passive_daily, ema_prompts", "corrected_beats_pct, clean_seconds, valid_night, steps, answered"),
      ("Reactivity and EEG spectra", "Results (Retention)", "02_descriptives_and_quality/laboratory_manipulation_checks.csv; eeg_alpha_by_channel.csv", "ecg_lab, eeg_lab, eeg_channel_level", "ln_rmssd, ln_rmssd_task, ln_rmssd_recovery, alpha_periodic_log, aperiodic_exponent"),
      ("Reliability, ICC, invariance, method variance", "Results (Retention); Methods", "02_descriptives_and_quality/", "scale_items, scale_scores, clinician_assessment, ecg_lab, eeg_lab", "item columns, composites"),
      ("RI-CLPM and traditional model", "Results (Separating; Stress precedes); Table 2; Fig. 2", "03_within_person_panel/", "scale_scores + clinician_assessment + ecg_lab + participants", "academic_stress, employment_anxiety, interpersonal_stress, phq9_total, gad7_total, cgi_s, cssrs_intensity, ln_rmssd; auxiliaries ctq_total, social_support, family_ses, age, psqi_total"),
      ("Moderated mediation", "Results (Appraisal moderates); Fig. 3b,c", "03_within_person_panel/moderation_tests.csv; moderation_simple_slopes.csv", "scale_scores", "sam_challenge, stress_mindset, erq_reappraisal, ders16_total"),
      ("Alternative outcomes, mediators and specifications", "Results (Indirect effect); Fig. 3a", "03_within_person_panel/indirect_effect_robustness.csv", "scale_scores, clinician_assessment, ecg_lab, eeg_lab, wearable_nightly, academic_job_logs, participants", "dass_stress, hf_power_ms2, ln_night_rmssd, alpha_periodic_log, faa_f4_f3, ln_rmssd_recovery, therapy_use, days_to_next_final, school, cohort"),
      ("Categorical outcomes", "Results (Stress precedes)", "03_within_person_panel/secondary_categorical_outcomes.csv", "scale_scores, clinician_assessment, ecg_lab", "phq9_total, gad7_total, cssrs_passive_ideation, cssrs_active_ideation"),
      ("Daily bursts", "Results (Daily); Fig. 4a,b,e", "04_daily_bursts_and_devices/dsem_parameters.csv; daily_multilevel_models.csv", "ema_prompts, wearable_nightly", "stress, neg_affect, ln_night_rmssd, valid_night"),
      ("Device agreement", "Results (Consumer ring); Fig. 4c,d", "04_daily_bursts_and_devices/device_agreement.csv; device_tracking.csv", "overnight_device_agreement, wearable_nightly, ecg_lab", "h10_night_rmssd_ms, ring_night_rmssd_ms, ln_night_rmssd, ln_rmssd, ln_rmssd_task"),
      ("Temporal profiles", "Results (Temporal profiles); Fig. 5a–c", "05_profiles_and_states/", "scale_scores, clinician_assessment, ecg_lab, ema_prompts, participants", "composites, neg_affect, ln_rmssd, ctq_total"),
      ("Latent transitions", "Results (Temporal profiles); Fig. 5d–f", "05_profiles_and_states/rilta_*.csv", "scale_scores, clinician_assessment", "phq9_total, gad7_total, cgi_s, cssrs_intensity"),
      ("Prediction", "Results (Passive sensing); Fig. 6", "06_prediction/", "scale_scores, participants, ecg_lab, wearable_nightly, passive_daily, ema_prompts, safety_events", "phq9_total, gad7_total, screen_minutes, night_screen_minutes, steps, home_hours, location_entropy, referred_external"),
      ("Safety monitoring", "Results (Risk events); Methods (Safety protocol)", "07_safety/safety_events_summary.csv", "safety_events, scale_scores, scale_items", "trigger_source, risk_level, referred_external, response_hours, therapy_use, phq_9")]
save(pd.DataFrame(dm, columns=["Manuscript result", "Manuscript section", "Summary file", "Record-level source table", "Key variables"]), "08_verification/manuscript_to_data_map.csv", "For each main result: section, summary file, record-level tables and key variables", "All", {"Manuscript result": "Result.", "Manuscript section": "Section, table or figure.", "Summary file": "File in this package.", "Record-level source table": "Record-level tables behind the result.", "Key variables": "Main variables used."})

NONE = "Study schematic, no data"
PERSON = "Individual pairs are in the journal Source Data file, not in this package; summary statistics are in the listed file."
fm_ = [("Fig. 1a", "Timeline of the four waves", NONE, ""), ("Fig. 1b", "Measures and schedule", NONE, ""), ("Fig. 1c", "Flow from screening to the final wave", "01_participants/enrolment_flow.csv; 01_participants/retention_by_wave.csv", ""), ("Fig. 1d", "Hypothesized pathway", NONE, ""), ("Fig. 1e", "Valid sessions, nights, person-days and prompts", "02_descriptives_and_quality/data_quality.csv", ""),
       ("Fig. 2a", "Standardized lags, main RI-CLPM", "03_within_person_panel/riclpm_nine_paths.csv", ""), ("Fig. 2b", "All nine lags with 95% CIs", "03_within_person_panel/riclpm_nine_paths.csv", ""), ("Fig. 2c", "Traditional model vs RI-CLPM", "03_within_person_panel/lagged_paths_by_model.csv", "Standardized 95% CIs are in the journal Source Data file (sheet Fig 2c)."), ("Fig. 2d", "Between-person and within-person correlations", "03_within_person_panel/between_within_correlations.csv", ""),
       ("Fig. 3a", "Indirect effect across specifications", "03_within_person_panel/indirect_effect_robustness.csv", ""), ("Fig. 3b", "Simple slopes by appraisal", "03_within_person_panel/moderation_simple_slopes.csv; 03_within_person_panel/moderation_tests.csv", "Wald 95% CIs derived from each slope and its exact P are in the journal Source Data file (sheet Fig 3b,c)."), ("Fig. 3c", "Simple slopes by emotion regulation", "03_within_person_panel/moderation_simple_slopes.csv; 03_within_person_panel/moderation_tests.csv", "Wald 95% CIs derived from each slope and its exact P are in the journal Source Data file (sheet Fig 3b,c)."),
       ("Fig. 4a", "Daily stress and that night's lnRMSSD", "04_daily_bursts_and_devices/daily_coupling_bins.csv; 04_daily_bursts_and_devices/daily_multilevel_models.csv", ""), ("Fig. 4b", "Nocturnal lnRMSSD and next-day negative affect", "04_daily_bursts_and_devices/daily_coupling_bins.csv; 04_daily_bursts_and_devices/daily_multilevel_models.csv", ""), ("Fig. 4c", "Bland–Altman, ring vs Polar H10", "04_daily_bursts_and_devices/device_agreement.csv", PERSON), ("Fig. 4d", "Bland–Altman, ring vs laboratory", "04_daily_bursts_and_devices/device_agreement.csv", PERSON), ("Fig. 4e", "Daily paths from the DSEM", "04_daily_bursts_and_devices/dsem_parameters.csv", ""),
       ("Fig. 5a", "BIC of the profile solutions", "05_profiles_and_states/lpa_fit.csv", ""), ("Fig. 5b", "Crisis score by profile", "05_profiles_and_states/profile_crisis_by_wave.csv", ""), ("Fig. 5c", "Bias-corrected means by profile", "05_profiles_and_states/bch_means.csv; 05_profiles_and_states/bch_contrasts.csv", ""), ("Fig. 5d", "BIC of the transition models", "05_profiles_and_states/rilta_fit.csv", ""), ("Fig. 5e", "Transition probabilities", "05_profiles_and_states/rilta_transitions.csv", ""), ("Fig. 5f", "State prevalence by wave", "05_profiles_and_states/rilta_prevalence.csv", ""),
       ("Fig. 6a", "Change in AUC", "06_prediction/prediction_performance.csv", ""), ("Fig. 6b", "Calibration in deciles", "06_prediction/calibration_deciles.csv", "Recomputed with the stored analysis design; the slope and intercept reproduce the stored output."), ("Fig. 6c", "Net benefit", "06_prediction/net_benefit.csv", ""), ("Fig. 6d", "Mean absolute Shapley values", "06_prediction/feature_contributions.csv", ""),
       ("Table 1", "Baseline characteristics", "01_participants/participant_characteristics.csv", ""), ("Table 2", "Confirmatory tests", "03_within_person_panel/confirmatory_tests.csv", "")]
for _, _, f_, _ in fm_:
    for part in f_.split("; "):
        if part != NONE and not (OUT / part).exists():
            raise FileNotFoundError(part)
save(pd.DataFrame(fm_, columns=["Panel", "Content", "Data file", "Note"]), "08_verification/figure_to_data_map.csv", "For each figure panel and table: the file holding its data", "All figures and tables", {"Panel": "Figure panel or table.", "Content": "What it shows.", "Data file": "File in this package.", "Note": "Explanation where needed."})

# ======== 09_codebook ========
WAVE ="Wave (0 = T0, 1 = T1 at about 2 months, 2 = T2 at about 4 months, 3 = T3 at about 6 months)."
TIER = "core = core sample; extended = extended sample."
GR = "Y1–Y4 = undergraduate year; PG = postgraduate."
DESC = {"pid": "Pseudonymous participant identifier.", "wave": WAVE, "week": "Week within the wave.", "deadlines_due": "Deadlines due.", "exams_scheduled": "Examinations scheduled.", "hours_before_deadline": "Hours between submission and deadline.", "job_applications": "Job applications submitted.", "lms_logins": "Learning-platform logins.",
        "cgi_s": "Clinical Global Impression severity, 1–7 (clinician, blind to device data).", "cssrs_passive_ideation": "C-SSRS passive ideation.", "cssrs_active_ideation": "C-SSRS active ideation.", "cssrs_intensity": "C-SSRS ideation intensity, 0–25.", "cssrs_behavior": "C-SSRS behaviour.", "rater_id": "Clinician rater identifier.",
        "tier": TIER, "sampling_rate_hz": "Sampling rate (Hz).", "test_hour": "Time of test (hour of day).", "rest_hr_bpm": "Resting heart rate (beats/min).", "rmssd_ms": "Resting RMSSD (ms).", "ln_rmssd": "Resting lnRMSSD.", "hf_power_ms2": "High-frequency power (ms²).", "ln_rmssd_task": "lnRMSSD during the task.", "ln_rmssd_recovery": "lnRMSSD during recovery.", "corrected_beats_pct": "Corrected beats, % (above 5 set to missing).", "resp_rate_bpm": "Respiration rate (breaths/min).", "sync_source": "Sync signal aligning ECG with the task log (marker, offset or none).",
        "condition": "eyes_closed / eyes_open.", "channel": "Electrode name, including recovered FCz.", "alpha_periodic_log": "Aperiodic-adjusted alpha peak height (log).", "alpha_total_log": "Unadjusted alpha power (log).", "aperiodic_exponent": "Aperiodic exponent.", "aperiodic_offset": "Aperiodic offset.", "iapf_hz": "Individual alpha peak frequency (Hz).", "alpha_bandwidth_hz": "Alpha peak bandwidth (Hz).", "fit_r2": "Goodness of fit of the spectral parameterization.", "impedance_kohm": "Electrode impedance (kΩ; blank for FCz).", "interpolated": "Bad channel interpolated (0/1).",
        "clean_seconds": "Clean eyes-closed data, s (below 180 set to missing).", "n_channels_recorded": "Channels recorded.", "n_channels_interpolated": "Channels interpolated.", "data_loss_pct": "Eyes-closed data lost, %.", "faa_f4_f3": "Frontal alpha asymmetry, F4 − F3.", "alpha_periodic_log_eo": "Posterior alpha peak height, eyes open (log).", "median_impedance_kohm": "Median electrode impedance (kΩ).", "n_ica_removed": "Independent components removed.",
        "burst_day": "Day within the burst (1–14).", "prompt": "Prompt number within the day.", "date": "Calendar date.", "answered": "Prompt answered (0/1).", "stress": "Momentary stress, 1–7.", "neg_affect": "Negative affect, 1–7.", "pos_affect": "Positive affect, 1–7.", "ideation_item": "Momentary ideation item, 0–6.",
        "h10_night_rmssd_ms": "Polar H10 RMSSD on the same night (ms).", "ring_night_rmssd_ms": "Ring RMSSD on the same night (ms).", "h10_corrected_beats_pct": "H10 corrected beats, %.",
        "school": "University code.", "cohort": "Start cohort.", "sex": "F / M.", "grade": GR, "age": "Age in years.", "bmi": "Body mass index.", "smoker": "Smoker (0/1).", "family_ses": "Family socioeconomic status, 1–5.", "ctq_total": "Childhood Trauma Questionnaire total (minimum 25).", "social_support": "Social support total.", "platform": "Phone platform; iOS provides no screen-time features.", "start_date": "Enrolment date.", "last_wave_attended": "Last wave completed.",
        "screen_minutes": "Screen time (min; blank on iOS or days without data).", "night_screen_minutes": "Night-time screen use (min).", "steps": "Steps.", "home_hours": "Hours at home.", "location_entropy": "Location entropy.", "exam_day": "Examination on that day (0/1).",
        "trigger_source": "Risk trigger: phq9_item9, cssrs or ema_ideation.", "risk_level": "Risk level: low, moderate or high.", "contacted": "Contacted (0/1).", "referred_external": "Referred outside the university (0/1).", "response_hours": "Response time (h).",
        "psqi_total": "Pittsburgh Sleep Quality Index total.", "caffeine_cups_day": "Caffeine, cups per day.", "exercise_hours_week": "Exercise, hours per week.", "visit_date": "Date of the laboratory or questionnaire visit.", "days_to_next_final": "Days to the next final examination (time-varying covariate).", "academic_stress": "Academic stress item mean, 1–5.", "employment_anxiety": "Employment anxiety item mean, 1–5.", "interpersonal_stress": "Interpersonal stress item mean, 1–5.", "phq9_total": "PHQ-9 total.", "gad7_total": "GAD-7 total.", "dass_stress": "DASS-21 stress subscale (×2).", "sam_challenge": "Stress appraisal, challenge item mean.", "stress_mindset": "Stress mindset item mean.", "erq_reappraisal": "ERQ reappraisal item mean.", "ders16_total": "DERS-16 total.", "therapy_use": "Receiving psychological treatment or referral, including self-referral (0/1).", "safety_contact": "Safety contact at or before this wave (0/1), recorded separately from treatment.",
        "screen_phq9": "PHQ-9 at screening.", "screen_gad7": "GAD-7 at screening.", "screen_cssrs_ideation": "C-SSRS ideation level at screening, 0–5.", "prior_attempt_12m": "Suicide attempt in the past 12 months (0/1).", "exclusion_medication": "Excluded for medication (0/1).", "exclusion_somatic": "Excluded for somatic disease (0/1).", "eligible": "Met the criteria (0/1).", "consented": "Consented (0/1).", "enrolled": "Enrolled (0/1).",
        "night_rmssd_ms": "Ring nocturnal RMSSD (ms).", "ln_night_rmssd": "Ring nocturnal lnRMSSD.", "night_hr_bpm": "Nocturnal heart rate (beats/min).", "sleep_minutes": "Sleep (min).", "valid_night": "Valid night (0/1)."}
ITEM = {"acad": "Academic stress", "emp": "Employment anxiety", "inter": "Interpersonal stress", "phq": "PHQ-9", "gad": "GAD-7", "dass": "DASS-21 stress", "sam": "Stress appraisal, challenge", "smm": "Stress mindset", "erq": "ERQ reappraisal", "ders": "DERS-16"}
UNIT = {"screening": ("Screened individual", "Results (Retention); Fig. 1c"), "participants": ("Enrolled participant", "Table 1; Methods"), "scale_items": ("Participant × wave", "Methods; Results (Retention)"), "scale_scores": ("Participant × wave", "Table 1; panel models"), "clinician_assessment": ("Participant × wave", "Panel models; Methods"), "ecg_lab": ("Participant × laboratory ECG session", "Table 1; panel models; Fig. 1e"), "eeg_lab": ("Core participant × EEG session", "Results (Indirect effect); Fig. 1e"), "eeg_channel_level": ("EEG session × condition × channel", "Results (Retention)"), "wearable_nightly": ("Participant × wave × burst night", "Results (Daily; Indirect effect); Fig. 4"), "overnight_device_agreement": ("Night with simultaneous H10 and ring", "Results (Consumer ring); Fig. 4c"), "ema_prompts": ("Scheduled momentary prompt", "Results (Daily); Fig. 4"), "passive_daily": ("Participant × wave × burst day", "Results (Passive sensing); Fig. 6"), "academic_job_logs": ("Participant × wave × week", "Results (Indirect effect)"), "safety_events": ("Risk event", "Results (Risk events); prediction exclusions")}
crows, trows = [], []
for t in TABLES:
    d = T[t]
    trows.append({"table": t, "unit": UNIT[t][0], "rows": f"{len(d):,}", "n_variables": d.shape[1], "manuscript": UNIT[t][1], "included_in_package": "no (see README, section 6)"})
    for c in d.columns:
        m = re.fullmatch(r"([a-z]+)_(\d+)", c)
        desc = f"{ITEM[m[1]]} item {m[2]}." if m and m[1] in ITEM else DESC[c]
        s_ = d[c].dropna()
        if c == "pid":
            rng = f"{d[c].nunique():,} distinct"
        elif "date" in c:
            rng = "ISO date; range not shown"
        elif pd.api.types.is_numeric_dtype(s_) and s_.nunique() > 8:
            rng = f"{s_.min():g} – {s_.max():g}"
        elif pd.api.types.is_numeric_dtype(s_):
            rng = ", ".join(f"{v:g}" for v in sorted(s_.unique()))
        else:
            rng = ", ".join(sorted(s_.astype(str).unique())) if s_.nunique() <= 8 else f"{s_.nunique()} distinct"
        crows.append({"table": t, "variable": c, "description": desc, "non_missing": f"{s_.size:,}", "range_or_levels": rng})
trt = pd.DataFrame(trows)
crt = pd.DataFrame(crows)
save(trt, "09_codebook/record_level_tables.csv", "The 14 record-level tables: unit, rows, variables, manuscript sections", "All", {"table": "Record-level table.", "unit": "Unit of observation.", "rows": "Rows.", "n_variables": "Variables.", "manuscript": "Manuscript sections that use it.", "included_in_package": "Whether the table is in this package."})
save(crt, "09_codebook/codebook_record_level_tables.csv", "One row per variable of each record-level table", "All", {"table": "Record-level table.", "variable": "Variable.", "description": "Description.", "non_missing": "Non-missing values.", "range_or_levels": "Range or levels. Date ranges are not shown."})
cb_path = "09_codebook/codebook_summary_tables.csv"
INVENTORY["09_codebook"].append(("codebook_summary_tables.csv", "One row per column of every summary file", "All"))
self_desc = [{"file": cb_path, "column": c, "description": v} for c, v in {"file": "Summary file.", "column": "Column.", "description": "Description."}.items()]
pd.DataFrame(CODEBOOK + self_desc).to_csv(OUT / cb_path, index=False, encoding="utf-8-sig")


# ======== README ========
def md_table(head, body):
    rows_ = [head] + body
    w = [max(len(str(r[i])) for r in rows_) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([fmt(head), "| " + " | ".join("-" * x for x in w) + " |"] + [fmt(r) for r in body])


vc = ver.Match.value_counts()
overview = md_table(["Item", "Value"], [
    ["Design", "Four-wave panel at months 0, 2, 4 and 6 with two start cohorts in three universities"],
    ["Screened / eligible", "2,000 screened; 1,807 eligible; 1,132 eligible and consented"],
    ["Enrolled", "600 (core 300, extended 300), sampled with stratification on year of study, sex and stress level"],
    ["Retained", "T0, T1, T2, T3: 600 / 553 / 521 / 498"],
    ["Laboratory ECG", "Lead II, 1,000 Hz; core sample at all waves, extended sample at T0 and T3 (1,625 sessions)"],
    ["Laboratory EEG", "Core sample; 32 active electrodes, 1,000 Hz, eyes open and closed (1,072 sessions)"],
    ["Bursts", "14 days after each wave: ring, three momentary prompts per day, smartphone sensing (2,172 bursts)"],
    ["Clinician ratings", "CGI-S and C-SSRS at each wave, blind to device and smartphone data"],
    ["Primary model", "Random intercept cross-lagged panel model with auxiliary variables; n = 600"],
    ["Safety", "Three-source risk monitoring with clinician contact within 24 h (540 events)"]])
sections = []
for folder in sorted(INVENTORY):
    sections.append(f"### {folder}\n\n" + md_table(["File", "Content", "Manuscript"], [[f"`{a}`", b, c] for a, b, c in INVENTORY[folder]]))
headline = md_table(["Manuscript value", "File"], [
    ["2,000 screened, 1,807 eligible, 1,132 consented, 600 enrolled", "`01_participants/enrolment_flow.csv`"],
    ["Retention 553 / 521 / 498", "`01_participants/retention_by_wave.csv`"],
    ["Table 1", "`01_participants/participant_characteristics.csv`"],
    ["95.5% of ECG and 96.5% of EEG sessions valid; ring 68.9%; prompts 64.1%", "`02_descriptives_and_quality/data_quality.csv`"],
    ["RI-CLPM fit: χ²(51) = 56.4, BIC 22,850.3", "`03_within_person_panel/model_fit.csv`"],
    ["Stress to lnRMSSD β = −0.14; lnRMSSD to crisis β = −0.11", "`03_within_person_panel/riclpm_nine_paths.csv`"],
    ["Indirect effect 0.025 [0.005, 0.054]; Holm-adjusted P (Table 2)", "`03_within_person_panel/confirmatory_tests.csv`"],
    ["Indirect effect across specifications (Fig. 3a)", "`03_within_person_panel/indirect_effect_robustness.csv`"],
    ["Appraisal interaction 0.130; IMM −0.018 [−0.037, −0.003]", "`03_within_person_panel/moderation_tests.csv`"],
    ["Daily paths −0.100 and −0.028 (Fig. 4e)", "`04_daily_bursts_and_devices/dsem_parameters.csv`"],
    ["Ring vs H10 concordance 0.978 on 148 nights", "`04_daily_bursts_and_devices/device_agreement.csv`"],
    ["Five profiles in 507 students; P5 − P1 contrasts", "`05_profiles_and_states/profile_composition.csv`, `bch_contrasts.csv`"],
    ["Five latent states; S5 prevalence 2.8% to 9.4%", "`05_profiles_and_states/rilta_fit.csv`, `rilta_prevalence.csv`"],
    ["Self-report AUC 0.808; passive sensing +0.017 [0.007, 0.026]", "`06_prediction/prediction_performance.csv`"],
    ["540 risk events in 328 students", "`07_safety/safety_events_summary.csv`"]])
readme = f"""# Data package: Within-person autonomic pathways link educational stress to subclinical crisis in university students

This package holds the summary-level data behind the numbers reported in the manuscript. It also documents the record-level tables from which these numbers were computed. Each summary file is described column by column in `09_codebook/codebook_summary_tables.csv`. Each record-level table is described variable by variable in `09_codebook/codebook_record_level_tables.csv`.

## 1. Study overview

{overview}

## 2. Conventions

- **Format:** comma-separated values, UTF-8 with byte-order mark (opens directly in Excel), one header row.

- **Number formatting:** cells such as `7.44 (5.03)` are M (SD). Cells such as `[0.005, 0.054]` are 95% intervals. Thousands separators (`1,625`) appear where the manuscript uses them. Many files keep more decimals than the manuscript prints, so that rounding can be checked.

- **Waves:** T0, T1, T2 and T3 are months 0, 2, 4 and 6. The record-level `wave` variable codes them 0 to 3.

- **Composites:** educational stress is the mean of the standardized academic stress, employment anxiety and interpersonal stress scales. Subclinical crisis is the mean of the standardized PHQ-9, GAD-7, CGI-S and C-SSRS ideation intensity scores. All are standardized on baseline values. lnRMSSD is resting laboratory lnRMSSD unless stated.

- **Estimates:** β is the standardized within-person lag averaged over waves. Indirect effects are unstandardized a × b with Monte Carlo 95% CIs from 20,000 draws. Daily paths are posterior medians with 95% credible intervals.

- **Profiles and states:** P1 to P5 are numbered from lowest to highest crisis at T3. S1 to S5 are ordered from least to most severe. Transition intervals from S4 and S5 are close to the [0, 1] bounds and are poorly determined.

## 3. File inventory

{chr(10).join(s + chr(10) for s in sections)}
## 4. Locating the headline results

{headline}

## 5. Verification

`08_verification/manuscript_vs_recomputed.csv` compares {len(ver)} manuscript values with this package's sources. The `Basis` column states how each value was obtained: recomputed or re-run from the record-level tables, re-fitted in lavaan, or read from the stored analysis output. Result: {vc.get('Yes', 0)} match, {vc.get('Yes (rounding)', 0)} match within rounding, {vc.get('No', 0)} differ, {vc.get('Source documents not inspected', 0)} could not be checked because source documents were not inspected, and {vc.get('Not verifiable', 0)} cannot be verified from the stored files. The `Note` column explains each exception.

`08_verification/figure_to_data_map.csv` names the data file for every figure panel and table. The Bland–Altman panels (Fig. 4c,d) plot individual pairs, which are in the Source Data file submitted with the manuscript; this package holds their summary statistics (`04_daily_bursts_and_devices/device_agreement.csv`).

## 6. Record-level data

The summary values were computed from the {len(TABLES)} record-level tables listed in `09_codebook/record_level_tables.csv`. These tables cover screening, participants, questionnaire items and scores, clinician ratings, laboratory ECG and EEG, nocturnal ring and chest-strap recordings, momentary prompts, smartphone sensing, platform logs and safety events. They are not included in this package.

Access follows the manuscript's Data availability statement: summary-level data are openly available at https://github.com/guoxi7618-blip/educational-stress-autonomic-pathway, and record-level tables, raw physiological recordings and restricted variables are available from the corresponding author on reasonable request. This package itself holds summary-level data only.

The record-level tables contain information that requires controlled access:

- Suicidal ideation and behaviour at item level (`scale_items`, `clinician_assessment`, `ema_prompts`, `screening`) and individual risk events and referrals (`safety_events`).
- Clinical and physiological records for each person (`scale_scores`, `ecg_lab`, `eeg_lab`, `eeg_channel_level`, `wearable_nightly`, `overnight_device_agreement`).
- Daily behaviour from smartphones and academic records (`passive_daily`, `academic_job_logs`), including dated observations.
- Demographics and institution code (`participants`, `screening`).

Person-level model outputs (profile and state assignments, safety alert lists) are treated the same way and are also not included.

Participant identifiers are pseudonymous. No names, student numbers, contact details, raw coordinates or message content are included in any table. Date ranges are not shown in the codebook.
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")
print(ver.Match.value_counts().to_string())
print(ver[~ver.Match.isin(["Yes"])][["Quantity", "Manuscript", "Recomputed", "Match"]].to_string())
