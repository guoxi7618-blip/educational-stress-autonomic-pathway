# Analyses the plan requires beyond main_models.R (framework 3 and 4.7). Usage: Rscript supplement_models.R <processed_dir> <out_dir> [table_prefix]
suppressPackageStartupMessages({library(lavaan); library(mclust); library(data.table); library(nlme)})
args <- commandArgs(TRUE); DIR <- args[1]; OUT <- args[2]; PREFIX <- ifelse(length(args) >= 3, args[3], "")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
rd <- function(f) fread(file.path(DIR, paste0(PREFIX, f, ".csv")))
P <- rd("participants"); S <- rd("scale_scores"); I <- rd("scale_items"); E <- rd("ecg_lab"); G <- rd("eeg_lab"); C <- rd("clinician_assessment"); W <- rd("wearable_nightly"); EMA <- rd("ema_prompts"); LOG <- rd("academic_job_logs")
res <- list()

# ---- wide data, same composites as main_models.R ----
L <- merge(S, C[, .(pid, wave, cgi_s, cssrs_intensity, cssrs_passive_ideation, cssrs_active_ideation)], by = c("pid", "wave"), all = TRUE)
L <- merge(L, E[, .(pid, wave, ln_rmssd, hf_power_ms2, ln_rmssd_task, ln_rmssd_recovery, test_hour, resp_rate_bpm)], by = c("pid", "wave"), all = TRUE)
L <- merge(L, G[, .(pid, wave, alpha_periodic_log, faa_f4_f3)], by = c("pid", "wave"), all = TRUE)
zb <- function(x, w) (x - mean(x[w == 0], na.rm = TRUE)) / sd(x[w == 0], na.rm = TRUE)
L[, `:=`(zac = zb(academic_stress, wave), zem = zb(employment_anxiety, wave), zin = zb(interpersonal_stress, wave), zph = zb(phq9_total, wave), zga = zb(gad7_total, wave), zcg = zb(cgi_s, wave), zcs = zb(cssrs_intensity, wave), zds = zb(dass_stress, wave))]
L[, X := rowMeans(.SD), .SDcols = c("zac", "zem", "zin")]; L[, Y := rowMeans(.SD), .SDcols = c("zph", "zga", "zcg", "zcs")]
L[, `:=`(M = zb(ln_rmssd, wave), A = zb(alpha_periodic_log, wave), HF = zb(log(hf_power_ms2), wave), FA = zb(faa_f4_f3, wave), REC = zb(ln_rmssd_recovery - ln_rmssd, wave), YS = rowMeans(cbind(zph, zga)), YC = rowMeans(cbind(zcg, zcs)), YD = zds, PSQ = zb(psqi_total, wave), DTF = as.numeric(scale(log1p(days_to_next_final))), TH = as.numeric(therapy_use))]
L[, PHY := rowMeans(cbind(M, A))]
wb <- W[valid_night == 1, .(n = .N, Wr = mean(ln_night_rmssd)), by = .(pid, wave)][n >= 5]
L <- merge(L, wb[, .(pid, wave, Wr)], by = c("pid", "wave"), all.x = TRUE); L[, R := zb(Wr, wave)]
lo <- LOG[, .(dl = sum(deadlines_due), ex = sum(exams_scheduled), ap = sum(job_applications), late = mean(hours_before_deadline)), by = .(pid, wave)]
L <- merge(L, lo, by = c("pid", "wave"), all.x = TRUE); L[, O := rowMeans(cbind(zb(dl, wave), zb(ex, wave), zb(ap, wave)))]
L <- merge(L, P[, .(pid, tier, sex, grade, ctq_total, age, bmi, smoker, cohort, school)], by = "pid")
L[, `:=`(female = as.numeric(sex == "F"), upper = as.numeric(grade %in% c("Y3", "Y4")), ctq = as.numeric(scale(ctq_total)))]
base <- L[wave == 0, .(pid, W1 = as.numeric(scale(rowMeans(cbind(scale(sam_challenge), scale(stress_mindset))))), W2 = as.numeric(scale(rowMeans(cbind(scale(erq_reappraisal), -scale(ders16_total))))))]
L <- merge(L, base, by = "pid", all.x = TRUE)
vars <- c("X", "M", "Y", "A", "HF", "FA", "REC", "YS", "YC", "YD", "PSQ", "DTF", "TH", "PHY", "R", "O", "zph", "zga", "zcg", "zcs")
D <- as.data.frame(dcast(L, pid + W1 + W2 + female + upper + ctq + tier + cohort ~ wave, value.var = vars, sep = ""))

ri_model <- function(vn = c("X", "M", "Y"), waves = 0:3, invariant = TRUE, ri = TRUE, extra = "") {
  s <- ""
  for (v in vn) {
    if (ri) s <- paste0(s, "RI", v, " =~ ", paste0("1*", v, waves, collapse = "+"), "\n")
    for (t in waves) s <- paste0(s, "w", v, t, " =~ 1*", v, t, "\n", v, t, " ~~ 0*", v, t, "\n")
    s <- paste0(s, paste0(v, waves, " ~ m", v, waves, "*1", collapse = "\n"), "\n")
  }
  for (t in waves[-1]) for (to in vn) {
    rhs <- sapply(vn, function(fr) paste0(if (invariant) paste0("B", fr, to, "*") else "", "w", fr, t - 1))
    s <- paste0(s, "w", to, t, " ~ ", paste(rhs, collapse = "+"), "\n")
  }
  for (pr in combn(vn, 2, simplify = FALSE)) {
    s <- paste0(s, "w", pr[1], waves[1], " ~~ w", pr[2], waves[1], "\n")
    for (t in waves[-1]) s <- paste0(s, "w", pr[1], t, " ~~ ", if (invariant) paste0("C", pr[1], pr[2], "*") else "", "w", pr[2], t, "\n")
    if (ri) s <- paste0(s, "RI", pr[1], " ~~ RI", pr[2], "\n")
  }
  for (v in vn) {
    s <- paste0(s, "w", v, waves[1], " ~~ w", v, waves[1], "\n")
    if (ri) { s <- paste0(s, "RI", v, " ~~ RI", v, "\n"); for (t in waves) s <- paste0(s, "RI", v, " ~~ 0*w", v, t, "\n"); for (o in setdiff(vn, v)) for (t in waves) s <- paste0(s, "RI", v, " ~~ 0*w", o, t, "\n") }
    for (t in waves[-1]) s <- paste0(s, "w", v, t, " ~~ ", if (invariant) paste0("R", v, "*") else "", "w", v, t, "\n")
  }
  paste0(s, extra)
}
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "aux_fiml.R")); AUXD <- aux_data(P, S)
mfit <- function(syn, data = D, aux = TRUE, extra_obs = character(0), ...) { if (aux) { data <- with_aux(data, AUXD); syn <- paste0(syn, aux_syntax(syn, extra_obs)) }; sem(syn, data, missing = "fiml", estimator = "MLR", meanstructure = TRUE, orthogonal = FALSE, auto.fix.first = FALSE, auto.var = FALSE, auto.cov.lv.x = FALSE, auto.cov.y = FALSE, se = "robust.mlr", ...) }
mc <- function(m, pa, pb, nsim = 20000) { set.seed(20261101); co <- coef(m); V <- vcov(m); i <- match(c(pa, pb), names(co)); dr <- MASS::mvrnorm(nsim, co[i], V[i, i]); p <- dr[, 1] * dr[, 2]; c(est = unname(co[i[1]] * co[i[2]]), lo = unname(quantile(p, .025)), hi = unname(quantile(p, .975)), p_two = 2 * min(mean(p > 0), mean(p < 0))) }
est <- function(m, lab) { pe <- parameterEstimates(m); r <- pe[pe$label == lab, ][1, ]; c(est = r$est, se = r$se, p = r$pvalue) }
stdpath <- function(m, labs, level = .95) {
  pe <- parameterEstimates(m); ss <- standardizedSolution(m, level = level)
  do.call(rbind, lapply(labs, function(l) { k <- paste(pe$lhs, pe$rhs)[pe$label == l & pe$op == "~"]; s <- ss[ss$op == "~" & paste(ss$lhs, ss$rhs) %in% k, ]; data.frame(label = l, std = mean(s$est.std), lo = mean(s$ci.lower), hi = mean(s$ci.upper)) }))
}
safely <- function(expr) tryCatch(expr, error = function(e) { message("skipped: ", conditionMessage(e)); NULL })
m0 <- mfit(ri_model())

# ---- 3.2 ICC and within-person correlations ----
Lw <- copy(L)[, .(pid, wave, X, M, Y, R)]
for (v in c("X", "M", "Y", "R")) Lw[, (v) := get(v) - mean(get(v), na.rm = TRUE), by = pid]
wc <- cor(Lw[, .(X, M, Y, R)], use = "pairwise"); bm <- L[, lapply(.SD, mean, na.rm = TRUE), by = pid, .SDcols = c("X", "M", "Y", "R")]; bc <- cor(bm[, .(X, M, Y, R)], use = "pairwise")
res$r_within <- round(wc, 3); res$r_between <- round(bc, 3)

# ---- 3.4 all nine paths, Orth labels, reverse paths against +-.07, Holm ----
labs <- c("BXX", "BMX", "BYX", "BXM", "BMM", "BYM", "BXY", "BMY", "BYY")
s95 <- stdpath(m0, labs); s90 <- stdpath(m0, labs, .90); pe <- parameterEstimates(m0)
T4 <- merge(s95, data.frame(label = labs, b = sapply(labs, function(l) est(m0, l)["est"]), se = sapply(labs, function(l) est(m0, l)["se"]), p = sapply(labs, function(l) est(m0, l)["p"])), by = "label")
T4 <- merge(T4, setnames(s90[, c("label", "lo", "hi")], c("lo", "hi"), c("lo90", "hi90")), by = "label")
orth <- function(v) ifelse(abs(v) < .03, "可忽略", ifelse(abs(v) < .07, "小", ifelse(abs(v) < .12, "中", "大")))
T4$orth <- orth(T4$std); T4$within_07 <- T4$lo90 > -.07 & T4$hi90 < .07
T4$path <- c(BXX = "X→X", BMX = "M→X", BYX = "Y→X", BXM = "X→M", BMM = "M→M", BYM = "Y→M", BXY = "X→Y", BMY = "M→Y", BYY = "Y→Y")[T4$label]
fwrite(T4, file.path(OUT, "s34_nine_paths.csv"))
ma <- mfit(ri_model(extra = paste0(paste0(sapply(1:3, function(t) paste0("wM", t, " ~ gXW1*PX", t - 1, "\n")), collapse = ""), "wM1 ~ dW1*W1\nwM2 ~ dW1*W1\nwM3 ~ dW1*W1\nPX0 ~~ wX0 + wM0 + wY0\nPX1 ~~ wX1 + wM1 + wY1\nPX2 ~~ wX2 + wM2 + wY2\nPX0 ~~ PX0\nPX1 ~~ PX1\nPX2 ~~ PX2\nPX0 ~~ PX1 + PX2\nPX1 ~~ PX2\nW1 ~~ W1\nW1 ~~ RIX + RIM + RIY + PX0 + PX1 + PX2\nW1 ~ 1\nPX0 ~ 1\nPX1 ~ 1\nPX2 ~ 1\n")), extra_obs = c("PX0", "PX1", "PX2", "W1"),
  data = { Lp <- copy(L); Lp[, `:=`(Xw = X - mean(X, na.rm = TRUE)), by = pid]; Lp[, PX := Xw * W1]; as.data.frame(dcast(Lp, pid + W1 ~ wave, value.var = c("X", "M", "Y", "PX"), sep = "")) })
mb <- mfit(ri_model(extra = paste0(paste0(sapply(1:3, function(t) paste0("wY", t, " ~ gMW2*PM", t - 1, "\n")), collapse = ""), "wY1 ~ dW2*W2\nwY2 ~ dW2*W2\nwY3 ~ dW2*W2\nPM0 ~~ wX0 + wM0 + wY0\nPM1 ~~ wX1 + wM1 + wY1\nPM2 ~~ wX2 + wM2 + wY2\nPM0 ~~ PM0\nPM1 ~~ PM1\nPM2 ~~ PM2\nPM0 ~~ PM1 + PM2\nPM1 ~~ PM2\nW2 ~~ W2\nW2 ~~ RIX + RIM + RIY + PM0 + PM1 + PM2\nW2 ~ 1\nPM0 ~ 1\nPM1 ~ 1\nPM2 ~ 1\n")), extra_obs = c("PM0", "PM1", "PM2", "W2"),
  data = { Lp <- copy(L); Lp[, `:=`(Mw = M - mean(M, na.rm = TRUE)), by = pid]; Lp[, PM := Mw * W2]; as.data.frame(dcast(Lp, pid + W2 ~ wave, value.var = c("X", "M", "Y", "PM"), sep = "")) })
ind <- mc(m0, "BXM", "BMY")
H <- data.frame(hypothesis = c("H1 X→M", "H2 M→Y", "H3 间接效应", "H4a 认知评价调节 a", "H4b 情绪调节调节 b"), estimate = c(est(m0, "BXM")["est"], est(m0, "BMY")["est"], ind["est"], est(ma, "gXW1")["est"], est(mb, "gMW2")["est"]), p = c(est(m0, "BXM")["p"], est(m0, "BMY")["p"], ind["p_two"], est(ma, "gXW1")["p"], est(mb, "gMW2")["p"]))
H$p_holm <- p.adjust(H$p, "holm"); H$support_a05 <- H$p_holm < .05; fwrite(H, file.path(OUT, "s35_hypotheses_holm.csv"))
res$ind_ci <- ind

# ---- 3.8 alternative outcomes and mediators, each swapped into the same model ----
rob <- list()
swap <- function(name, vn, data = D, pa, pb, aux = TRUE) { m <- safely(mfit(ri_model(vn), data, aux = aux)); if (is.null(m)) return(NULL); i <- mc(m, pa, pb); s <- stdpath(m, c(pa, pb)); fm <- fit_stats(m, c("cfi.robust", "rmsea.robust", "srmr")); data.frame(model = name, n = lavInspect(m, "ntotal"), a_std = s$std[1], a_p = est(m, pa)["p"], b_std = s$std[2], b_p = est(m, pb)["p"], ind = i["est"], lo = i["lo"], hi = i["hi"], cfi = fm[1], rmsea = fm[2], srmr = fm[3]) }
core <- D[D$tier == "core", ]
rob$main <- swap("主模型：心电 lnRMSSD，合成 Y", c("X", "M", "Y"), D, "BXM", "BMY")
rob$noaux <- swap("不含辅助变量的 FIML", c("X", "M", "Y"), D, "BXM", "BMY", aux = FALSE)
rob$selfY <- swap("Y 只用自评（PHQ、GAD）", c("X", "M", "YS"), D, "BXM", "BMYS")
rob$clinY <- swap("Y 只用咨询师（CGI、C-SSRS）", c("X", "M", "YC"), D, "BXM", "BMYC")
rob$dass <- swap("Y 用 DASS-21 压力", c("X", "M", "YD"), D, "BXM", "BMYD")
rob$hf <- swap("M 用 HF-HRV（次要）", c("X", "HF", "Y"), D, "BXHF", "BHFY")
rob$ring <- swap("M 用戒指夜间 RMSSD", c("X", "R", "Y"), D, "BXR", "BRY")
rob$alpha <- swap("M 用 alpha（核心样本，RQ2）", c("X", "A", "Y"), core, "BXA", "BAY")
rob$faa <- swap("M 用 FAA（核心样本，探索）", c("X", "FA", "Y"), core, "BXFA", "BFAY")
rob$comp <- swap("M 用心电+alpha 合成（核心样本）", c("X", "PHY", "Y"), core, "BXPHY", "BPHYY")
rob$rec <- swap("M 用任务后恢复（核心样本）", c("X", "REC", "Y"), core, "BXREC", "BRECY")
rob$obj <- swap("X 用客观负荷（日志）", c("O", "M", "Y"), D, "BOM", "BMY")
# latent outcome, as the plan defines it: a factor per wave with equal loadings over waves
lat <- paste0(sapply(0:3, function(t) paste0("Y", t, " =~ 1*zph", t, " + l2*zga", t, " + l3*zcg", t, " + l4*zcs", t, "\n")), collapse = "")
lat <- paste0(lat, paste0(sapply(0:3, function(t) paste0("zph", t, " ~~ zph", t, "\nzga", t, " ~~ zga", t, "\nzcg", t, " ~~ zcg", t, "\nzcs", t, " ~~ zcs", t, "\nzph", t, " ~ 1\nzga", t, " ~ 1\nzcg", t, " ~ 1\nzcs", t, " ~ 1\n")), collapse = ""))
lat_syn <- gsub("Y([0-3]) ~~ 0\\*Y[0-3]\n", "Y\\1 ~~ 0*Y\\1\n", paste0(ri_model(), lat))
lat_syn <- gsub("mY[0-3]\\*1", "0*1", lat_syn)
mlat <- safely(mfit(lat_syn, aux = FALSE)); if (!is.null(mlat)) { i <- mc(mlat, "BXM", "BMY"); s <- stdpath(mlat, c("BXM", "BMY")); fm <- fitMeasures(mlat, c("cfi.robust", "rmsea.robust", "srmr")); rob$latent <- data.frame(model = "Y 为四波潜变量（方案定义）", n = lavInspect(mlat, "ntotal"), a_std = s$std[1], a_p = est(mlat, "BXM")["p"], b_std = s$std[2], b_p = est(mlat, "BMY")["p"], ind = i["est"], lo = i["lo"], hi = i["hi"], cfi = fm[1], rmsea = fm[2], srmr = fm[3]) }
# time-varying covariates: within-person sleep, exam proximity and therapy are removed from X, M and Y wave by wave, then the same model is fitted
Ltv <- copy(L)
for (v in c("PSQ", "DTF", "TH")) Ltv[, (paste0(v, "w")) := get(v) - mean(get(v), na.rm = TRUE), by = pid]
for (v in c("X", "M", "Y")) { ok <- complete.cases(Ltv[, c(v, "PSQw", "DTFw", "THw"), with = FALSE]); cf <- coef(lm(as.formula(paste(v, "~ PSQw + DTFw + THw + factor(wave)")), data = Ltv[ok])); Ltv[, (v) := get(v) - cf["PSQw"] * fifelse(is.na(PSQw), 0, PSQw) - cf["DTFw"] * fifelse(is.na(DTFw), 0, DTFw) - cf["THw"] * fifelse(is.na(THw), 0, THw)] }
rob$tv <- swap("去除时变协变量：睡眠、距考试、治疗", c("X", "M", "Y"), as.data.frame(dcast(Ltv, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")), "BXM", "BMY")
# concurrent-effect specifications (Speyer 2025)
conc <- function(name, drop, add) { syn <- ri_model(); for (d in drop) syn <- gsub(d, "", syn); m <- safely(mfit(paste0(syn, add))); if (is.null(m)) return(NULL); i <- mc(m, "BXM", "BMY"); s <- stdpath(m, c("BXM", "BMY")); fm <- fit_stats(m, c("cfi.robust", "rmsea.robust", "srmr")); data.frame(model = name, n = lavInspect(m, "ntotal"), a_std = s$std[1], a_p = est(m, "BXM")["p"], b_std = s$std[2], b_p = est(m, "BMY")["p"], ind = i["est"], lo = i["lo"], hi = i["hi"], cfi = fm[1], rmsea = fm[2], srmr = fm[3]) }
rob$concMY <- conc("同期 M→Y 代替残差相关", "wM[1-3] ~~ CMY\\*wY[1-3]\n", "wY1 ~ cc*wM1\nwY2 ~ cc*wM2\nwY3 ~ cc*wM3\n")
rob$concXY <- conc("同期 X→Y 代替残差相关", "wX[1-3] ~~ CXY\\*wY[1-3]\n", "wY1 ~ cc*wX1\nwY2 ~ cc*wX2\nwY3 ~ cc*wX3\n")
rob$cohort <- { dc <- copy(L); for (v in c("X", "M", "Y")) dc[, (v) := get(v) - mean(get(v), na.rm = TRUE), by = .(wave, cohort)]; dd <- as.data.frame(dcast(dc, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")); swap("按批次×波次去均值（校历效应）", c("X", "M", "Y"), dd, "BXM", "BMY") }
# three schools are too few for cluster-robust errors, so school enters as a school-by-wave fixed effect
rob$school <- { dc <- copy(L); for (v in c("X", "M", "Y")) dc[, (v) := get(v) - mean(get(v), na.rm = TRUE), by = .(wave, school)]; dd <- as.data.frame(dcast(dc, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")); swap("按学校×波次去均值（学校固定效应）", c("X", "M", "Y"), dd, "BXM", "BMY") }
fwrite(rbindlist(lapply(c("X", "M", "Y"), function(v) data.frame(variable = v, r2_school = summary(lm(L[[v]] ~ factor(L$wave) * factor(L$school)))$r.squared - summary(lm(L[[v]] ~ factor(L$wave)))$r.squared))), file.path(OUT, "s38_school_variance.csv"))
# ring mediator under the laboratory ECG observation pattern: separates night averaging from the extra ring waves in the extended sample
rob$ring_lab <- { d2 <- D; for (t in 0:3) d2[[paste0("R", t)]][is.na(d2[[paste0("M", t)]])] <- NA; swap("夜间 RMSSD，仅保留实验室心电的观测模式", c("X", "R", "Y"), d2, "BXR", "BRY") }
# contrast of the alpha and lnRMSSD indirect effects in one parallel model, core sample
mpar <- safely(mfit(ri_model(c("X", "M", "A", "Y")), core))
if (!is.null(mpar)) { set.seed(20261101); co <- coef(mpar); V <- vcov(mpar); k <- match(c("BXM", "BMY", "BXA", "BAY"), names(co)); dr <- MASS::mvrnorm(20000, co[k], V[k, k]); ih <- dr[, 1] * dr[, 2]; ia <- dr[, 3] * dr[, 4]; qq <- function(v) unname(quantile(v, c(.025, .975))); fwrite(data.frame(ind_lnrmssd = co[k[1]] * co[k[2]], ind_lnrmssd_lo = qq(ih)[1], ind_lnrmssd_hi = qq(ih)[2], ind_alpha = co[k[3]] * co[k[4]], ind_alpha_lo = qq(ia)[1], ind_alpha_hi = qq(ia)[2], diff = co[k[3]] * co[k[4]] - co[k[1]] * co[k[2]], diff_lo = qq(ia - ih)[1], diff_hi = qq(ia - ih)[2]), file.path(OUT, "s38_mediator_contrast.csv")) }
ROB <- rbindlist(rob, fill = TRUE); fwrite(ROB, file.path(OUT, "s38_robustness.csv"))

# ---- 3.8 multi-group by sex and grade: equal versus free focal paths ----
mg <- function(name, g) {
  d <- D[!is.na(D[[g]]), ]
  free <- safely(mfit(ri_model(), d, aux = FALSE, group = g, group.label = c("0", "1"), group.equal = character(0)))
  if (is.null(free)) return(NULL)
  pe <- parameterEstimates(free)
  syn <- ri_model(); syn2 <- gsub("BXM\\*", "c(BXMa,BXMb)*", gsub("BMY\\*", "c(BMYa,BMYb)*", syn))
  for (k in setdiff(c("BXX", "BMX", "BYX", "BMM", "BYM", "BXY", "BYY"), character(0))) syn2 <- gsub(paste0(k, "\\*"), paste0("c(", k, "a,", k, "b)*"), syn2)
  for (k in c("CXM", "CXY", "CMY", "RX", "RM", "RY")) syn2 <- gsub(paste0(k, "\\*"), paste0("c(", k, "a,", k, "b)*"), syn2)
  syn2 <- gsub("m(X|M|Y)([0-3])\\*1", "c(m\\1\\2a,m\\1\\2b)*1", syn2)
  f <- mfit(syn2, d, aux = FALSE, group = g, group.label = c("0", "1"))
  eq <- mfit(paste0(syn2, "BXMa == BXMb\nBMYa == BMYb\n"), d, aux = FALSE, group = g, group.label = c("0", "1"))
  t <- as.data.frame(lavTestLRT(f, eq, method = "satorra.bentler.2001")); pef <- parameterEstimates(f)
  g1 <- function(l) pef[pef$label == l, ][1, c("est", "pvalue")]
  data.frame(split = name, a_group0 = g1("BXMa")$est, a_p0 = g1("BXMa")$pvalue, a_group1 = g1("BXMb")$est, a_p1 = g1("BXMb")$pvalue, b_group0 = g1("BMYa")$est, b_p0 = g1("BMYa")$pvalue, b_group1 = g1("BMYb")$est, b_p1 = g1("BMYb")$pvalue, lrt_p = t[2, "Pr(>Chisq)"])
}
MG <- rbindlist(list(safely(mg("性别（0 男，1 女）", "female")), safely(mg("年级（0 低年级或研究生，1 大三大四）", "upper"))), fill = TRUE); fwrite(MG, file.path(OUT, "s38_multigroup.csv"))

# ---- 3.5 Johnson-Neyman points and simple slopes at -1, 0, +1 SD ----
JN <- function(m, a, g) { co <- coef(m); V <- vcov(m); ww <- seq(-2.5, 2.5, .01); sl <- co[a] + co[g] * ww; se <- sqrt(V[a, a] + 2 * ww * V[a, g] + ww^2 * V[g, g]); data.frame(w = c(-1, 0, 1), slope = co[a] + co[g] * c(-1, 0, 1), p = 2 * pnorm(-abs((co[a] + co[g] * c(-1, 0, 1)) / sqrt(V[a, a] + 2 * c(-1, 0, 1) * V[a, g] + c(-1, 0, 1)^2 * V[g, g]))), jn_low = min(ww[abs(sl / se) < 1.96], NA, na.rm = TRUE), jn_high = max(ww[abs(sl / se) < 1.96], NA, na.rm = TRUE)) }
fwrite(rbind(cbind(path = "a by W1", JN(ma, "BXM", "gXW1")), cbind(path = "b by W2", JN(mb, "BMY", "gMW2"))), file.path(OUT, "s35_simple_slopes.csv"))

# ---- 3.6 profiles on the indicators the plan names; BIC, aBIC, BLRT, entropy, three-step (BCH) distal comparison ----
slope <- function(v) { ok <- !is.na(v); if (sum(ok) < 2) return(NA_real_); coef(lm(v[ok] ~ I((0:3)[ok])))[2] }
Ym <- as.matrix(D[, paste0("Y", 0:3)]); Mm <- as.matrix(D[, paste0("M", 0:3)])
ft <- data.frame(pid = D$pid, level = rowMeans(Ym, na.rm = TRUE), slope = apply(Ym, 1, slope), jump = apply(Ym, 1, function(v) { d <- diff(v); if (all(is.na(d))) NA else max(d, na.rm = TRUE) }), dM = apply(Mm, 1, slope))
ft <- merge(ft, as.data.frame(EMA[answered == 1, .(burst_sd = mean(tapply(neg_affect, wave, sd, na.rm = TRUE), na.rm = TRUE)), by = pid]), by = "pid", all.x = TRUE)
inds <- c("level", "slope", "jump", "burst_sd", "dM")
run_lpa <- function(cols, tag, nboot) {
  fx <- ft[complete.cases(ft[, cols]), c("pid", cols)]; Z <- scale(fx[, cols]); n <- nrow(Z)
  fits <- lapply(1:6, function(k) Mclust(Z, G = k, modelNames = if (k == 1) "VII" else "VVI", verbose = FALSE))
  ent <- function(z) { if (ncol(z) == 1) return(NA_real_); z <- pmax(z, 1e-12); 1 + sum(z * log(z)) / (nrow(z) * log(ncol(z))) }
  tab <- rbindlist(lapply(1:6, function(k) { m <- fits[[k]]; p <- m$df; data.frame(indicators = tag, k = k, loglik = m$loglik, bic = -m$bic, abic = -2 * m$loglik + p * log((n + 2) / 24), entropy = ent(m$z), smallest = min(table(m$classification)) / n) }))
  bl <- safely(mclustBootstrapLRT(Z, modelName = "VVI", nboot = nboot, maxG = 5, verbose = FALSE))
  if (!is.null(bl)) tab$blrt_p <- c(bl$p.value, rep(NA, 6 - length(bl$p.value)))
  list(tab = tab, fits = fits, fx = fx)
}
A1 <- run_lpa(inds, "方案五指标", 100); A2 <- run_lpa(c("level", "slope", "jump"), "症状三指标", 100)
fwrite(rbind(A1$tab, A2$tab, fill = TRUE), file.path(OUT, "s36_lpa_fit.csv"))
best <- A1$fits[[which.min(A1$tab$bic)]]; three <- A1$fits[[3]]
bch <- function(fit, fx, tag) {
  # Vermunt (2010): d[s,t] = P(assigned s | true t); each person's weight for class t is the (t, assigned class) element of its inverse
  z <- fit$z; cl <- fit$classification; K <- ncol(z)
  Dm <- sapply(1:K, function(t) sapply(1:K, function(s) sum(z[cl == s, t]) / sum(z[, t]))); inv <- solve(Dm); wgt <- t(sapply(cl, function(s) inv[, s]))
  dist <- merge(fx[, "pid", drop = FALSE], D[, c("pid", "W1", "W2", "X0", "M0", "ctq", "Y3")], by = "pid", sort = FALSE)
  rbindlist(lapply(c("W1", "W2", "X0", "M0", "ctq", "Y3"), function(v) { y <- dist[[v]]; ok <- !is.na(y); data.frame(solution = tag, distal = v, t(sapply(1:K, function(j) c(mean = sum(wgt[ok, j] * y[ok]) / sum(wgt[ok, j]))))) }), fill = TRUE)
}
fwrite(bch(three, A1$fx, "3 类"), file.path(OUT, "s36_bch_distal.csv"))
fwrite(bch(best, A1$fx, paste(best$G, "类")), file.path(OUT, "s36_bch_distal_best.csv"))
# person bootstrap of the contrast between the classes with the highest and lowest mean crisis at T3, with BCH weights held fixed; T3 crisis itself is an indicator input and is not a distal test
bch_contrast <- function(fit, fx, nboot = 2000) {
  z <- fit$z; cl <- fit$classification; K <- ncol(z)
  Dm <- sapply(1:K, function(t) sapply(1:K, function(s) sum(z[cl == s, t]) / sum(z[, t]))); wgt <- t(sapply(cl, function(s) solve(Dm)[, s]))
  dist <- merge(fx[, "pid", drop = FALSE], D[, c("pid", "W1", "W2", "X0", "M0", "ctq", "Y3")], by = "pid", sort = FALSE)
  wm <- function(y, j, i) { ok <- !is.na(y[i]); sum(wgt[i, j][ok] * y[i][ok]) / sum(wgt[i, j][ok]) }
  all_i <- seq_len(nrow(dist)); y3 <- sapply(1:K, function(j) wm(dist$Y3, j, all_i)); hi <- which.max(y3); lo <- which.min(y3); set.seed(20261102)
  rbindlist(lapply(c("W1", "W2", "X0", "M0", "ctq"), function(v) { b <- replicate(nboot, { i <- sample(all_i, replace = TRUE); wm(dist[[v]], hi, i) - wm(dist[[v]], lo, i) }); data.frame(distal = v, high = wm(dist[[v]], hi, all_i), low = wm(dist[[v]], lo, all_i), diff = wm(dist[[v]], hi, all_i) - wm(dist[[v]], lo, all_i), lo95 = unname(quantile(b, .025)), hi95 = unname(quantile(b, .975))) }))
}
fwrite(bch_contrast(best, A1$fx), file.path(OUT, "s36_bch_contrast_best.csv"))
fwrite(data.frame(pid = A1$fx$pid, class_best = best$classification, class_3 = three$classification), file.path(OUT, "s36_lpa_assignment.csv"))
fwrite(data.frame(k_bic = best$G, k3_sizes = paste(round(table(three$classification) / length(three$classification), 3), collapse = "/")), file.path(OUT, "s36_summary.csv"))

# LCGA convergence check, class-specific linear trajectories without random effects
lcg <- safely({
  suppressPackageStartupMessages(library(lcmm))
  long <- L[!is.na(Y), .(id = as.integer(factor(pid)), pid, wave, Y)]; long <- as.data.frame(long); set.seed(1)
  m1 <- hlme(Y ~ wave, subject = "id", ng = 1, data = long, verbose = FALSE)
  out <- list(m1); for (k in 2:4) out[[k]] <- gridsearch(rep = 8, maxiter = 15, minit = m1, hlme(Y ~ wave, mixture = ~wave, subject = "id", ng = k, data = long, verbose = FALSE))
  list(tab = rbindlist(lapply(seq_along(out), function(k) { m <- out[[k]]; data.frame(k = k, bic = m$BIC, loglik = m$loglik, smallest = if (k > 1) min(table(m$pprob$class)) / nrow(m$pprob) else 1) })), mods = out, long = long)
})
if (!is.null(lcg)) { fwrite(lcg$tab, file.path(OUT, "s36_lcga_fit.csv")); ids <- unique(lcg$long[, c("id", "pid")]); cl <- merge(lcg$mods[[3]]$pprob[, c("id", "class")], ids, by = "id"); cc <- merge(cl, data.frame(pid = A1$fx$pid, lpa = three$classification), by = "pid"); res$lcga_vs_lpa_ari <- adjustedRandIndex(cc$class, cc$lpa) }

# ---- multilevel lagged models on the burst days, a frequentist companion to the DSEM in dsem_burst.py ----
dd <- EMA[answered == 1, .(st = mean(stress, na.rm = TRUE), na = mean(neg_affect, na.rm = TRUE)), by = .(pid, wave, burst_day)]
dd <- merge(dd, W[valid_night == 1, .(pid, wave, burst_day, lnr = ln_night_rmssd)], by = c("pid", "wave", "burst_day"), all = TRUE)
setorder(dd, pid, wave, burst_day)
dd[, `:=`(stw = st - mean(st, na.rm = TRUE), naw = na - mean(na, na.rm = TRUE), lw = lnr - mean(lnr, na.rm = TRUE)), by = .(pid, wave)]
dd[, `:=`(lw_l1 = shift(lw), stw_l1 = shift(stw), naw_l1 = shift(naw)), by = .(pid, wave)]; dd[, id := paste(pid, wave)]
ml <- function(f, lab, data = dd) { m <- lme(f, random = ~1 | pid, data = data, na.action = na.omit, control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)); tt <- summary(m)$tTable; k <- which(rownames(tt) == lab); data.frame(est = tt[k, "Value"], se = tt[k, "Std.Error"], p = tt[k, "p-value"], n = nobs(m)) }
DS <- rbindlist(list(cbind(model = "当日压力→当夜 lnRMSSD", ml(lw ~ stw + lw_l1, "stw")), cbind(model = "当夜 lnRMSSD→次日负性情绪", ml(naw ~ lw_l1 + naw_l1, "lw_l1")), cbind(model = "负性情绪→当夜 lnRMSSD（反向）", ml(lw ~ naw + lw_l1, "naw"))))
fwrite(DS, file.path(OUT, "s38_burst_multilevel.csv"))

# ---- objective load against perceived stress ----
oc <- L[, .(r_deadlines = cor(dl, academic_stress, use = "pair"), r_exams = cor(ex, academic_stress, use = "pair"), r_jobs = cor(ap, employment_anxiety, use = "pair"), r_days = cor(days_to_next_final, academic_stress, use = "pair"))]
res$objective_vs_perceived <- round(as.data.frame(oc), 3)

sink(file.path(OUT, "supplement_summary.txt")); print(res); print(H); print(T4); print(ROB); print(MG); print(DS); sink()
saveRDS(res, file.path(OUT, "supplement.rds")); cat("SUPPLEMENT DONE\n")
