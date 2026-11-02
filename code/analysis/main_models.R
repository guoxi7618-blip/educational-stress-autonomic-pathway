# Primary analyses (framework section 4.7). Usage: Rscript main_models.R <processed_dir> <out_dir> [table_prefix]
suppressPackageStartupMessages({library(lavaan); library(mclust); library(psych); library(data.table)})
args <- commandArgs(TRUE); DIR <- ifelse(length(args) >= 1, args[1], "processed"); OUT <- ifelse(length(args) >= 2, args[2], file.path(DIR, "results")); PREFIX <- ifelse(length(args) >= 3, args[3], "")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
rd <- function(f) fread(file.path(DIR, paste0(PREFIX, f, ".csv")))
P <- rd("participants"); S <- rd("scale_scores"); I <- rd("scale_items"); E <- rd("ecg_lab"); G <- rd("eeg_lab"); C <- rd("clinician_assessment")
W <- rd("wearable_nightly"); EMA <- rd("ema_prompts")
res <- list()

# ---- composites on the baseline scale ----
L <- merge(S, C[, .(pid, wave, cgi_s, cssrs_intensity)], by = c("pid", "wave"), all = TRUE)
L <- merge(L, E[, .(pid, wave, ln_rmssd)], by = c("pid", "wave"), all = TRUE)
L <- merge(L, G[, .(pid, wave, alpha_periodic_log)], by = c("pid", "wave"), all = TRUE)
zb <- function(x, w) (x - mean(x[w == 0], na.rm = TRUE)) / sd(x[w == 0], na.rm = TRUE)
L[, `:=`(zac = zb(academic_stress, wave), zem = zb(employment_anxiety, wave), zin = zb(interpersonal_stress, wave), zph = zb(phq9_total, wave), zga = zb(gad7_total, wave), zcg = zb(cgi_s, wave), zcs = zb(cssrs_intensity, wave), zds = zb(dass_stress, wave))]
L[, X := rowMeans(.SD), .SDcols = c("zac", "zem", "zin")]
L[, Y := rowMeans(.SD), .SDcols = c("zph", "zga", "zcg", "zcs")]
L[, M := zb(ln_rmssd, wave)]; L[, A := zb(alpha_periodic_log, wave)]
base <- L[wave == 0, .(pid, W1 = as.numeric(scale(rowMeans(cbind(scale(sam_challenge), scale(stress_mindset))))), W2 = as.numeric(scale(rowMeans(cbind(scale(erq_reappraisal), -scale(ders16_total))))), psqi0 = psqi_total)]
L <- merge(L, base, by = "pid", all.x = TRUE); L <- merge(L, P[, .(pid, tier, sex, grade, ctq_total)], by = "pid")
L[, `:=`(female = as.numeric(sex == "F"), ctq = as.numeric(scale(ctq_total)))]
L[, `:=`(Xw = X - mean(X, na.rm = TRUE), Mw = M - mean(M, na.rm = TRUE)), by = pid]
L[, `:=`(PX = Xw * W1, PM = Mw * W2)]
vars <- c("X", "M", "Y", "A", "PX", "PM", "zac", "zem", "zin", "zph", "zga", "zds", "zcg", "zcs")
D <- dcast(L, pid + W1 + W2 + female + ctq + tier + grade ~ wave, value.var = vars, sep = "")
D <- as.data.frame(D)
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "aux_fiml.R")); AUXD <- aux_data(P, S)

# ---- descriptives, reliability, ICC ----
desc <- L[, .(n = .N, phq9_m = mean(phq9_total, na.rm = TRUE), phq9_sd = sd(phq9_total, na.rm = TRUE), gad7_m = mean(gad7_total, na.rm = TRUE), gad7_sd = sd(gad7_total, na.rm = TRUE), psqi_m = mean(psqi_total, na.rm = TRUE),
              acad_m = mean(academic_stress, na.rm = TRUE), emp_m = mean(employment_anxiety, na.rm = TRUE), inter_m = mean(interpersonal_stress, na.rm = TRUE),
              ln_rmssd_m = mean(ln_rmssd, na.rm = TRUE), ln_rmssd_sd = sd(ln_rmssd, na.rm = TRUE), alpha_m = mean(alpha_periodic_log, na.rm = TRUE), pct_phq10 = mean(phq9_total >= 10, na.rm = TRUE)), by = wave][order(wave)]
fwrite(desc, file.path(OUT, "t1_descriptives.csv"))
icc1 <- function(v) { d <- L[!is.na(get(v)), .(pid, y = get(v))]; m <- nlme::lme(y ~ 1, random = ~1 | pid, data = d); vc <- as.numeric(nlme::VarCorr(m)[, 1]); vc[1] / sum(vc) }
res$icc <- sapply(c("X", "M", "Y", "A"), icc1)
ialpha <- function(pre, w = 0) { cols <- grep(paste0("^", pre, "_"), names(I), value = TRUE); psych::alpha(as.data.frame(I[wave == w, ..cols]), warnings = FALSE, check.keys = FALSE)$total$raw_alpha }
res$alpha <- sapply(c("acad", "emp", "inter", "phq", "gad", "dass", "sam", "smm", "erq", "ders"), ialpha)

# ---- longitudinal measurement invariance (composite indicators) ----
mi_syntax <- function(ind, load = FALSE, int = FALSE) {
  s <- ""
  for (t in 0:3) {
    v <- paste0(ind, t)
    lab <- if (load) paste0("l", seq_along(ind), "*") else rep("", length(ind))
    rhs <- if (load) paste0("NA*", v[1], "+", paste0(lab, v, collapse = "+")) else paste0("NA*", v[1], "+", paste0(v[-1], collapse = "+"))
    s <- paste0(s, "F", t, " =~ ", rhs, "\n")
    if (int) s <- paste0(s, paste0(v, " ~ i", seq_along(ind), "*1", collapse = "\n"), "\n")
    else s <- paste0(s, paste0(v, " ~ 1", collapse = "\n"), "\n")
    s <- paste0(s, "F", t, " ~~ ", if (t == 0) "1" else "NA", "*F", t, "\n", "F", t, " ~ ", if (t == 0 || !int) "0" else "NA", "*1\n")
  }
  for (k in seq_along(ind)) for (a in 0:2) for (b in (a + 1):3) s <- paste0(s, ind[k], a, " ~~ ", ind[k], b, "\n")
  s
}
fitmi <- function(ind) {
  out <- list()
  for (lv in c("configural", "metric", "scalar")) {
    m <- cfa(mi_syntax(ind, lv != "configural", lv == "scalar"), D, missing = "fiml", estimator = "MLR", meanstructure = TRUE)
    out[[lv]] <- fitMeasures(m, c("chisq.scaled", "df", "cfi.robust", "rmsea.robust", "srmr"))
  }
  do.call(rbind, out)
}
res$mi_X <- fitmi(c("zac", "zem", "zin")); res$mi_Y <- fitmi(c("zph", "zga", "zcg", "zcs")); res$mi_Y_dass <- fitmi(c("zph", "zga", "zds"))

# ---- common method variance: Harman plus single-method loading share ----
item_cols <- grep("^(acad|emp|inter|phq|gad|dass)_", names(I), value = TRUE)
ev <- eigen(cor(as.matrix(I[wave == 0, ..item_cols]), use = "pair"))$values
res$harman_first_factor <- ev[1] / sum(ev)

# ---- RI-CLPM helpers ----
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
  pairs <- combn(vn, 2, simplify = FALSE)
  for (pr in pairs) {
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
mfit <- function(syn, data = D, aux = TRUE, extra_obs = character(0)) { if (aux) { data <- with_aux(data, AUXD); syn <- paste0(syn, aux_syntax(syn, extra_obs)) }; sem(syn, data, missing = "fiml", estimator = "MLR", meanstructure = TRUE, orthogonal = FALSE, auto.fix.first = FALSE, auto.var = FALSE, auto.cov.lv.x = FALSE, auto.cov.y = FALSE, se = "robust.mlr") }
fm <- function(m) round(fit_stats(m), 3)
paths <- function(m) {
  ss <- standardizedSolution(m); pe <- parameterEstimates(m)
  r <- pe[pe$op == "~" & nzchar(pe$label), c("lhs", "rhs", "label", "est", "se", "z", "pvalue", "ci.lower", "ci.upper")]
  std <- merge(r[, c("lhs", "rhs")], ss[ss$op == "~", c("lhs", "rhs", "est.std")], by = c("lhs", "rhs"))
  r$est.std <- std$est.std[match(paste(r$lhs, r$rhs), paste(std$lhs, std$rhs))]
  uu <- r[!duplicated(r$label), c("label", "est", "se", "z", "pvalue", "ci.lower", "ci.upper")]
  uu$std_mean <- tapply(r$est.std, r$label, mean)[uu$label]; uu
}
mc_ind <- function(m, pa, pb, nsim = 20000) {
  set.seed(20261101)  # fixed so confidence limits are reproducible
  co <- coef(m); V <- vcov(m); i <- match(c(pa, pb), names(co))
  dr <- MASS::mvrnorm(nsim, co[i], V[i, i]); prod <- dr[, 1] * dr[, 2]
  c(est = co[i[1]] * co[i[2]], lo = unname(quantile(prod, .025)), hi = unname(quantile(prod, .975)), p_two = 2 * min(mean(prod > 0), mean(prod < 0)))
}

# ---- CLPM vs RI-CLPM; invariance test ----
free <- mfit(ri_model(invariant = FALSE)); inv <- mfit(ri_model(invariant = TRUE))
clpm <- mfit(ri_model(invariant = TRUE, ri = FALSE))
res$fit <- rbind(RI_free = fm(free), RI_invariant = fm(inv), CLPM_invariant = fm(clpm))
inv0 <- mfit(ri_model(), aux = FALSE); res$fit_no_aux <- fm(inv0); fwrite(paths(inv0), file.path(OUT, "t4b_riclpm_no_auxiliary.csv")); res$indirect_HRV_no_aux <- mc_ind(inv0, "BXM", "BMY")
res$lrt_ri_vs_clpm <- tryCatch(as.data.frame(lavTestLRT(inv, clpm, method = "satorra.bentler.2001")), error = function(e) NA)
res$lrt_invariance <- tryCatch(as.data.frame(lavTestLRT(free, inv, method = "satorra.bentler.2001")), error = function(e) NA)
p_main <- paths(inv); fwrite(p_main, file.path(OUT, "t4_riclpm_within_paths.csv"))
p_clpm <- paths(clpm); fwrite(p_clpm, file.path(OUT, "t3b_clpm_paths.csv"))
ri_cor <- standardizedSolution(inv); res$ri_corr <- ri_cor[ri_cor$op == "~~" & grepl("^RI", ri_cor$lhs) & grepl("^RI", ri_cor$rhs) & ri_cor$lhs != ri_cor$rhs, c("lhs", "rhs", "est.std", "pvalue")]
res$indirect_HRV <- mc_ind(inv, "BXM", "BMY")

# ---- moderated mediation: observed within-person products ----
ext_a <- paste0(sapply(1:3, function(t) paste0("wM", t, " ~ gXW1*PX", t - 1, "\n")), collapse = "")
ext_a <- paste0(ext_a, "wM1 ~ dW1*W1\nwM2 ~ dW1*W1\nwM3 ~ dW1*W1\nPX0 ~~ wX0 + wM0 + wY0\nPX1 ~~ wX1 + wM1 + wY1\nPX2 ~~ wX2 + wM2 + wY2\nPX0 ~~ PX0\nPX1 ~~ PX1\nPX2 ~~ PX2\nPX0 ~~ PX1 + PX2\nPX1 ~~ PX2\nW1 ~~ W1\nW1 ~~ RIX + RIM + RIY + PX0 + PX1 + PX2\nW1 ~ 1\nPX0 ~ 1\nPX1 ~ 1\nPX2 ~ 1\n")
ma <- mfit(ri_model(extra = ext_a), extra_obs = c("PX0", "PX1", "PX2", "W1"))
ext_b <- paste0(sapply(1:3, function(t) paste0("wY", t, " ~ gMW2*PM", t - 1, "\n")), collapse = "")
ext_b <- paste0(ext_b, "wY1 ~ dW2*W2\nwY2 ~ dW2*W2\nwY3 ~ dW2*W2\nPM0 ~~ wX0 + wM0 + wY0\nPM1 ~~ wX1 + wM1 + wY1\nPM2 ~~ wX2 + wM2 + wY2\nPM0 ~~ PM0\nPM1 ~~ PM1\nPM2 ~~ PM2\nPM0 ~~ PM1 + PM2\nPM1 ~~ PM2\nW2 ~~ W2\nW2 ~~ RIX + RIM + RIY + PM0 + PM1 + PM2\nW2 ~ 1\nPM0 ~ 1\nPM1 ~ 1\nPM2 ~ 1\n")
mb <- mfit(ri_model(extra = ext_b), extra_obs = c("PM0", "PM1", "PM2", "W2"))
pick <- function(m, lab) { pe <- parameterEstimates(m); pe[pe$label == lab, c("est", "se", "z", "pvalue", "ci.lower", "ci.upper")][1, ] }
res$mod_a <- pick(ma, "gXW1"); res$mod_b <- pick(mb, "gMW2")
cc <- coef(ma); res$simple_a <- c(a_low = cc["BXM"] - cc["gXW1"], a_mean = cc["BXM"], a_high = cc["BXM"] + cc["gXW1"])
cb <- coef(mb); res$simple_b <- c(b_low = cb["BMY"] - cb["gMW2"], b_mean = cb["BMY"], b_high = cb["BMY"] + cb["gMW2"])
res$imm_a <- mc_ind(ma, "gXW1", "BMY"); res$imm_b <- c(est = unname(cb["BXM"] * cb["gMW2"]), mc_ind(mb, "BXM", "gMW2")[2:4])

# ---- alpha as second mediator (core sample) ----
Dc <- D[D$tier == "core", ]
malpha <- mfit(ri_model(c("X", "A", "Y")), Dc)
p_alpha <- paths(malpha); fwrite(p_alpha, file.path(OUT, "t5b_alpha_paths.csv")); res$fit_alpha <- fm(malpha); res$indirect_alpha <- mc_ind(malpha, "BXA", "BAY")
mcore <- mfit(ri_model(), Dc); res$indirect_HRV_core_only <- mc_ind(mcore, "BXM", "BMY"); res$fit_core <- fm(mcore); fwrite(paths(mcore), file.path(OUT, "t5c_core_only_paths.csv"))

# ---- subtypes: temporal features, LPA + convergence check by trajectory clustering ----
wide <- D[, c("pid", paste0("Y", 0:3), paste0("X", 0:3), paste0("M", 0:3))]
feat <- data.frame(pid = wide$pid)
Ym <- as.matrix(wide[, paste0("Y", 0:3)]); Xm <- as.matrix(wide[, paste0("X", 0:3)]); Mm <- as.matrix(wide[, paste0("M", 0:3)])
slope <- function(v) { ok <- !is.na(v); if (sum(ok) < 2) return(NA); coef(lm(v[ok] ~ I((0:3)[ok])))[2] }
feat$level <- rowMeans(Ym, na.rm = TRUE); feat$base <- Ym[, 1]
feat$slope <- apply(Ym, 1, slope)
feat$jump <- apply(Ym, 1, function(v) { d <- diff(v); if (all(is.na(d))) NA else max(d, na.rm = TRUE) })
ema_v <- EMA[answered == 1, .(ema_sd = sd(neg_affect, na.rm = TRUE)), by = pid]
feat <- merge(feat, as.data.frame(ema_v), by = "pid", all.x = TRUE)
fx <- feat[complete.cases(feat[, c("base", "slope", "jump")]), ]; Z <- scale(fx[, c("base", "jump")])
# multi-start EM: keep the best log-likelihood across random initial partitions for each k
fit_k <- function(k, nstart = 30) {
  best <- Mclust(Z, G = k, modelNames = "VVI", verbose = FALSE)
  if (k > 1) for (s in seq_len(nstart)) {
    m <- tryCatch(Mclust(Z, G = k, modelNames = "VVI", initialization = list(hcPairs = hcRandomPairs(Z, seed = s)), verbose = FALSE), error = function(e) NULL)
    if (!is.null(m) && m$loglik > best$loglik) best <- m
  }
  best
}
fits <- lapply(1:6, fit_k); bics <- sapply(fits, function(m) -m$bic); lp <- fits[[which.min(bics)]]
ent <- function(z) { z <- pmax(z, 1e-12); 1 + sum(z * log(z)) / (nrow(z) * log(ncol(z))) }
lpa_table <- do.call(rbind, lapply(1:6, function(k) { m <- fits[[k]]; data.frame(k = k, loglik = m$loglik, bic = -m$bic, entropy = if (k > 1) ent(m$z) else NA, smallest = min(table(m$classification)) / nrow(Z)) }))
fwrite(lpa_table, file.path(OUT, "t6_lpa_fit.csv"))
fx$class <- lp$classification
prof <- cbind(as.data.table(fx)[, lapply(.SD, mean), by = class, .SDcols = c("level", "slope", "jump", "base")], n = as.numeric(table(fx$class)))
fwrite(prof[order(class)], file.path(OUT, "t6b_profiles.csv"))
fwrite(fx[, c("pid", "class")], file.path(OUT, "lpa_class_assignment.csv"))
res$lpa <- list(G = lp$G, model = lp$modelName, n = nrow(Z), entropy = ent(lp$z))

# ---- daily within-person: stress -> nocturnal HRV ----
dd <- EMA[answered == 1, .(st = mean(stress, na.rm = TRUE)), by = .(pid, wave, burst_day)]
dd <- merge(dd, W[valid_night == 1, .(pid, wave, burst_day, rmssd = night_rmssd_ms, lnr = ln_night_rmssd)], by = c("pid", "wave", "burst_day"))
dd[, `:=`(stw = st - mean(st), stb = mean(st)), by = .(pid, wave)]
ml <- nlme::lme(rmssd ~ stw + stb, random = ~1 | pid/wave, data = dd, na.action = na.omit, control = nlme::lmeControl(opt = "optim"))
res$daily_stress_rmssd_ms <- summary(ml)$tTable["stw", ]


# ---- convergence check: trajectory mixture on complete 4-wave profiles ----
trw <- wide[complete.cases(wide[, paste0("Y", 0:3)]), c("pid", paste0("Y", 0:3))]
tm <- Mclust(trw[, -1], G = 1:6, modelNames = c("EEI", "VVI"))
tm3 <- Mclust(trw[, -1], G = lp$G, modelNames = "VVI")
tr <- merge(data.frame(pid = trw$pid, tc = tm3$classification), fx[, c("pid", "class")], by = "pid")
res$traj <- list(G = tm$G, model = tm$modelName, ari_with_lpa = adjustedRandIndex(tr$tc, tr$class), means = aggregate(trw[, paste0("Y", 0:3)], list(class = tm3$classification), mean))

# ---- profile validity against baseline moderators and physiology ----
fx2 <- merge(fx, as.data.frame(D[, c("pid", "W1", "W2", "M0", "X0", "ctq")]), by = "pid")
res$profile_anova <- sapply(c("W1", "W2", "M0", "X0", "ctq"), function(v) summary(aov(fx2[[v]] ~ factor(fx2$class)))[[1]][1, 5])
res$profile_means <- aggregate(fx2[, c("W1", "W2", "M0", "X0", "ctq")], list(class = fx2$class), mean, na.rm = TRUE)

# ---- robustness ----
rob <- list()
cov_ext <- "RIX ~ female + ctq\nRIM ~ female + ctq\nRIY ~ female + ctq\nfemale ~~ female\nctq ~~ ctq\nfemale ~~ ctq\nfemale ~ 1\nctq ~ 1\n"
mcov <- mfit(ri_model(extra = cov_ext), aux = FALSE); rob$covariates <- mc_ind(mcov, "BXM", "BMY")
Lt <- copy(L); Lt[therapy_use == 1, `:=`(Y = NA_real_, M = NA_real_, X = NA_real_)]
Dt <- as.data.frame(dcast(Lt, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")); mth <- mfit(ri_model(), Dt); rob$censor_therapy <- mc_ind(mth, "BXM", "BMY")
Wb <- W[valid_night == 1, .(n = .N, Wr = mean(ln_night_rmssd)), by = .(pid, wave)][n >= 5]
Lw <- merge(L, Wb, by = c("pid", "wave"), all.x = TRUE); Lw[, M := zb(Wr, wave)]
Dw <- as.data.frame(dcast(Lw, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")); mw <- mfit(ri_model(), Dw); rob$wearable_mediator <- mc_ind(mw, "BXM", "BMY"); fwrite(paths(mw), file.path(OUT, "t8_wearable_mediator_paths.csv"))
conc <- gsub("wX1 ~~ CXM\\*wM1\\n|wX2 ~~ CXM\\*wM2\\n|wX3 ~~ CXM\\*wM3\\n", "", ri_model())
conc <- paste0(conc, "wM1 ~ cc*wX1\nwM2 ~ cc*wX2\nwM3 ~ cc*wX3\n"); mco <- mfit(conc); rob$concurrent_XtoM <- mc_ind(mco, "BXM", "BMY")
# pattern-mixture sensitivity: multiple imputation of Y, a shift in baseline SDs on imputed values, X and M left to FIML; Monte Carlo draws stacked across imputations (Zhou & Reiter 2010)
Dmi <- as.data.frame(dcast(L, pid ~ wave, value.var = c("X", "M", "Y"), sep = "")); Ycols <- paste0("Y", 0:3); Ymiss <- is.na(as.matrix(Dmi[, Ycols]))
mi_imp <- mice::mice(merge(Dmi, AUXD, by = "pid")[, -1], m = 20, maxit = 10, method = "norm", seed = 20261102, printFlag = FALSE)
mnar <- function(dl, draws = 1000) {
  set.seed(20261102)
  prod <- unlist(lapply(seq_len(mi_imp$m), function(i) {
    Dp <- Dmi; Yi <- as.matrix(mice::complete(mi_imp, i)[, Ycols]); Yi[Ymiss] <- Yi[Ymiss] + dl * sd(Dmi$Y0, na.rm = TRUE); Dp[, Ycols] <- Yi
    m <- mfit(ri_model(), Dp); co <- coef(m); V <- vcov(m); k <- match(c("BXM", "BMY"), names(co)); dr <- MASS::mvrnorm(draws, co[k], V[k, k]); dr[, 1] * dr[, 2]
  }))
  c(est = mean(prod), lo = unname(quantile(prod, .025)), hi = unname(quantile(prod, .975)), p_two = 2 * min(mean(prod > 0), mean(prod < 0)))
}
rob$mnar_delta_0 <- mnar(0); rob$mnar_delta_0.3 <- mnar(.3); rob$mnar_delta_0.6 <- mnar(.6)
res$robust <- do.call(rbind, rob)

# ---- attrition analysis ----
al <- merge(P[, .(pid, last_wave_attended, sex)], base, by = "pid"); al <- merge(al, L[wave == 0, .(pid, X, Y)], by = "pid")
res$attrition <- summary(glm(I(last_wave_attended < 3) ~ Y + X + W1 + W2 + I(sex == "M"), binomial, al))$coefficients

# ---- unmeasured latent method factor at T0 ----
ucols <- grep("^(acad|emp|inter|phq|gad)_", names(I), value = TRUE); U <- as.data.frame(I[wave == 0, ..ucols])
fx_ <- paste0("FX =~ ", paste(grep("^(acad|emp|inter)", ucols, value = TRUE), collapse = "+")); fy_ <- paste0("FY =~ ", paste(grep("^(phq|gad)", ucols, value = TRUE), collapse = "+"))
res$ulmc <- tryCatch({
  m0 <- cfa(paste(fx_, fy_, sep = "\n"), U, estimator = "MLR", std.lv = TRUE)
  mm <- cfa(paste(fx_, fy_, paste0("FM =~ ", paste0("l*", ucols, collapse = "+")), "FM ~~ 0*FX\nFM ~~ 0*FY", sep = "\n"), U, estimator = "MLR", std.lv = TRUE)
  ssm <- standardizedSolution(mm); ss0 <- standardizedSolution(m0)
  c(cfi0 = unname(fitMeasures(m0, "cfi.robust")), cfi1 = unname(fitMeasures(mm, "cfi.robust")), method_var_share = mean(ssm$est.std[ssm$lhs == "FM" & ssm$op == "=~"]^2), r0 = ss0[ss0$lhs == "FX" & ss0$rhs == "FY", "est.std"], r1 = ssm[ssm$lhs == "FX" & ssm$rhs == "FY", "est.std"])
}, error = function(e) c(error = conditionMessage(e)))

# ---- Johnson-Neyman style region for moderated paths ----
Va <- vcov(ma)[c("BXM", "gXW1"), c("BXM", "gXW1")]; ww <- seq(-2.5, 2.5, .01)
nsr <- function(sl, se) { i <- ww[abs(sl / se) <= 1.96]; if (length(i)) range(i) else c(NA, NA) }
sl <- cc["BXM"] + cc["gXW1"] * ww; se <- sqrt(Va[1, 1] + 2 * ww * Va[1, 2] + ww^2 * Va[2, 2]); res$jn_a_nonsig_interval <- nsr(sl, se)
Vb <- vcov(mb)[c("BMY", "gMW2"), c("BMY", "gMW2")]; sl <- cb["BMY"] + cb["gMW2"] * ww; se <- sqrt(Vb[1, 1] + 2 * ww * Vb[1, 2] + ww^2 * Vb[2, 2]); res$jn_b_nonsig_interval <- nsr(sl, se)

saveRDS(res, file.path(OUT, "results.rds"))
sink(file.path(OUT, "results_summary.txt")); print(res); print(p_main); print(p_alpha); print(lpa_table); print(prof); print(desc); sink()
cat("DONE\n")
