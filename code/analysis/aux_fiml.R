# Auxiliary variables in FIML as saturated correlates (Graham, 2003). They improve the missing-at-random treatment and leave the model's degrees of freedom unchanged.
# Each auxiliary covaries freely with every wave's within-person latent, with any other observed variable in the model and with the other auxiliaries.
AUX <- c("aCTQ", "aSUP", "aSES", "aAGE", "aPSQ")

aux_data <- function(P, S) {
  d <- merge(P[, .(pid, aCTQ = as.numeric(scale(ctq_total)), aSUP = as.numeric(scale(social_support)), aSES = as.numeric(scale(family_ses)), aAGE = as.numeric(scale(age)))], S[wave == 0, .(pid, aPSQ = as.numeric(scale(psqi_total)))], by = "pid", all.x = TRUE)
  as.data.frame(d)
}

aux_syntax <- function(syn, extra_obs = character(0)) {
  lat <- unique(regmatches(syn, gregexpr("w[A-Za-z]+[0-3](?= =~)", syn, perl = TRUE))[[1]])
  s <- paste0(AUX, " ~ 1\n", AUX, " ~~ ", AUX, "\n", collapse = "")
  for (a in AUX) s <- paste0(s, a, " ~~ ", paste(c(lat, extra_obs), collapse = " + "), "\n")
  for (p in combn(AUX, 2, simplify = FALSE)) s <- paste0(s, p[1], " ~~ ", p[2], "\n")
  s
}

with_aux <- function(data, aux_d) if (all(AUX %in% names(data))) data else merge(data, aux_d, by = "pid", all.x = TRUE, sort = FALSE)

# CFI against a baseline in which the substantive variables are independent and the auxiliaries stay saturated with them; SRMR over the substantive variables only
fit_stats <- function(m, what = c("chisq.scaled", "df", "pvalue.scaled", "cfi.robust", "tli.robust", "rmsea.robust", "srmr", "bic")) {
  ov <- setdiff(lavNames(m, "ov"), AUX)
  if (length(ov) == length(lavNames(m, "ov"))) return(fitMeasures(m, what))
  syn <- paste0(paste0(ov, " ~~ ", ov, "\n", ov, " ~ 1\n", collapse = ""), paste0(AUX, " ~ 1\n", AUX, " ~~ ", AUX, "\n", collapse = ""))
  for (a in AUX) syn <- paste0(syn, a, " ~~ ", paste(ov, collapse = " + "), "\n")
  for (p in combn(AUX, 2, simplify = FALSE)) syn <- paste0(syn, p[1], " ~~ ", p[2], "\n")
  b <- sem(syn, as.data.frame(lavInspect(m, "data")), missing = "fiml", estimator = "MLR", meanstructure = TRUE, orthogonal = FALSE, auto.fix.first = FALSE, auto.var = FALSE, auto.cov.lv.x = FALSE, auto.cov.y = FALSE, se = "none")
  f <- fitMeasures(m, what, baseline.model = b)
  r <- lavResiduals(m, type = "cor.bentler")$cov[ov, ov]
  if ("srmr" %in% what) f["srmr"] <- sqrt(mean(r[lower.tri(r, diag = TRUE)]^2))
  f
}
