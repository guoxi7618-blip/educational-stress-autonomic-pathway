# Data package: Within-person autonomic pathways link educational stress to subclinical crisis in university students

This package holds the summary-level data behind the numbers reported in the manuscript. It also documents the record-level tables from which these numbers were computed. Each summary file is described column by column in `09_codebook/codebook_summary_tables.csv`. Each record-level table is described variable by variable in `09_codebook/codebook_record_level_tables.csv`.

## 1. Study overview

| Item                | Value                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Design              | Four-wave panel at months 0, 2, 4 and 6 with two start cohorts in three universities              |
| Screened / eligible | 2,000 screened; 1,807 eligible; 1,132 eligible and consented                                      |
| Enrolled            | 600 (core 300, extended 300), sampled with stratification on year of study, sex and stress level  |
| Retained            | T0, T1, T2, T3: 600 / 553 / 521 / 498                                                             |
| Laboratory ECG      | Lead II, 1,000 Hz; core sample at all waves, extended sample at T0 and T3 (1,625 sessions)        |
| Laboratory EEG      | Core sample; 32 active electrodes, 1,000 Hz, eyes open and closed (1,072 sessions)                |
| Bursts              | 14 days after each wave: ring, three momentary prompts per day, smartphone sensing (2,172 bursts) |
| Clinician ratings   | CGI-S and C-SSRS at each wave, blind to device and smartphone data                                |
| Primary model       | Random intercept cross-lagged panel model with auxiliary variables; n = 600                       |
| Safety              | Three-source risk monitoring with clinician contact within 24 h (540 events)                      |

## 2. Conventions

- **Format:** comma-separated values, UTF-8 with byte-order mark (opens directly in Excel), one header row.

- **Number formatting:** cells such as `7.44 (5.03)` are M (SD). Cells such as `[0.005, 0.054]` are 95% intervals. Thousands separators (`1,625`) appear where the manuscript uses them. Many files keep more decimals than the manuscript prints, so that rounding can be checked.

- **Waves:** T0, T1, T2 and T3 are months 0, 2, 4 and 6. The record-level `wave` variable codes them 0 to 3.

- **Composites:** educational stress is the mean of the standardized academic stress, employment anxiety and interpersonal stress scales. Subclinical crisis is the mean of the standardized PHQ-9, GAD-7, CGI-S and C-SSRS ideation intensity scores. All are standardized on baseline values. lnRMSSD is resting laboratory lnRMSSD unless stated.

- **Estimates:** β is the standardized within-person lag averaged over waves. Indirect effects are unstandardized a × b with Monte Carlo 95% CIs from 20,000 draws. Daily paths are posterior medians with 95% credible intervals.

- **Profiles and states:** P1 to P5 are numbered from lowest to highest crisis at T3. S1 to S5 are ordered from least to most severe. Transition intervals from S4 and S5 are close to the [0, 1] bounds and are poorly determined.

## 3. File inventory

### 01_participants

| File                              | Content                                                       | Manuscript                   |
| --------------------------------- | ------------------------------------------------------------- | ---------------------------- |
| `enrolment_flow.csv`              | Screening and enrolment counts (2,000 / 1,807 / 1,132 / 600)  | Results (Retention); Fig. 1c |
| `participant_characteristics.csv` | Baseline characteristics, total and by core / extended sample | Table 1                      |
| `retention_by_wave.csv`           | Participants retained at each wave (600 / 553 / 521 / 498)    | Results (Retention); Fig. 1c |
| `dropout_model.csv`               | Logistic model of not completing T3 (recomputed)              | Results (Retention)          |

### 02_descriptives_and_quality

| File                                 | Content                                                                   | Manuscript                   |
| ------------------------------------ | ------------------------------------------------------------------------- | ---------------------------- |
| `outcomes_by_wave.csv`               | Symptoms, stress, sleep and resting lnRMSSD at each wave                  | Results (Retention)          |
| `data_quality.csv`                   | Valid sessions, nights, person-days and answered prompts                  | Results (Retention); Fig. 1e |
| `laboratory_manipulation_checks.csv` | Autonomic response to the serial subtraction task and EEG spectral checks | Results (Retention)          |
| `eeg_alpha_by_channel.csv`           | Aperiodic-adjusted alpha peak height for each of the 33 channels          | Results (Retention)          |
| `scale_reliability.csv`              | Internal consistency at baseline (recomputed)                             | Methods (Questionnaires)     |
| `intraclass_correlations.csv`        | Share of variance between persons (recomputed)                            | Results (Retention)          |
| `measurement_invariance.csv`         | Longitudinal measurement invariance                                       | Results (Retention)          |
| `common_method_variance.csv`         | Single-factor test and latent method factor                               | Results (Retention)          |

### 03_within_person_panel

| File                                 | Content                                                                  | Manuscript                                     |
| ------------------------------------ | ------------------------------------------------------------------------ | ---------------------------------------------- |
| `model_fit.csv`                      | Fit of the panel models                                                  | Results (Separating)                           |
| `model_comparisons.csv`              | Likelihood-ratio comparisons of the panel models                         | Results (Separating)                           |
| `riclpm_nine_paths.csv`              | All nine within-person lags of the main RI-CLPM                          | Results (Stress precedes); Table 2; Fig. 2a,b  |
| `lagged_paths_by_model.csv`          | Lagged paths of the main, comparison and alternative-mediator models     | Results (Separating; Indirect effect); Fig. 2c |
| `random_intercept_correlations.csv`  | Correlations among the random intercepts                                 | Results (Separating)                           |
| `between_within_correlations.csv`    | Correlations of person means and person-centred scores (recomputed)      | Results (Separating); Fig. 2d                  |
| `confirmatory_tests.csv`             | The five confirmatory tests with Holm-adjusted P values                  | Table 2                                        |
| `indirect_effect_robustness.csv`     | Indirect effect under alternative outcomes, mediators and specifications | Results (Indirect effect); Fig. 3a             |
| `moderation_tests.csv`               | Interaction, index of moderated mediation and Johnson–Neyman range       | Results (Appraisal moderates); Table 2         |
| `moderation_simple_slopes.csv`       | Simple slopes at −1, 0 and +1 SD of each moderator                       | Results (Appraisal moderates); Fig. 3b,c       |
| `multigroup_tests.csv`               | Focal paths by sex and year of study                                     | Results (Indirect effect)                      |
| `parallel_mediators.csv`             | Alpha and lnRMSSD routes in a parallel model of the core sample          | Results (Indirect effect)                      |
| `school_variance.csv`                | Variance explained by school                                             | Results (Indirect effect)                      |
| `secondary_categorical_outcomes.csv` | Six tests of categorical outcomes at the next wave                       | Results (Stress precedes)                      |

### 04_daily_bursts_and_devices

| File                          | Content                                                            | Manuscript                         |
| ----------------------------- | ------------------------------------------------------------------ | ---------------------------------- |
| `dsem_parameters.csv`         | Parameters of the dynamic structural equation model (2,172 bursts) | Results (Daily); Fig. 4e           |
| `dsem_diagnostics.csv`        | Sampler settings and convergence                                   | Results (Daily); Methods           |
| `daily_multilevel_models.csv` | Multilevel models with observed centring                           | Results (Daily); Fig. 4a,b         |
| `device_agreement.csv`        | Agreement of the ring with Polar H10 and laboratory ECG            | Results (Consumer ring); Fig. 4c,d |
| `device_tracking.csv`         | Ring error and within-person tracking (recomputed)                 | Results (Consumer ring)            |
| `daily_coupling_bins.csv`     | Daily coupling in eight quantile bins of the predictor             | Fig. 4a,b                          |

### 05_profiles_and_states

| File                          | Content                                                                            | Manuscript                           |
| ----------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------ |
| `profile_composition.csv`     | Five-profile solution: size, crisis score by wave, severe-state share (recomputed) | Results (Temporal profiles); Fig. 5b |
| `profile_crisis_by_wave.csv`  | Mean crisis score with 95% CI by profile and wave                                  | Fig. 5b                              |
| `lpa_fit.csv`                 | Latent profile enumeration, one to six classes                                     | Results (Temporal profiles); Fig. 5a |
| `lcga_fit.csv`                | Growth-based enumeration                                                           | Results (Temporal profiles)          |
| `three_profile_solution.csv`  | Class sizes in the three-profile solution                                          | Results (Temporal profiles)          |
| `three_profile_bch_means.csv` | Bias-corrected means in the three-profile solution                                 | Results (Temporal profiles)          |
| `bch_means.csv`               | Bias-corrected three-step means by profile                                         | Results (Temporal profiles); Fig. 5c |
| `bch_contrasts.csv`           | Escalating against stable low profile                                              | Results (Temporal profiles); Fig. 5c |
| `rilta_fit.csv`               | Latent transition enumeration with and without a random intercept                  | Results (Temporal profiles); Fig. 5d |
| `rilta_state_means.csv`       | Indicator values in each of the five states                                        | Results (Temporal profiles)          |
| `rilta_loadings.csv`          | Random-intercept loadings of the five-state model                                  | Methods; Results (Temporal profiles) |
| `rilta_transitions.csv`       | Transition probabilities between consecutive waves                                 | Results (Temporal profiles); Fig. 5e |
| `rilta_prevalence.csv`        | State prevalence at each wave                                                      | Results (Temporal profiles); Fig. 5f |

### 06_prediction

| File                         | Content                                                                  | Manuscript                           |
| ---------------------------- | ------------------------------------------------------------------------ | ------------------------------------ |
| `prediction_performance.csv` | Out-of-sample prediction of the subclinical state at the next wave       | Results (Passive sensing); Fig. 6a,b |
| `net_benefit.csv`            | Net benefit across risk thresholds                                       | Results (Passive sensing); Fig. 6c   |
| `feature_contributions.csv`  | Exact Shapley values for the linear model with all features              | Results (Passive sensing); Fig. 6d   |
| `calibration_deciles.csv`    | Calibration of the model with all features, in deciles of predicted risk | Fig. 6b                              |

### 07_safety

| File                        | Content                                                              | Manuscript                                       |
| --------------------------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| `safety_events_summary.csv` | Risk events by source and level, referrals, response time, treatment | Results (Risk events); Methods (Safety protocol) |

### 08_verification

| File                           | Content                                                                            | Manuscript             |
| ------------------------------ | ---------------------------------------------------------------------------------- | ---------------------- |
| `manuscript_vs_recomputed.csv` | Manuscript values compared with the record-level tables and analysis outputs       | All                    |
| `manuscript_to_data_map.csv`   | For each main result: section, summary file, record-level tables and key variables | All                    |
| `figure_to_data_map.csv`       | For each figure panel and table: the file holding its data                         | All figures and tables |

### 09_codebook

| File                               | Content                                                                | Manuscript |
| ---------------------------------- | ---------------------------------------------------------------------- | ---------- |
| `record_level_tables.csv`          | The 14 record-level tables: unit, rows, variables, manuscript sections | All        |
| `codebook_record_level_tables.csv` | One row per variable of each record-level table                        | All        |
| `codebook_summary_tables.csv`      | One row per column of every summary file                               | All        |

## 4. Locating the headline results

| Manuscript value                                                        | File                                                                  |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 2,000 screened, 1,807 eligible, 1,132 consented, 600 enrolled           | `01_participants/enrolment_flow.csv`                                  |
| Retention 553 / 521 / 498                                               | `01_participants/retention_by_wave.csv`                               |
| Table 1                                                                 | `01_participants/participant_characteristics.csv`                     |
| 95.5% of ECG and 96.5% of EEG sessions valid; ring 68.9%; prompts 64.1% | `02_descriptives_and_quality/data_quality.csv`                        |
| RI-CLPM fit: χ²(51) = 56.4, BIC 22,850.3                                | `03_within_person_panel/model_fit.csv`                                |
| Stress to lnRMSSD β = −0.14; lnRMSSD to crisis β = −0.11                | `03_within_person_panel/riclpm_nine_paths.csv`                        |
| Indirect effect 0.025 [0.005, 0.054]; Holm-adjusted P (Table 2)         | `03_within_person_panel/confirmatory_tests.csv`                       |
| Indirect effect across specifications (Fig. 3a)                         | `03_within_person_panel/indirect_effect_robustness.csv`               |
| Appraisal interaction 0.130; IMM −0.018 [−0.037, −0.003]                | `03_within_person_panel/moderation_tests.csv`                         |
| Daily paths −0.100 and −0.028 (Fig. 4e)                                 | `04_daily_bursts_and_devices/dsem_parameters.csv`                     |
| Ring vs H10 concordance 0.978 on 148 nights                             | `04_daily_bursts_and_devices/device_agreement.csv`                    |
| Five profiles in 507 students; P5 − P1 contrasts                        | `05_profiles_and_states/profile_composition.csv`, `bch_contrasts.csv` |
| Five latent states; S5 prevalence 2.8% to 9.4%                          | `05_profiles_and_states/rilta_fit.csv`, `rilta_prevalence.csv`        |
| Self-report AUC 0.808; passive sensing +0.017 [0.007, 0.026]            | `06_prediction/prediction_performance.csv`                            |
| 540 risk events in 328 students                                         | `07_safety/safety_events_summary.csv`                                 |

## 5. Verification

`08_verification/manuscript_vs_recomputed.csv` compares 207 manuscript values with this package's sources. The `Basis` column states how each value was obtained: recomputed or re-run from the record-level tables, re-fitted in lavaan, or read from the stored analysis output. Result: 204 match, 0 match within rounding, 0 differ, 1 could not be checked because source documents were not inspected, and 2 cannot be verified from the stored files. The `Note` column explains each exception.

`08_verification/figure_to_data_map.csv` names the data file for every figure panel and table. The Bland–Altman panels (Fig. 4c,d) plot individual pairs, which are in the Source Data file submitted with the manuscript; this package holds their summary statistics (`04_daily_bursts_and_devices/device_agreement.csv`).

## 6. Record-level data

The summary values were computed from the 14 record-level tables listed in `09_codebook/record_level_tables.csv`. These tables cover screening, participants, questionnaire items and scores, clinician ratings, laboratory ECG and EEG, nocturnal ring and chest-strap recordings, momentary prompts, smartphone sensing, platform logs and safety events. They are not included in this package.

Access follows the manuscript's Data availability statement: processed feature-level data are to be deposited at [GitHub repository URL to be inserted], and raw physiological recordings and restricted variables are available from the corresponding author on reasonable request. This package itself holds summary-level data only.

The record-level tables contain information that requires controlled access:

- Suicidal ideation and behaviour at item level (`scale_items`, `clinician_assessment`, `ema_prompts`, `screening`) and individual risk events and referrals (`safety_events`).
- Clinical and physiological records for each person (`scale_scores`, `ecg_lab`, `eeg_lab`, `eeg_channel_level`, `wearable_nightly`, `overnight_device_agreement`).
- Daily behaviour from smartphones and academic records (`passive_daily`, `academic_job_logs`), including dated observations.
- Demographics and institution code (`participants`, `screening`).

Person-level model outputs (profile and state assignments, safety alert lists) are treated the same way and are also not included.

Participant identifiers are pseudonymous. No names, student numbers, contact details, raw coordinates or message content are included in any table. Date ranges are not shown in the codebook.
