---
title: "Online Resource 5: NCA-KT, the item-level ladder, DBE-KT22 and the simulated worlds"
subtitle: "When does a concept graph help a knowledge tracer? Null-graph controls, null-calibrated gating and closed-loop tests for graph-based learner models"
author: "Hai Bang Truong, Saigon International University (truonghaibang@siu.edu.vn)"
---

Every value below is computed by the scripts of release 6.0.0 of the accompanying code (`kt/src/r20_*.py`) from the run files of `kt/runs_r20` and `kt/runs_r20_sim` and of the earlier releases; the result files are `kt/results_r20/*.json`. Section numbers continue those of Online Resources 1, 3 and 4. Differences are test AUC of the first condition minus the second, on the same subsample and seed. Intervals: mixed model with a random intercept per corpus (mixed) or ordinary least squares over units where a corpus contributes one unit (ols); corpus-clustered bootstrap; Wilcoxon signed-rank test over corpus means. All terms are exploratory.

# S17 NCA-KT

## S17.1 Summary of every term, first attempts and all predictions

**Table S17.1** NCA-KT terms, first attempts

| Term | Units | Mean | >0 | 95% interval | Fit | Bootstrap 95% | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| nca-empty | 37 | +0.0055 | 10/11 | +0.0023, +0.0079 | mixed | +0.0025, +0.0086 | 0.003 |
| nca-support_emb | 37 | +0.0014 | 8/11 | −0.0014, +0.0044 | mixed | −0.0012, +0.0042 | 0.240 |
| nca-nca_dp (edges) | 15 | +0.0011 | 7/11 | +0.0000, +0.0028 | mixed | −0.0009, +0.0031 | 0.320 |
| support_emb-empty (same units) | 37 | +0.0040 | 10/11 | +0.0015, +0.0053 | mixed | +0.0019, +0.0065 | 0.003 |
| support_emb-dp_emb (same units) | 35 | +0.0004 | 5/10 | −0.0013, +0.0024 | mixed | −0.0014, +0.0021 | 0.625 |
| nca-nca_static | 11 | −0.0001 | 4/11 | −0.0005, +0.0004 | ols | −0.0005, +0.0004 | 0.700 |
| nca-nca_nonull | 11 | −0.0019 | 4/11 | −0.0052, +0.0014 | ols | −0.0055, +0.0004 | 0.365 |
| nca-nca_nogate | 11 | +0.0022 | 8/11 | +0.0004, +0.0040 | ols | +0.0004, +0.0039 | 0.067 |
| nca-nca_nomargin | 11 | +0.0032 | 9/11 | +0.0007, +0.0058 | ols | +0.0010, +0.0058 | 0.024 |
| nca_nonull-empty | 11 | +0.0042 | 10/11 | −0.0002, +0.0087 | ols | −0.0001, +0.0082 | 0.042 |
| empty_fa-empty | 37 | +0.0035 | 9/11 | +0.0006, +0.0066 | mixed | +0.0003, +0.0066 | 0.032 |
| nca-empty_fa | 37 | +0.0020 | 8/11 | −0.0011, +0.0037 | mixed | −0.0003, +0.0046 | 0.032 |
| nca_dp-empty_fa | 15 | −0.0046 | 4/11 | −0.0110, +0.0037 | mixed | −0.0101, +0.0003 | 0.320 |
| nca_nomargin-empty | 11 | −0.0009 | 8/11 | −0.0050, +0.0031 | ols | −0.0051, +0.0026 | 0.638 |
| support_emb-empty (unit 0) | 11 | +0.0026 | 9/11 | +0.0001, +0.0051 | ols | +0.0001, +0.0049 | 0.042 |
| nca_nogate-empty | 11 | +0.0001 | 8/11 | −0.0045, +0.0047 | ols | −0.0047, +0.0040 | 0.365 |
| 100: nca-empty | 11 | +0.0028 | 8/11 | +0.0005, +0.0050 | ols | +0.0006, +0.0048 | 0.042 |
| 100: nca-support | 11 | +0.0010 | 6/11 | −0.0018, +0.0037 | ols | −0.0018, +0.0035 | 0.413 |
| 100: support-empty | 11 | +0.0018 | 9/11 | +0.0004, +0.0032 | ols | +0.0005, +0.0031 | 0.024 |
| 100: empty_fa-empty | 11 | +0.0024 | 8/11 | +0.0001, +0.0047 | ols | +0.0002, +0.0046 | 0.067 |
| 100: nca-empty_fa | 11 | +0.0003 | 6/11 | −0.0026, +0.0033 | ols | −0.0025, +0.0031 | 0.638 |

**Table S17.2** NCA-KT terms, all predictions

| Term | Units | Mean | >0 | 95% interval | Fit | Bootstrap 95% | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| nca-empty | 37 | +0.0002 | 7/11 | −0.0011, +0.0012 | mixed | −0.0011, +0.0014 | 0.700 |
| nca-support_emb | 37 | −0.0022 | 1/11 | −0.0040, −0.0007 | mixed | −0.0038, −0.0010 | 0.002 |
| nca-nca_dp (edges) | 15 | +0.0002 | 6/11 | −0.0021, +0.0025 | mixed | −0.0022, +0.0021 | 0.465 |
| support_emb-empty (same units) | 37 | +0.0024 | 10/11 | +0.0008, +0.0040 | mixed | +0.0010, +0.0039 | 0.002 |
| support_emb-dp_emb (same units) | 35 | +0.0009 | 8/10 | +0.0001, +0.0019 | mixed | +0.0001, +0.0018 | 0.064 |
| nca-nca_static | 11 | −0.0001 | 3/11 | −0.0006, +0.0003 | ols | −0.0006, +0.0003 | 0.465 |
| nca-nca_nonull | 11 | −0.0001 | 5/11 | −0.0020, +0.0018 | ols | −0.0016, +0.0019 | 0.365 |
| nca-nca_nogate | 11 | −0.0012 | 4/11 | −0.0033, +0.0009 | ols | −0.0034, +0.0006 | 0.320 |
| nca-nca_nomargin | 11 | −0.0013 | 4/11 | −0.0035, +0.0008 | ols | −0.0036, +0.0005 | 0.320 |
| nca_nonull-empty | 11 | +0.0008 | 8/11 | −0.0019, +0.0035 | ols | −0.0020, +0.0031 | 0.365 |
| empty_fa-empty | 37 | −0.0068 | 0/11 | −0.0102, −0.0039 | mixed | −0.0100, −0.0041 | 0.001 |
| nca-empty_fa | 37 | +0.0069 | 11/11 | +0.0045, +0.0103 | mixed | +0.0045, +0.0098 | 0.001 |
| nca_dp-empty_fa | 15 | +0.0046 | 11/11 | +0.0036, +0.0063 | mixed | +0.0024, +0.0075 | 0.001 |
| nca_nomargin-empty | 11 | +0.0020 | 9/11 | +0.0008, +0.0032 | ols | +0.0010, +0.0032 | 0.010 |
| support_emb-empty (unit 0) | 11 | +0.0034 | 11/11 | +0.0016, +0.0052 | ols | +0.0019, +0.0053 | 0.001 |
| nca_nogate-empty | 11 | +0.0019 | 9/11 | +0.0005, +0.0033 | ols | +0.0005, +0.0032 | 0.054 |
| 100: nca-empty | 11 | −0.0025 | 0/11 | −0.0036, −0.0013 | ols | −0.0037, −0.0015 | 0.001 |
| 100: nca-support | 11 | −0.0026 | 0/11 | −0.0035, −0.0016 | ols | −0.0035, −0.0017 | 0.001 |
| 100: support-empty | 11 | +0.0001 | 7/11 | −0.0006, +0.0008 | ols | −0.0006, +0.0007 | 0.765 |
| 100: empty_fa-empty | 11 | −0.0040 | 0/11 | −0.0063, −0.0018 | ols | −0.0063, −0.0021 | 0.001 |
| 100: nca-empty_fa | 11 | +0.0016 | 8/11 | +0.0002, +0.0029 | ols | +0.0004, +0.0030 | 0.032 |

## S17.2 By corpus (first attempts)

**Table S17.3** NCA-KT terms by corpus, mean over the units of each corpus

| Corpus | NCA − ∅ | NCA − G (emb.) | NCA(G) − NCA(DP) | − mastery cond. | − calibration | − gate | − margin | Reweighting − ∅ | NCA − reweighting |
|---|---|---|---|---|---|---|---|---|---|
| Algebra 2005 | +0.0108 | +0.0102 | +0.0013 | +0.0001 | −0.0183 | +0.0077 | +0.0107 | +0.0106 | +0.0001 |
| ASSISTments 2009 | +0.0142 | +0.0060 | +0.0047 | −0.0008 | −0.0010 | +0.0021 | +0.0097 | +0.0076 | +0.0067 |
| ASSISTments 2012 | +0.0042 | −0.0012 | −0.0018 | −0.0006 | −0.0000 | +0.0004 | +0.0013 | +0.0007 | +0.0035 |
| ASSISTments 2015 | +0.0073 | +0.0020 | +0.0011 | −0.0005 | +0.0024 | +0.0027 | +0.0008 | −0.0010 | +0.0083 |
| ASSISTments 2017 | +0.0045 | +0.0016 | −0.0055 | −0.0009 | −0.0012 | −0.0005 | −0.0023 | +0.0050 | −0.0005 |
| Bridge to Alg. 2006 | +0.0052 | +0.0024 | +0.0078 | +0.0009 | +0.0017 | +0.0050 | +0.0082 | +0.0060 | −0.0008 |
| DBE-KT22 | +0.0030 | +0.0015 | +0.0022 | −0.0007 | −0.0008 | +0.0036 | +0.0041 | +0.0019 | +0.0011 |
| EdNet KT1 | +0.0027 | +0.0006 | −0.0003 | +0.0006 | +0.0003 | +0.0024 | +0.0007 | +0.0012 | +0.0015 |
| Eedi | +0.0014 | +0.0023 | +0.0002 | −0.0003 | +0.0003 | −0.0004 | +0.0020 | +0.0002 | +0.0012 |
| Junyi Academy | +0.0084 | −0.0040 | +0.0030 | −0.0002 | −0.0037 | −0.0034 | −0.0006 | +0.0092 | −0.0008 |
| Statics 2011 | −0.0017 | −0.0058 | −0.0006 | +0.0017 | −0.0005 | +0.0045 | +0.0010 | −0.0035 | +0.0018 |

**Table S17.4** With all training learners, by corpus (first attempts)

| Corpus | NCA − ∅ | NCA − G | G − ∅ | Reweighting − ∅ | NCA − reweighting |
|---|---|---|---|---|---|
| Algebra 2005 | −0.0034 | −0.0089 | +0.0055 | +0.0067 | −0.0101 |
| ASSISTments 2009 | +0.0045 | +0.0018 | +0.0027 | +0.0085 | −0.0040 |
| ASSISTments 2012 | +0.0061 | +0.0089 | −0.0028 | +0.0049 | +0.0012 |
| ASSISTments 2015 | +0.0057 | +0.0015 | +0.0043 | +0.0020 | +0.0038 |
| ASSISTments 2017 | +0.0038 | +0.0014 | +0.0024 | +0.0021 | +0.0017 |
| Bridge to Alg. 2006 | +0.0076 | +0.0044 | +0.0032 | +0.0064 | +0.0013 |
| DBE-KT22 | −0.0013 | −0.0011 | −0.0003 | −0.0016 | +0.0002 |
| EdNet KT1 | +0.0002 | −0.0002 | +0.0004 | +0.0005 | −0.0003 |
| Eedi | −0.0019 | −0.0028 | +0.0009 | −0.0014 | −0.0005 |
| Junyi Academy | +0.0025 | −0.0007 | +0.0031 | +0.0026 | −0.0001 |
| Statics 2011 | +0.0065 | +0.0060 | +0.0005 | −0.0042 | +0.0107 |

**Table S17.5** Mean gate value alpha at the target, by condition and training size (test predictions; first and repeated attempts)

| cond | frac | alpha | first | repeat |
|---|---|---|---|---|
| nca | 0.1 | 0.323 | 0.327 | 0.322 |
| nca | 1.0 | 0.100 | 0.097 | 0.100 |
| nca_dp | 0.1 | 0.327 | 0.329 | 0.327 |
| nca_expert | 0.1 | 0.292 | 0.308 | 0.285 |
| nca_llm | 0.1 | 0.281 | 0.301 | 0.273 |
| nca_map | 0.1 | 0.299 | 0.300 | 0.298 |
| nca_nogate | 0.1 | 1.000 | 1.000 | 1.000 |
| nca_nomargin | 0.1 | 0.328 | 0.330 | 0.329 |
| nca_nonull | 0.1 | 0.341 | 0.343 | 0.341 |
| nca_static | 0.1 | 0.320 | 0.322 | 0.321 |

# S18 The item-level ladder

**Table S18.1** Item-level terms, first attempts at a KC. Conditions: q_kc KC table only; q_id item table only; q_qm item table and Q-matrix; q_rqm item table and size-preserving random Q-matrix; q_qm_g, q_qm_dp Q-matrix with the support graph or its rewiring; q_diff KC table and item difficulty.

| Term | Units | Mean | >0 | 95% interval | Fit | Bootstrap 95% | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| q_kc-empty | 10 | +0.0004 | 2/7 | −0.0031, +0.0037 | mixed | −0.0026, +0.0035 | 0.938 |
| q_id-q_kc | 10 | −0.0076 | 4/7 | −0.0355, +0.0203 | mixed | −0.0346, +0.0163 | 0.938 |
| q_qm-q_id | 10 | +0.0076 | 5/7 | −0.0019, +0.0170 | mixed | +0.0002, +0.0170 | 0.156 |
| q_qm-q_kc | 10 | −0.0001 | 4/7 | −0.0205, +0.0204 | mixed | −0.0192, +0.0184 | 0.812 |
| q_qm-q_rqm | 10 | +0.0089 | 6/7 | −0.0002, +0.0178 | mixed | +0.0016, +0.0178 | 0.078 |
| q_rqm-q_id | 10 | −0.0013 | 2/7 | −0.0051, +0.0018 | mixed | −0.0045, +0.0016 | 0.297 |
| q_qm_g-q_qm | 10 | +0.0040 | 6/7 | −0.0002, +0.0076 | mixed | +0.0005, +0.0081 | 0.047 |
| q_qm_g-q_qm_dp | 10 | −0.0007 | 4/7 | −0.0027, +0.0012 | mixed | −0.0026, +0.0009 | 1.000 |
| q_qm_dp-q_qm | 10 | +0.0047 | 6/7 | +0.0001, +0.0089 | mixed | +0.0006, +0.0091 | 0.047 |
| q_diff-q_kc | 10 | +0.0061 | 7/7 | +0.0011, +0.0110 | mixed | +0.0025, +0.0111 | 0.016 |
| q_qm-q_diff | 10 | −0.0061 | 4/7 | −0.0235, +0.0114 | mixed | −0.0230, +0.0095 | 0.812 |

**Table S18.2** Item-level terms, first attempts at an item. Conditions: q_kc KC table only; q_id item table only; q_qm item table and Q-matrix; q_rqm item table and size-preserving random Q-matrix; q_qm_g, q_qm_dp Q-matrix with the support graph or its rewiring; q_diff KC table and item difficulty.

| Term | Units | Mean | >0 | 95% interval | Fit | Bootstrap 95% | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| q_id-q_kc | 10 | +0.0057 | 5/7 | −0.0180, +0.0296 | mixed | −0.0167, +0.0266 | 0.688 |
| q_qm-q_id | 10 | +0.0057 | 5/7 | +0.0002, +0.0111 | mixed | +0.0009, +0.0108 | 0.078 |
| q_qm-q_kc | 10 | +0.0114 | 5/7 | −0.0077, +0.0305 | mixed | −0.0064, +0.0285 | 0.469 |
| q_qm-q_rqm | 10 | +0.0074 | 6/7 | +0.0022, +0.0122 | mixed | +0.0029, +0.0120 | 0.031 |
| q_rqm-q_id | 10 | −0.0017 | 2/7 | −0.0038, −0.0002 | mixed | −0.0031, −0.0003 | 0.078 |
| q_qm_g-q_qm | 10 | +0.0022 | 5/7 | −0.0006, +0.0047 | mixed | −0.0002, +0.0048 | 0.219 |
| q_qm_g-q_qm_dp | 10 | +0.0000 | 4/7 | −0.0019, +0.0018 | mixed | −0.0018, +0.0014 | 0.688 |
| q_qm_dp-q_qm | 10 | +0.0022 | 5/7 | −0.0002, +0.0046 | mixed | +0.0002, +0.0045 | 0.109 |
| q_diff-q_kc | 10 | +0.0072 | 7/7 | +0.0021, +0.0122 | mixed | +0.0036, +0.0123 | 0.016 |
| q_qm-q_diff | 10 | +0.0042 | 5/7 | −0.0118, +0.0204 | mixed | −0.0115, +0.0183 | 0.812 |

**Table S18.3** Item-level terms, all predictions. Conditions: q_kc KC table only; q_id item table only; q_qm item table and Q-matrix; q_rqm item table and size-preserving random Q-matrix; q_qm_g, q_qm_dp Q-matrix with the support graph or its rewiring; q_diff KC table and item difficulty.

| Term | Units | Mean | >0 | 95% interval | Fit | Bootstrap 95% | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| q_kc-empty | 10 | +0.0004 | 3/7 | −0.0025, +0.0036 | mixed | −0.0016, +0.0035 | 1.000 |
| q_id-q_kc | 10 | +0.0059 | 4/7 | −0.0179, +0.0298 | mixed | −0.0164, +0.0268 | 0.812 |
| q_qm-q_id | 10 | +0.0058 | 5/7 | +0.0003, +0.0112 | mixed | +0.0011, +0.0109 | 0.078 |
| q_qm-q_kc | 10 | +0.0117 | 5/7 | −0.0074, +0.0309 | mixed | −0.0061, +0.0289 | 0.375 |
| q_qm-q_rqm | 10 | +0.0073 | 6/7 | +0.0017, +0.0117 | mixed | +0.0026, +0.0120 | 0.031 |
| q_rqm-q_id | 10 | −0.0015 | 2/7 | −0.0036, +0.0001 | mixed | −0.0029, +0.0000 | 0.078 |
| q_qm_g-q_qm | 10 | +0.0021 | 5/7 | −0.0006, +0.0046 | mixed | −0.0003, +0.0045 | 0.219 |
| q_qm_g-q_qm_dp | 10 | +0.0000 | 4/7 | −0.0019, +0.0018 | mixed | −0.0017, +0.0014 | 0.688 |
| q_qm_dp-q_qm | 10 | +0.0021 | 5/7 | −0.0003, +0.0043 | mixed | +0.0002, +0.0043 | 0.078 |
| q_diff-q_kc | 10 | +0.0072 | 7/7 | +0.0022, +0.0123 | mixed | +0.0037, +0.0124 | 0.016 |
| q_qm-q_diff | 10 | +0.0045 | 5/7 | −0.0117, +0.0206 | mixed | −0.0114, +0.0187 | 0.688 |

**Table S18.4** Test AUC by corpus and condition, first attempts at a KC (mean over the subsamples run)

| Corpus | q_diff | q_id | q_kc | q_qm | q_qm_dp | q_qm_g | q_rqm |
|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 0.7169 | 0.6413 | 0.7152 | 0.6751 | 0.6871 | 0.6834 | 0.6416 |
| ASSISTments 2012 | 0.6760 | 0.6302 | 0.6706 | 0.6423 | 0.6574 | 0.6568 | 0.6275 |
| ASSISTments 2017 | 0.7077 | 0.7106 | 0.7023 | 0.7128 | 0.7139 | 0.7152 | 0.7101 |
| DBE-KT22 | 0.7947 | 0.8070 | 0.7936 | 0.8067 | 0.8078 | 0.8085 | 0.8062 |
| EdNet KT1 | 0.6817 | 0.6753 | 0.6750 | 0.6831 | 0.6838 | 0.6848 | 0.6744 |
| Eedi | 0.7448 | 0.7649 | 0.7245 | 0.7671 | 0.7664 | 0.7678 | 0.7698 |
| Statics 2011 | 0.7714 | 0.7681 | 0.7696 | 0.7633 | 0.7668 | 0.7618 | 0.7587 |

**Table S18.5** Test AUC by corpus and condition, first attempts at an item (mean over the subsamples run)

| Corpus | q_diff | q_id | q_kc | q_qm | q_qm_dp | q_qm_g | q_rqm |
|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 0.7382 | 0.6963 | 0.7364 | 0.7148 | 0.7190 | 0.7213 | 0.6970 |
| ASSISTments 2012 | 0.7084 | 0.6697 | 0.7039 | 0.6804 | 0.6891 | 0.6882 | 0.6671 |
| ASSISTments 2017 | 0.7274 | 0.7382 | 0.7206 | 0.7411 | 0.7412 | 0.7429 | 0.7361 |
| DBE-KT22 | 0.7796 | 0.7979 | 0.7768 | 0.7961 | 0.7964 | 0.7968 | 0.7962 |
| EdNet KT1 | 0.6820 | 0.6760 | 0.6751 | 0.6845 | 0.6844 | 0.6855 | 0.6734 |
| Eedi | 0.7472 | 0.7703 | 0.7252 | 0.7723 | 0.7722 | 0.7722 | 0.7708 |
| Statics 2011 | 0.7640 | 0.7886 | 0.7589 | 0.7872 | 0.7895 | 0.7849 | 0.7843 |

**Table S18.6** Test AUC by corpus and condition, all predictions (mean over the subsamples run)

| Corpus | q_diff | q_id | q_kc | q_qm | q_qm_dp | q_qm_g | q_rqm |
|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 0.7378 | 0.6960 | 0.7359 | 0.7145 | 0.7186 | 0.7210 | 0.6967 |
| ASSISTments 2012 | 0.7095 | 0.6717 | 0.7050 | 0.6822 | 0.6905 | 0.6895 | 0.6691 |
| ASSISTments 2017 | 0.7064 | 0.7176 | 0.6995 | 0.7214 | 0.7213 | 0.7229 | 0.7168 |
| DBE-KT22 | 0.7790 | 0.7976 | 0.7762 | 0.7958 | 0.7961 | 0.7966 | 0.7959 |
| EdNet KT1 | 0.6802 | 0.6722 | 0.6727 | 0.6812 | 0.6813 | 0.6825 | 0.6699 |
| Eedi | 0.7472 | 0.7703 | 0.7252 | 0.7723 | 0.7722 | 0.7722 | 0.7708 |
| Statics 2011 | 0.7650 | 0.7905 | 0.7599 | 0.7889 | 0.7907 | 0.7864 | 0.7862 |

**Table S18.7** Share of test answers on items that no training learner of the 10% subsample answered

| Corpus | Unseen items |
|---|---|
| ASSISTments 2009 | 29.8% |
| ASSISTments 2012 | 24.8% |
| ASSISTments 2017 | 0.8% |
| DBE-KT22 | 0.0% |
| EdNet KT1 | 9.1% |
| Eedi | 0.3% |
| Statics 2011 | 1.8% |

# S19 DBE-KT22

Five subsamples at 10% of the training learners, one seed each; *t* intervals over the five subsamples. The instructor relations of the release (KC_Relationships.csv) list every pair in both directions and are oriented here from the KC the training learners meet earlier to the one they meet later. The generated graph (data/dbe_kt22_llm.json) has 87 directed prerequisite pairs among the 59 KCs of the prepared corpus.

**Table S19.1** DBE-KT22 terms

| Term | Outcome | Units | Mean | >0 | 95% t interval |
|---|---|---|---|---|---|
| support-empty | first | 5 | +0.0015 | 3/5 | −0.0021, +0.0050 |
| support-empty | all | 5 | +0.0003 | 3/5 | −0.0030, +0.0036 |
| support_emb-empty | first | 5 | +0.0015 | 3/5 | −0.0018, +0.0049 |
| support_emb-empty | all | 5 | +0.0001 | 3/5 | −0.0034, +0.0035 |
| uniform_emb-empty | first | 5 | +0.0000 | 2/5 | −0.0026, +0.0027 |
| uniform_emb-empty | all | 5 | +0.0000 | 4/5 | −0.0024, +0.0025 |
| er_emb-uniform_emb | first | 5 | +0.0024 | 5/5 | −0.0004, +0.0051 |
| er_emb-uniform_emb | all | 5 | +0.0013 | 4/5 | −0.0013, +0.0040 |
| dp_emb-er_emb | first | 5 | −0.0001 | 1/5 | −0.0035, +0.0034 |
| dp_emb-er_emb | all | 5 | −0.0008 | 2/5 | −0.0038, +0.0023 |
| support_emb-dp_emb | first | 5 | −0.0008 | 2/5 | −0.0045, +0.0029 |
| support_emb-dp_emb | all | 5 | −0.0005 | 1/5 | −0.0024, +0.0013 |
| st_support-empty | first | 5 | +0.0006 | 4/5 | −0.0009, +0.0021 |
| st_support-empty | all | 5 | +0.0030 | 4/5 | −0.0007, +0.0068 |
| st_support-st_dp | first | 5 | +0.0012 | 5/5 | −0.0001, +0.0024 |
| st_support-st_dp | all | 5 | +0.0005 | 3/5 | −0.0004, +0.0015 |
| st_er-empty | first | 5 | −0.0005 | 1/5 | −0.0021, +0.0011 |
| st_er-empty | all | 5 | +0.0020 | 4/5 | −0.0017, +0.0057 |
| expert_emb-empty | first | 5 | +0.0019 | 3/5 | −0.0014, +0.0052 |
| expert_emb-empty | all | 5 | +0.0002 | 3/5 | −0.0019, +0.0022 |
| expert_emb-expert_dp_emb | first | 5 | −0.0012 | 2/5 | −0.0045, +0.0022 |
| expert_emb-expert_dp_emb | all | 5 | −0.0013 | 1/5 | −0.0035, +0.0010 |
| expert_emb-support_emb | first | 5 | +0.0003 | 2/5 | −0.0048, +0.0055 |
| expert_emb-support_emb | all | 5 | +0.0001 | 1/5 | −0.0019, +0.0021 |
| llm_emb-empty | first | 5 | +0.0003 | 2/5 | −0.0032, +0.0038 |
| llm_emb-empty | all | 5 | +0.0000 | 2/5 | −0.0034, +0.0035 |
| llm_emb-llm_dp_emb | first | 5 | −0.0010 | 2/5 | −0.0037, +0.0017 |
| llm_emb-llm_dp_emb | all | 5 | −0.0007 | 2/5 | −0.0023, +0.0009 |
| llm_emb-support_emb | first | 5 | −0.0012 | 1/5 | −0.0062, +0.0037 |
| llm_emb-support_emb | all | 5 | −0.0000 | 3/5 | −0.0022, +0.0021 |
| llm_emb-expert_emb | first | 5 | −0.0015 | 1/5 | −0.0059, +0.0028 |
| llm_emb-expert_emb | all | 5 | −0.0001 | 2/5 | −0.0025, +0.0022 |
| nca-empty | first | 5 | +0.0030 | 3/5 | −0.0031, +0.0091 |
| nca-empty | all | 5 | +0.0005 | 3/5 | −0.0040, +0.0049 |
| nca-nca_dp | first | 5 | +0.0022 | 2/5 | −0.0038, +0.0082 |
| nca-nca_dp | all | 5 | −0.0001 | 2/5 | −0.0009, +0.0007 |
| nca_expert-empty | first | 5 | +0.0004 | 2/5 | −0.0029, +0.0037 |
| nca_expert-empty | all | 5 | −0.0009 | 2/5 | −0.0059, +0.0041 |
| nca_llm-empty | first | 5 | −0.0005 | 2/5 | −0.0049, +0.0039 |
| nca_llm-empty | all | 5 | −0.0010 | 2/5 | −0.0070, +0.0051 |
| nca_expert-nca | first | 5 | −0.0026 | 2/5 | −0.0086, +0.0035 |
| nca_expert-nca | all | 5 | −0.0013 | 1/5 | −0.0031, +0.0004 |

**Table S19.2** Every DBE-KT22 run at 10%

| Subsample | Condition | Test AUC | First att. | Updates |
|---|---|---|---|---|
| 0 | dp_emb | 0.7788 | 0.8032 | 250 |
| 1 | dp_emb | 0.7743 | 0.7959 | 250 |
| 2 | dp_emb | 0.7746 | 0.7978 | 250 |
| 3 | dp_emb | 0.7754 | 0.8025 | 250 |
| 4 | dp_emb | 0.7784 | 0.8028 | 250 |
| 0 | empty | 0.7749 | 0.7987 | 250 |
| 1 | empty | 0.7782 | 0.7914 | 275 |
| 2 | empty | 0.7729 | 0.7945 | 250 |
| 3 | empty | 0.7728 | 0.8028 | 250 |
| 4 | empty | 0.7797 | 0.8031 | 275 |
| 0 | empty_fa | 0.7708 | 0.8038 | 250 |
| 1 | empty_fa | 0.7729 | 0.7999 | 250 |
| 2 | empty_fa | 0.7701 | 0.7985 | 250 |
| 3 | empty_fa | 0.7673 | 0.7909 | 275 |
| 4 | empty_fa | 0.7715 | 0.8071 | 250 |
| 0 | er_emb | 0.7768 | 0.7987 | 250 |
| 1 | er_emb | 0.7782 | 0.7963 | 250 |
| 2 | er_emb | 0.7766 | 0.7995 | 250 |
| 3 | er_emb | 0.7740 | 0.8026 | 250 |
| 4 | er_emb | 0.7797 | 0.8055 | 250 |
| 0 | expert_dp_emb | 0.7803 | 0.8022 | 250 |
| 1 | expert_dp_emb | 0.7767 | 0.7985 | 250 |
| 2 | expert_dp_emb | 0.7733 | 0.7968 | 250 |
| 3 | expert_dp_emb | 0.7756 | 0.8047 | 250 |
| 4 | expert_dp_emb | 0.7796 | 0.8034 | 250 |
| 0 | expert_emb | 0.7764 | 0.7974 | 275 |
| 1 | expert_emb | 0.7760 | 0.7967 | 250 |
| 2 | expert_emb | 0.7743 | 0.7976 | 250 |
| 3 | expert_emb | 0.7737 | 0.8025 | 250 |
| 4 | expert_emb | 0.7789 | 0.8055 | 250 |
| 0 | llm_dp_emb | 0.7793 | 0.8018 | 250 |
| 1 | llm_dp_emb | 0.7772 | 0.7971 | 250 |
| 2 | llm_dp_emb | 0.7714 | 0.7930 | 250 |
| 3 | llm_dp_emb | 0.7744 | 0.8008 | 250 |
| 4 | llm_dp_emb | 0.7798 | 0.8046 | 250 |
| 0 | llm_emb | 0.7782 | 0.8011 | 250 |
| 1 | llm_emb | 0.7756 | 0.7954 | 250 |
| 2 | llm_emb | 0.7716 | 0.7937 | 250 |
| 3 | llm_emb | 0.7754 | 0.8017 | 250 |
| 4 | llm_emb | 0.7778 | 0.8001 | 250 |
| 0 | nca | 0.7803 | 0.8037 | 275 |
| 1 | nca | 0.7737 | 0.8009 | 250 |
| 2 | nca | 0.7741 | 0.7987 | 250 |
| 3 | nca | 0.7737 | 0.8023 | 275 |
| 4 | nca | 0.7791 | 0.8000 | 275 |
| 0 | nca_dp | 0.7804 | 0.8016 | 275 |
| 1 | nca_dp | 0.7734 | 0.7903 | 275 |
| 2 | nca_dp | 0.7734 | 0.7991 | 250 |
| 3 | nca_dp | 0.7748 | 0.8023 | 275 |
| 4 | nca_dp | 0.7792 | 0.8012 | 275 |
| 0 | nca_expert | 0.7796 | 0.8004 | 275 |
| 1 | nca_expert | 0.7718 | 0.7901 | 275 |
| 2 | nca_expert | 0.7722 | 0.7990 | 250 |
| 3 | nca_expert | 0.7706 | 0.8017 | 275 |
| 4 | nca_expert | 0.7798 | 0.8014 | 275 |
| 0 | nca_llm | 0.7801 | 0.8021 | 275 |
| 1 | nca_llm | 0.7737 | 0.7884 | 275 |
| 2 | nca_llm | 0.7725 | 0.7973 | 250 |
| 3 | nca_llm | 0.7746 | 0.7980 | 275 |
| 4 | nca_llm | 0.7729 | 0.8020 | 250 |
| 0 | nca_nogate | 0.7764 | 0.8001 | 250 |
| 0 | nca_nomargin | 0.7773 | 0.7996 | 275 |
| 0 | nca_nonull | 0.7818 | 0.8045 | 275 |
| 0 | nca_static | 0.7792 | 0.8044 | 275 |
| 0 | q_diff | 0.7797 | 0.7976 | 275 |
| 1 | q_diff | 0.7783 | 0.7918 | 275 |
| 0 | q_id | 0.7973 | 0.8073 | 250 |
| 1 | q_id | 0.7980 | 0.8066 | 250 |
| 0 | q_kc | 0.7768 | 0.7964 | 275 |
| 1 | q_kc | 0.7756 | 0.7908 | 275 |
| 0 | q_qm | 0.7958 | 0.8085 | 250 |
| 1 | q_qm | 0.7958 | 0.8050 | 250 |
| 0 | q_qm_dp | 0.7960 | 0.8090 | 250 |
| 1 | q_qm_dp | 0.7962 | 0.8066 | 250 |
| 0 | q_qm_g | 0.7962 | 0.8093 | 250 |
| 1 | q_qm_g | 0.7969 | 0.8077 | 250 |
| 0 | q_rqm | 0.7970 | 0.8091 | 250 |
| 1 | q_rqm | 0.7949 | 0.8032 | 250 |
| 0 | st_dp | 0.7778 | 0.7981 | 275 |
| 1 | st_dp | 0.7771 | 0.7897 | 275 |
| 2 | st_dp | 0.7759 | 0.7963 | 250 |
| 3 | st_dp | 0.7789 | 0.8012 | 275 |
| 4 | st_dp | 0.7812 | 0.8023 | 275 |
| 0 | st_er | 0.7779 | 0.7977 | 275 |
| 1 | st_er | 0.7771 | 0.7899 | 275 |
| 2 | st_er | 0.7743 | 0.7962 | 250 |
| 3 | st_er | 0.7793 | 0.8018 | 275 |
| 4 | st_er | 0.7798 | 0.8025 | 275 |
| 0 | st_support | 0.7788 | 0.7989 | 275 |
| 1 | st_support | 0.7779 | 0.7924 | 275 |
| 2 | st_support | 0.7758 | 0.7966 | 250 |
| 3 | st_support | 0.7804 | 0.8016 | 275 |
| 4 | st_support | 0.7808 | 0.8040 | 275 |
| 0 | support | 0.7777 | 0.8016 | 250 |
| 1 | support | 0.7765 | 0.7912 | 275 |
| 2 | support | 0.7744 | 0.8001 | 250 |
| 3 | support | 0.7749 | 0.8035 | 250 |
| 4 | support | 0.7764 | 0.8015 | 250 |
| 0 | support_emb | 0.7776 | 0.8012 | 250 |
| 1 | support_emb | 0.7761 | 0.7908 | 275 |
| 2 | support_emb | 0.7744 | 0.8002 | 250 |
| 3 | support_emb | 0.7746 | 0.8039 | 250 |
| 4 | support_emb | 0.7761 | 0.8021 | 250 |
| 0 | uniform_emb | 0.7761 | 0.7984 | 250 |
| 1 | uniform_emb | 0.7797 | 0.7941 | 275 |
| 2 | uniform_emb | 0.7730 | 0.7959 | 250 |
| 3 | uniform_emb | 0.7735 | 0.8023 | 250 |
| 4 | uniform_emb | 0.7764 | 0.8001 | 250 |

# S20 Simulated worlds and the closed loop

World parameters are in data/<world>_world.npz and the true graphs in data/<world>_true.json (kt/src/r20_sim.py). Training: 3,000 simulated learners, the hash split of the study (2,100 training learners), subsample 0 at 10% and all training learners. Closed loop: 500 new learners (seed 777), 20 warm-up steps, 40 adaptive steps. The score check is the largest absolute difference between the all-KC scores used by the policies and the model's own forward pass on three random histories (it is at most 1e-7 for every model).

**Table S20.1** DBE-KT22 world: test AUC by condition (mean over seeds)

| Size and condition | Seeds | First att. | All pred. |
|---|---|---|---|
| 0.1:empty | 2 | 0.6987 | 0.7603 |
| 0.1:support_emb | 2 | 0.7001 | 0.7610 |
| 0.1:er_emb | 2 | 0.6997 | 0.7607 |
| 0.1:dp_emb | 2 | 0.6990 | 0.7610 |
| 0.1:true_emb | 2 | 0.7037 | 0.7622 |
| 0.1:true_dp_emb | 2 | 0.6999 | 0.7612 |
| 0.1:nca | 2 | 0.7017 | 0.7566 |
| 0.1:nca_dp | 2 | 0.7017 | 0.7573 |
| 0.1:nca_true | 2 | 0.7021 | 0.7578 |
| 0.1:st_true | 2 | 0.7027 | 0.7625 |
| 0.1:st_true_dp | 2 | 0.6993 | 0.7598 |
| 1.0:empty | 1 | 0.7222 | 0.7733 |
| 1.0:support_emb | 1 | 0.7229 | 0.7738 |
| 1.0:er_emb | 1 | 0.7234 | 0.7740 |
| 1.0:dp_emb | 1 | 0.7229 | 0.7740 |
| 1.0:true_emb | 1 | 0.7226 | 0.7740 |
| 1.0:true_dp_emb | 1 | 0.7230 | 0.7739 |
| 1.0:nca | 1 | 0.7263 | 0.7731 |
| 1.0:nca_dp | 1 | 0.7266 | 0.7732 |
| 1.0:nca_true | 1 | 0.7265 | 0.7728 |

**Table S20.3** DBE-KT22 world: prediction terms

| Term | First att. | All pred. | Per seed (first) |
|---|---|---|---|
| 0.1:true_emb-true_dp_emb | +0.0038 | +0.0010 | +0.0035, +0.0040 |
| 0.1:true_emb-empty | +0.0050 | +0.0019 | +0.0061, +0.0039 |
| 0.1:support_emb-empty | +0.0014 | +0.0008 | +0.0016, +0.0012 |
| 0.1:support_emb-dp_emb | +0.0011 | +0.0001 | +0.0004, +0.0017 |
| 0.1:er_emb-empty | +0.0010 | +0.0004 | +0.0016, +0.0005 |
| 0.1:true_emb-support_emb | +0.0036 | +0.0011 | +0.0045, +0.0026 |
| 0.1:nca_true-empty | +0.0034 | −0.0025 | +0.0037, +0.0030 |
| 0.1:nca-nca_dp | +0.0001 | −0.0007 | −0.0003, +0.0004 |
| 0.1:nca_true-true_emb | −0.0016 | −0.0044 | −0.0024, −0.0008 |
| 0.1:nca-empty | +0.0031 | −0.0037 | +0.0017, +0.0044 |
| 0.1:st_true-st_true_dp | +0.0034 | +0.0027 | +0.0025, +0.0043 |
| 0.1:st_true-empty | +0.0040 | +0.0022 | +0.0029, +0.0051 |
| 1.0:true_emb-true_dp_emb | −0.0003 | +0.0001 | −0.0003 |
| 1.0:true_emb-empty | +0.0005 | +0.0007 | +0.0005 |
| 1.0:support_emb-empty | +0.0008 | +0.0006 | +0.0008 |
| 1.0:support_emb-dp_emb | −0.0000 | −0.0002 | −0.0000 |
| 1.0:er_emb-empty | +0.0012 | +0.0007 | +0.0012 |
| 1.0:true_emb-support_emb | −0.0003 | +0.0002 | −0.0003 |
| 1.0:nca_true-empty | +0.0044 | −0.0004 | +0.0044 |
| 1.0:nca-nca_dp | −0.0003 | −0.0000 | −0.0003 |
| 1.0:nca_true-true_emb | +0.0039 | −0.0012 | +0.0039 |
| 1.0:nca-empty | +0.0041 | −0.0001 | +0.0041 |

**Table S20.5** DBE-KT22 world: reference policies (gain in KCs; ttm5 steps to five new masteries, 41 if never; reached5 share of learners reaching five; frustration; success rate; regret of the target rule under the true model)

| Policy | gain | ttm5 | reached5 | frustration | success | regret |
|---|---|---|---|---|---|---|
| oracle | -2.406 | 41.000 | 0.000 | 0.003 | 0.853 | 0.000 |
| oracle_frontier | 7.776 | 24.246 | 0.754 | 0.453 | 0.223 | 0.222 |
| curriculum | -1.010 | 40.936 | 0.008 | 0.081 | 0.703 | 0.095 |
| random | -0.616 | 40.704 | 0.024 | 0.070 | 0.594 | 0.121 |

**Table S20.7** DBE-KT22 world: tracer-driven policies (first_choice_agreement and jaccard10 between the two seeds of a tracer: same first adaptive choice; overlap of the first ten KCs practised)

| Size:tracer:rule | gain | ttm5 | frustration | success | regret | first_choice_agreement | jaccard10 | Seeds |
|---|---|---|---|---|---|---|---|---|
| 1.0:dp_emb:target | -0.428 | 40.686 | 0.022 | 0.685 | 0.094 | -- | -- | 1 |
| 1.0:dp_emb:stable | -0.180 | 40.690 | 0.021 | 0.685 | 0.093 | -- | -- | 1 |
| 1.0:dp_emb:frontier | -1.264 | 40.976 | 0.006 | 0.803 | 0.067 | -- | -- | 1 |
| 1.0:empty:target | -0.254 | 40.500 | 0.021 | 0.682 | 0.098 | -- | -- | 1 |
| 1.0:empty:stable | -0.148 | 40.712 | 0.022 | 0.682 | 0.098 | -- | -- | 1 |
| 1.0:empty:frontier | -1.296 | 40.992 | 0.006 | 0.797 | 0.068 | -- | -- | 1 |
| 1.0:er_emb:target | -0.342 | 40.522 | 0.021 | 0.676 | 0.099 | -- | -- | 1 |
| 1.0:er_emb:stable | -0.122 | 40.474 | 0.022 | 0.675 | 0.098 | -- | -- | 1 |
| 1.0:er_emb:frontier | -1.282 | 40.994 | 0.007 | 0.792 | 0.068 | -- | -- | 1 |
| 1.0:nca:target | -0.558 | 40.862 | 0.018 | 0.707 | 0.088 | -- | -- | 1 |
| 1.0:nca:stable | -0.570 | 40.774 | 0.017 | 0.711 | 0.086 | -- | -- | 1 |
| 1.0:nca:frontier | -1.534 | 40.988 | 0.005 | 0.817 | 0.065 | -- | -- | 1 |
| 1.0:nca_dp:target | -0.566 | 40.832 | 0.017 | 0.708 | 0.088 | -- | -- | 1 |
| 1.0:nca_dp:stable | -0.514 | 40.830 | 0.016 | 0.707 | 0.087 | -- | -- | 1 |
| 1.0:nca_dp:frontier | -1.478 | 41.000 | 0.007 | 0.812 | 0.066 | -- | -- | 1 |
| 1.0:nca_true:target | -0.426 | 40.736 | 0.021 | 0.687 | 0.093 | -- | -- | 1 |
| 1.0:nca_true:stable | -0.214 | 40.742 | 0.021 | 0.680 | 0.094 | -- | -- | 1 |
| 1.0:nca_true:frontier | -1.440 | 40.998 | 0.006 | 0.802 | 0.065 | -- | -- | 1 |
| 1.0:support_emb:target | -0.262 | 40.544 | 0.020 | 0.681 | 0.096 | -- | -- | 1 |
| 1.0:support_emb:stable | -0.174 | 40.734 | 0.020 | 0.681 | 0.098 | -- | -- | 1 |
| 1.0:support_emb:frontier | -1.284 | 40.978 | 0.007 | 0.790 | 0.069 | -- | -- | 1 |
| 1.0:true_dp_emb:target | -0.338 | 40.504 | 0.022 | 0.666 | 0.102 | -- | -- | 1 |
| 1.0:true_dp_emb:stable | -0.174 | 40.728 | 0.026 | 0.668 | 0.101 | -- | -- | 1 |
| 1.0:true_dp_emb:frontier | -1.282 | 40.972 | 0.006 | 0.792 | 0.072 | -- | -- | 1 |
| 1.0:true_emb:target | -0.444 | 40.682 | 0.020 | 0.693 | 0.091 | -- | -- | 1 |
| 1.0:true_emb:stable | -0.334 | 40.742 | 0.022 | 0.689 | 0.092 | -- | -- | 1 |
| 1.0:true_emb:frontier | -1.514 | 41.000 | 0.006 | 0.811 | 0.059 | -- | -- | 1 |
| 0.1:dp_emb:target | -0.522 | 40.738 | 0.021 | 0.677 | 0.099 | 0.066 | 0.180 | 2 |
| 0.1:dp_emb:stable | -0.514 | 40.805 | 0.015 | 0.704 | 0.094 | 0.126 | 0.286 | 2 |
| 0.1:dp_emb:frontier | -1.317 | 40.999 | 0.007 | 0.804 | 0.070 | 0.112 | 0.212 | 2 |
| 0.1:empty:target | -0.613 | 40.795 | 0.022 | 0.672 | 0.100 | 0.054 | 0.163 | 2 |
| 0.1:empty:stable | -0.535 | 40.880 | 0.014 | 0.698 | 0.095 | 0.152 | 0.289 | 2 |
| 0.1:empty:frontier | -1.308 | 40.988 | 0.008 | 0.794 | 0.073 | 0.036 | 0.190 | 2 |
| 0.1:er_emb:target | -0.549 | 40.767 | 0.020 | 0.678 | 0.100 | 0.058 | 0.170 | 2 |
| 0.1:er_emb:stable | -0.437 | 40.764 | 0.015 | 0.697 | 0.095 | 0.138 | 0.290 | 2 |
| 0.1:er_emb:frontier | -1.375 | 40.993 | 0.007 | 0.802 | 0.070 | 0.068 | 0.202 | 2 |
| 0.1:nca:target | -0.854 | 40.945 | 0.019 | 0.681 | 0.096 | 0.048 | 0.155 | 2 |
| 0.1:nca:stable | -0.750 | 40.922 | 0.012 | 0.724 | 0.086 | 0.118 | 0.258 | 2 |
| 0.1:nca:frontier | -1.346 | 40.994 | 0.007 | 0.790 | 0.073 | 0.074 | 0.181 | 2 |
| 0.1:nca_dp:target | -0.806 | 40.888 | 0.019 | 0.680 | 0.095 | 0.050 | 0.168 | 2 |
| 0.1:nca_dp:stable | -0.733 | 40.848 | 0.012 | 0.721 | 0.088 | 0.122 | 0.275 | 2 |
| 0.1:nca_dp:frontier | -1.318 | 40.975 | 0.006 | 0.789 | 0.073 | 0.086 | 0.191 | 2 |
| 0.1:nca_true:target | -0.737 | 40.869 | 0.019 | 0.676 | 0.097 | 0.044 | 0.162 | 2 |
| 0.1:nca_true:stable | -0.649 | 40.850 | 0.013 | 0.713 | 0.089 | 0.144 | 0.282 | 2 |
| 0.1:nca_true:frontier | -1.333 | 41.000 | 0.007 | 0.787 | 0.073 | 0.096 | 0.195 | 2 |
| 0.1:support_emb:target | -0.439 | 40.674 | 0.022 | 0.665 | 0.102 | 0.054 | 0.171 | 2 |
| 0.1:support_emb:stable | -0.489 | 40.736 | 0.015 | 0.695 | 0.095 | 0.148 | 0.291 | 2 |
| 0.1:support_emb:frontier | -1.270 | 40.940 | 0.006 | 0.798 | 0.071 | 0.074 | 0.209 | 2 |
| 0.1:true_dp_emb:target | -0.544 | 40.715 | 0.021 | 0.670 | 0.101 | 0.042 | 0.165 | 2 |
| 0.1:true_dp_emb:stable | -0.469 | 40.746 | 0.018 | 0.693 | 0.096 | 0.162 | 0.295 | 2 |
| 0.1:true_dp_emb:frontier | -1.382 | 40.990 | 0.008 | 0.801 | 0.070 | 0.064 | 0.206 | 2 |
| 0.1:true_emb:target | -0.410 | 40.705 | 0.019 | 0.687 | 0.098 | 0.086 | 0.207 | 2 |
| 0.1:true_emb:stable | -0.353 | 40.756 | 0.015 | 0.699 | 0.094 | 0.178 | 0.315 | 2 |
| 0.1:true_emb:frontier | -1.387 | 40.997 | 0.006 | 0.799 | 0.067 | 0.108 | 0.228 | 2 |

**Table S20.9** DBE-KT22 world: paired differences per learner (seed-averaged; interval over learners)

| Pair:rule:outcome | Mean | 95% interval |
|---|---|---|
| 0.1:true_emb-true_dp_emb:target:gain | +0.134 | +0.052, +0.216 |
| 0.1:true_emb-true_dp_emb:target:regret | −0.003 | −0.004, −0.002 |
| 0.1:true_emb-true_dp_emb:target:frus | −0.002 | −0.004, +0.000 |
| 0.1:true_emb-true_dp_emb:stable:gain | +0.116 | +0.036, +0.196 |
| 0.1:true_emb-true_dp_emb:stable:regret | −0.002 | −0.003, −0.001 |
| 0.1:true_emb-true_dp_emb:stable:frus | −0.003 | −0.005, −0.001 |
| 0.1:true_emb-true_dp_emb:frontier:gain | −0.005 | −0.063, +0.053 |
| 0.1:true_emb-true_dp_emb:frontier:regret | −0.003 | −0.004, −0.002 |
| 0.1:true_emb-true_dp_emb:frontier:frus | −0.002 | −0.004, −0.001 |
| 0.1:true_emb-empty:target:gain | +0.203 | +0.122, +0.284 |
| 0.1:true_emb-empty:target:regret | −0.002 | −0.003, −0.001 |
| 0.1:true_emb-empty:target:frus | −0.002 | −0.004, −0.000 |
| 0.1:true_emb-empty:stable:gain | +0.182 | +0.106, +0.258 |
| 0.1:true_emb-empty:stable:regret | −0.001 | −0.002, +0.000 |
| 0.1:true_emb-empty:stable:frus | +0.001 | −0.001, +0.002 |
| 0.1:true_emb-empty:frontier:gain | −0.079 | −0.142, −0.016 |
| 0.1:true_emb-empty:frontier:regret | −0.005 | −0.006, −0.004 |
| 0.1:true_emb-empty:frontier:frus | −0.003 | −0.004, −0.001 |
| 0.1:support_emb-empty:target:gain | +0.174 | +0.093, +0.255 |
| 0.1:support_emb-empty:target:regret | +0.002 | +0.001, +0.003 |
| 0.1:support_emb-empty:target:frus | −0.000 | −0.002, +0.002 |
| 0.1:support_emb-empty:stable:gain | +0.046 | −0.030, +0.122 |
| 0.1:support_emb-empty:stable:regret | −0.000 | −0.001, +0.001 |
| 0.1:support_emb-empty:stable:frus | +0.000 | −0.001, +0.002 |
| 0.1:support_emb-empty:frontier:gain | +0.038 | −0.026, +0.102 |
| 0.1:support_emb-empty:frontier:regret | −0.002 | −0.003, −0.001 |
| 0.1:support_emb-empty:frontier:frus | −0.002 | −0.004, −0.000 |
| 0.1:nca_true-empty:target:gain | −0.124 | −0.208, −0.040 |
| 0.1:nca_true-empty:target:regret | −0.003 | −0.004, −0.002 |
| 0.1:nca_true-empty:target:frus | −0.003 | −0.005, −0.000 |
| 0.1:nca_true-empty:stable:gain | −0.114 | −0.190, −0.038 |
| 0.1:nca_true-empty:stable:regret | −0.006 | −0.007, −0.005 |
| 0.1:nca_true-empty:stable:frus | −0.001 | −0.003, +0.001 |
| 0.1:nca_true-empty:frontier:gain | −0.025 | −0.088, +0.038 |
| 0.1:nca_true-empty:frontier:regret | +0.000 | −0.001, +0.001 |
| 0.1:nca_true-empty:frontier:frus | −0.001 | −0.003, +0.000 |
| 0.1:nca_true-true_emb:target:gain | −0.327 | −0.413, −0.241 |
| 0.1:nca_true-true_emb:target:regret | −0.001 | −0.002, −0.000 |
| 0.1:nca_true-true_emb:target:frus | −0.000 | −0.002, +0.002 |
| 0.1:nca_true-true_emb:stable:gain | −0.296 | −0.379, −0.213 |
| 0.1:nca_true-true_emb:stable:regret | −0.005 | −0.006, −0.004 |
| 0.1:nca_true-true_emb:stable:frus | −0.002 | −0.003, +0.000 |
| 0.1:nca_true-true_emb:frontier:gain | +0.054 | −0.010, +0.118 |
| 0.1:nca_true-true_emb:frontier:regret | +0.005 | +0.005, +0.006 |
| 0.1:nca_true-true_emb:frontier:frus | +0.001 | +0.000, +0.002 |
| 0.1:nca-empty:target:gain | −0.241 | −0.324, −0.158 |
| 0.1:nca-empty:target:regret | −0.004 | −0.005, −0.003 |
| 0.1:nca-empty:target:frus | −0.002 | −0.005, −0.000 |
| 0.1:nca-empty:stable:gain | −0.215 | −0.293, −0.137 |
| 0.1:nca-empty:stable:regret | −0.009 | −0.010, −0.008 |
| 0.1:nca-empty:stable:frus | −0.002 | −0.003, −0.000 |
| 0.1:nca-empty:frontier:gain | −0.038 | −0.105, +0.029 |
| 0.1:nca-empty:frontier:regret | +0.000 | −0.001, +0.001 |
| 0.1:nca-empty:frontier:frus | −0.001 | −0.003, +0.000 |
| 1.0:true_emb-true_dp_emb:target:gain | −0.106 | −0.221, +0.009 |
| 1.0:true_emb-true_dp_emb:target:regret | −0.011 | −0.012, −0.010 |
| 1.0:true_emb-true_dp_emb:target:frus | −0.002 | −0.005, +0.001 |
| 1.0:true_emb-true_dp_emb:stable:gain | −0.160 | −0.273, −0.047 |
| 1.0:true_emb-true_dp_emb:stable:regret | −0.009 | −0.010, −0.008 |
| 1.0:true_emb-true_dp_emb:stable:frus | −0.004 | −0.007, −0.001 |
| 1.0:true_emb-true_dp_emb:frontier:gain | −0.232 | −0.326, −0.138 |
| 1.0:true_emb-true_dp_emb:frontier:regret | −0.013 | −0.014, −0.012 |
| 1.0:true_emb-true_dp_emb:frontier:frus | −0.000 | −0.002, +0.001 |
| 1.0:true_emb-empty:target:gain | −0.190 | −0.301, −0.079 |
| 1.0:true_emb-empty:target:regret | −0.007 | −0.009, −0.006 |
| 1.0:true_emb-empty:target:frus | −0.001 | −0.004, +0.002 |
| 1.0:true_emb-empty:stable:gain | −0.186 | −0.303, −0.069 |
| 1.0:true_emb-empty:stable:regret | −0.005 | −0.007, −0.004 |
| 1.0:true_emb-empty:stable:frus | +0.000 | −0.003, +0.003 |
| 1.0:true_emb-empty:frontier:gain | −0.218 | −0.311, −0.125 |
| 1.0:true_emb-empty:frontier:regret | −0.009 | −0.010, −0.008 |
| 1.0:true_emb-empty:frontier:frus | +0.000 | −0.001, +0.002 |
| 1.0:support_emb-empty:target:gain | −0.008 | −0.114, +0.098 |
| 1.0:support_emb-empty:target:regret | −0.002 | −0.004, −0.001 |
| 1.0:support_emb-empty:target:frus | −0.001 | −0.004, +0.002 |
| 1.0:support_emb-empty:stable:gain | −0.026 | −0.135, +0.083 |
| 1.0:support_emb-empty:stable:regret | +0.000 | −0.001, +0.001 |
| 1.0:support_emb-empty:stable:frus | −0.002 | −0.005, +0.001 |
| 1.0:support_emb-empty:frontier:gain | +0.012 | −0.077, +0.101 |
| 1.0:support_emb-empty:frontier:regret | +0.001 | −0.000, +0.002 |
| 1.0:support_emb-empty:frontier:frus | +0.001 | −0.001, +0.002 |
| 1.0:nca_true-empty:target:gain | −0.172 | −0.288, −0.056 |
| 1.0:nca_true-empty:target:regret | −0.006 | −0.007, −0.004 |
| 1.0:nca_true-empty:target:frus | −0.000 | −0.003, +0.003 |
| 1.0:nca_true-empty:stable:gain | −0.066 | −0.182, +0.050 |
| 1.0:nca_true-empty:stable:regret | −0.004 | −0.005, −0.002 |
| 1.0:nca_true-empty:stable:frus | −0.001 | −0.004, +0.002 |
| 1.0:nca_true-empty:frontier:gain | −0.144 | −0.240, −0.048 |
| 1.0:nca_true-empty:frontier:regret | −0.003 | −0.004, −0.001 |
| 1.0:nca_true-empty:frontier:frus | −0.000 | −0.002, +0.001 |
| 1.0:nca_true-true_emb:target:gain | +0.018 | −0.101, +0.137 |
| 1.0:nca_true-true_emb:target:regret | +0.002 | +0.000, +0.004 |
| 1.0:nca_true-true_emb:target:frus | +0.001 | −0.002, +0.004 |
| 1.0:nca_true-true_emb:stable:gain | +0.120 | +0.010, +0.230 |
| 1.0:nca_true-true_emb:stable:regret | +0.002 | +0.000, +0.003 |
| 1.0:nca_true-true_emb:stable:frus | −0.001 | −0.004, +0.002 |
| 1.0:nca_true-true_emb:frontier:gain | +0.074 | −0.014, +0.162 |
| 1.0:nca_true-true_emb:frontier:regret | +0.006 | +0.005, +0.008 |
| 1.0:nca_true-true_emb:frontier:frus | −0.000 | −0.002, +0.001 |
| 1.0:nca-empty:target:gain | −0.304 | −0.417, −0.191 |
| 1.0:nca-empty:target:regret | −0.011 | −0.012, −0.009 |
| 1.0:nca-empty:target:frus | −0.003 | −0.006, +0.000 |
| 1.0:nca-empty:stable:gain | −0.422 | −0.538, −0.306 |
| 1.0:nca-empty:stable:regret | −0.011 | −0.013, −0.010 |
| 1.0:nca-empty:stable:frus | −0.005 | −0.008, −0.002 |
| 1.0:nca-empty:frontier:gain | −0.238 | −0.328, −0.148 |
| 1.0:nca-empty:frontier:regret | −0.003 | −0.004, −0.002 |
| 1.0:nca-empty:frontier:frus | −0.001 | −0.002, +0.001 |

**Table S20.2** ASSISTments 2009 world: test AUC by condition (mean over seeds)

| Size and condition | Seeds | First att. | All pred. |
|---|---|---|---|
| 0.1:empty | 2 | 0.6730 | 0.7284 |
| 0.1:support_emb | 2 | 0.6733 | 0.7302 |
| 0.1:er_emb | 2 | 0.6736 | 0.7301 |
| 0.1:dp_emb | 2 | 0.6737 | 0.7295 |
| 0.1:true_emb | 2 | 0.6744 | 0.7299 |
| 0.1:true_dp_emb | 2 | 0.6718 | 0.7290 |
| 0.1:nca | 2 | 0.6741 | 0.7230 |
| 0.1:nca_dp | 2 | 0.6750 | 0.7230 |
| 0.1:nca_true | 2 | 0.6740 | 0.7234 |
| 0.1:st_true | 2 | 0.6725 | 0.7292 |
| 0.1:st_true_dp | 2 | 0.6730 | 0.7282 |
| 1.0:empty | 1 | 0.6922 | 0.7445 |
| 1.0:support_emb | 1 | 0.6915 | 0.7451 |
| 1.0:true_emb | 1 | 0.6900 | 0.7449 |
| 1.0:true_dp_emb | 1 | 0.6918 | 0.7446 |
| 1.0:nca_true | 1 | 0.7015 | 0.7455 |

**Table S20.4** ASSISTments 2009 world: prediction terms

| Term | First att. | All pred. | Per seed (first) |
|---|---|---|---|
| 0.1:true_emb-true_dp_emb | +0.0026 | +0.0008 | +0.0027, +0.0024 |
| 0.1:true_emb-empty | +0.0014 | +0.0014 | −0.0002, +0.0031 |
| 0.1:support_emb-empty | +0.0003 | +0.0018 | +0.0002, +0.0005 |
| 0.1:support_emb-dp_emb | −0.0004 | +0.0007 | −0.0023, +0.0015 |
| 0.1:er_emb-empty | +0.0006 | +0.0017 | −0.0018, +0.0030 |
| 0.1:true_emb-support_emb | +0.0011 | −0.0003 | −0.0003, +0.0025 |
| 0.1:nca_true-empty | +0.0010 | −0.0050 | +0.0004, +0.0017 |
| 0.1:nca-nca_dp | −0.0009 | −0.0000 | −0.0012, −0.0005 |
| 0.1:nca_true-true_emb | −0.0004 | −0.0065 | +0.0006, −0.0014 |
| 0.1:nca-empty | +0.0011 | −0.0055 | −0.0007, +0.0030 |
| 0.1:st_true-st_true_dp | −0.0005 | +0.0010 | −0.0011, −0.0000 |
| 0.1:st_true-empty | −0.0005 | +0.0008 | −0.0004, −0.0005 |
| 1.0:true_emb-true_dp_emb | −0.0018 | +0.0003 | −0.0018 |
| 1.0:true_emb-empty | −0.0022 | +0.0004 | −0.0022 |
| 1.0:support_emb-empty | −0.0007 | +0.0006 | −0.0007 |
| 1.0:true_emb-support_emb | −0.0015 | −0.0002 | −0.0015 |
| 1.0:nca_true-empty | +0.0093 | +0.0010 | +0.0093 |
| 1.0:nca_true-true_emb | +0.0115 | +0.0006 | +0.0115 |

**Table S20.6** ASSISTments 2009 world: reference policies (gain in KCs; ttm5 steps to five new masteries, 41 if never; reached5 share of learners reaching five; frustration; success rate; regret of the target rule under the true model)

| Policy | gain | ttm5 | reached5 | frustration | success | regret |
|---|---|---|---|---|---|---|
| oracle | -1.710 | 41.000 | 0.000 | 0.003 | 0.851 | 0.000 |
| oracle_frontier | 8.834 | 22.138 | 0.800 | 0.420 | 0.242 | 0.200 |
| curriculum | 0.694 | 40.076 | 0.098 | 0.167 | 0.538 | 0.132 |
| random | 0.894 | 39.732 | 0.086 | 0.245 | 0.377 | 0.172 |

**Table S20.8** ASSISTments 2009 world: tracer-driven policies (first_choice_agreement and jaccard10 between the two seeds of a tracer: same first adaptive choice; overlap of the first ten KCs practised)

| Size:tracer:rule | gain | ttm5 | frustration | success | regret | first_choice_agreement | jaccard10 | Seeds |
|---|---|---|---|---|---|---|---|---|
| 1.0:empty:target | 0.628 | 40.406 | 0.051 | 0.623 | 0.102 | -- | -- | 1 |
| 1.0:empty:stable | 0.526 | 40.330 | 0.043 | 0.639 | 0.096 | -- | -- | 1 |
| 1.0:empty:frontier | -0.478 | 40.986 | 0.024 | 0.733 | 0.062 | -- | -- | 1 |
| 1.0:nca_true:target | 0.174 | 40.650 | 0.061 | 0.582 | 0.107 | -- | -- | 1 |
| 1.0:nca_true:stable | 0.176 | 40.496 | 0.048 | 0.617 | 0.099 | -- | -- | 1 |
| 1.0:nca_true:frontier | -0.704 | 40.974 | 0.018 | 0.730 | 0.061 | -- | -- | 1 |
| 1.0:support_emb:target | 0.894 | 40.284 | 0.051 | 0.607 | 0.108 | -- | -- | 1 |
| 1.0:support_emb:stable | 0.588 | 40.326 | 0.043 | 0.635 | 0.098 | -- | -- | 1 |
| 1.0:support_emb:frontier | -0.302 | 40.920 | 0.026 | 0.727 | 0.062 | -- | -- | 1 |
| 1.0:true_dp_emb:target | 0.808 | 40.254 | 0.054 | 0.593 | 0.109 | -- | -- | 1 |
| 1.0:true_dp_emb:stable | 0.658 | 40.414 | 0.051 | 0.620 | 0.099 | -- | -- | 1 |
| 1.0:true_dp_emb:frontier | -0.388 | 40.926 | 0.022 | 0.720 | 0.064 | -- | -- | 1 |
| 1.0:true_emb:target | 0.424 | 40.698 | 0.038 | 0.629 | 0.115 | -- | -- | 1 |
| 1.0:true_emb:stable | 0.520 | 40.420 | 0.035 | 0.633 | 0.109 | -- | -- | 1 |
| 1.0:true_emb:frontier | -0.596 | 41.000 | 0.023 | 0.730 | 0.074 | -- | -- | 1 |
| 0.1:dp_emb:target | 0.419 | 40.486 | 0.051 | 0.599 | 0.105 | 0.166 | 0.257 | 2 |
| 0.1:dp_emb:stable | 0.263 | 40.491 | 0.038 | 0.647 | 0.093 | 0.256 | 0.368 | 2 |
| 0.1:dp_emb:frontier | -0.675 | 40.996 | 0.017 | 0.742 | 0.062 | 0.270 | 0.413 | 2 |
| 0.1:empty:target | 0.540 | 40.432 | 0.065 | 0.560 | 0.117 | 0.088 | 0.154 | 2 |
| 0.1:empty:stable | 0.342 | 40.628 | 0.047 | 0.614 | 0.104 | 0.182 | 0.277 | 2 |
| 0.1:empty:frontier | -0.468 | 40.939 | 0.024 | 0.702 | 0.072 | 0.230 | 0.280 | 2 |
| 0.1:er_emb:target | 0.131 | 40.591 | 0.046 | 0.612 | 0.101 | 0.248 | 0.264 | 2 |
| 0.1:er_emb:stable | -0.068 | 40.789 | 0.032 | 0.664 | 0.089 | 0.360 | 0.362 | 2 |
| 0.1:er_emb:frontier | -0.966 | 40.988 | 0.023 | 0.755 | 0.057 | 0.342 | 0.425 | 2 |
| 0.1:nca:target | 0.432 | 40.392 | 0.053 | 0.585 | 0.110 | 0.090 | 0.150 | 2 |
| 0.1:nca:stable | 0.221 | 40.629 | 0.033 | 0.643 | 0.095 | 0.160 | 0.277 | 2 |
| 0.1:nca:frontier | -0.545 | 40.968 | 0.019 | 0.737 | 0.062 | 0.216 | 0.303 | 2 |
| 0.1:nca_dp:target | 0.405 | 40.518 | 0.052 | 0.575 | 0.114 | 0.110 | 0.162 | 2 |
| 0.1:nca_dp:stable | 0.158 | 40.676 | 0.034 | 0.641 | 0.097 | 0.166 | 0.281 | 2 |
| 0.1:nca_dp:frontier | -0.604 | 40.941 | 0.016 | 0.738 | 0.064 | 0.246 | 0.310 | 2 |
| 0.1:nca_true:target | 0.549 | 40.383 | 0.055 | 0.559 | 0.117 | 0.094 | 0.144 | 2 |
| 0.1:nca_true:stable | 0.364 | 40.529 | 0.033 | 0.627 | 0.101 | 0.150 | 0.271 | 2 |
| 0.1:nca_true:frontier | -0.423 | 40.876 | 0.023 | 0.708 | 0.071 | 0.184 | 0.265 | 2 |
| 0.1:support_emb:target | 0.539 | 40.398 | 0.051 | 0.591 | 0.105 | 0.166 | 0.210 | 2 |
| 0.1:support_emb:stable | 0.349 | 40.604 | 0.037 | 0.636 | 0.095 | 0.270 | 0.317 | 2 |
| 0.1:support_emb:frontier | -0.547 | 40.985 | 0.020 | 0.724 | 0.065 | 0.238 | 0.341 | 2 |
| 0.1:true_dp_emb:target | 0.547 | 40.323 | 0.063 | 0.569 | 0.117 | 0.134 | 0.181 | 2 |
| 0.1:true_dp_emb:stable | 0.315 | 40.612 | 0.047 | 0.620 | 0.104 | 0.250 | 0.295 | 2 |
| 0.1:true_dp_emb:frontier | -0.377 | 40.918 | 0.029 | 0.705 | 0.074 | 0.238 | 0.289 | 2 |
| 0.1:true_emb:target | 0.521 | 40.439 | 0.046 | 0.602 | 0.104 | 0.160 | 0.203 | 2 |
| 0.1:true_emb:stable | 0.349 | 40.637 | 0.035 | 0.641 | 0.096 | 0.268 | 0.311 | 2 |
| 0.1:true_emb:frontier | -0.575 | 40.963 | 0.016 | 0.742 | 0.060 | 0.232 | 0.359 | 2 |

**Table S20.10** ASSISTments 2009 world: paired differences per learner (seed-averaged; interval over learners)

| Pair:rule:outcome | Mean | 95% interval |
|---|---|---|
| 0.1:true_emb-true_dp_emb:target:gain | −0.026 | −0.111, +0.059 |
| 0.1:true_emb-true_dp_emb:target:regret | −0.013 | −0.014, −0.011 |
| 0.1:true_emb-true_dp_emb:target:frus | −0.017 | −0.020, −0.014 |
| 0.1:true_emb-true_dp_emb:stable:gain | +0.034 | −0.049, +0.117 |
| 0.1:true_emb-true_dp_emb:stable:regret | −0.009 | −0.010, −0.007 |
| 0.1:true_emb-true_dp_emb:stable:frus | −0.011 | −0.014, −0.008 |
| 0.1:true_emb-true_dp_emb:frontier:gain | −0.198 | −0.265, −0.131 |
| 0.1:true_emb-true_dp_emb:frontier:regret | −0.013 | −0.015, −0.012 |
| 0.1:true_emb-true_dp_emb:frontier:frus | −0.013 | −0.016, −0.011 |
| 0.1:true_emb-empty:target:gain | −0.019 | −0.105, +0.067 |
| 0.1:true_emb-empty:target:regret | −0.013 | −0.014, −0.011 |
| 0.1:true_emb-empty:target:frus | −0.019 | −0.023, −0.015 |
| 0.1:true_emb-empty:stable:gain | +0.007 | −0.076, +0.090 |
| 0.1:true_emb-empty:stable:regret | −0.008 | −0.010, −0.007 |
| 0.1:true_emb-empty:stable:frus | −0.012 | −0.016, −0.008 |
| 0.1:true_emb-empty:frontier:gain | −0.107 | −0.176, −0.038 |
| 0.1:true_emb-empty:frontier:regret | −0.012 | −0.013, −0.010 |
| 0.1:true_emb-empty:frontier:frus | −0.008 | −0.011, −0.006 |
| 0.1:support_emb-empty:target:gain | −0.001 | −0.087, +0.085 |
| 0.1:support_emb-empty:target:regret | −0.012 | −0.013, −0.010 |
| 0.1:support_emb-empty:target:frus | −0.014 | −0.018, −0.010 |
| 0.1:support_emb-empty:stable:gain | +0.007 | −0.069, +0.083 |
| 0.1:support_emb-empty:stable:regret | −0.009 | −0.010, −0.008 |
| 0.1:support_emb-empty:stable:frus | −0.011 | −0.014, −0.007 |
| 0.1:support_emb-empty:frontier:gain | −0.079 | −0.142, −0.016 |
| 0.1:support_emb-empty:frontier:regret | −0.007 | −0.009, −0.006 |
| 0.1:support_emb-empty:frontier:frus | −0.004 | −0.007, −0.002 |
| 0.1:nca_true-empty:target:gain | +0.009 | −0.077, +0.095 |
| 0.1:nca_true-empty:target:regret | +0.000 | −0.001, +0.001 |
| 0.1:nca_true-empty:target:frus | −0.010 | −0.014, −0.006 |
| 0.1:nca_true-empty:stable:gain | +0.022 | −0.065, +0.109 |
| 0.1:nca_true-empty:stable:regret | −0.003 | −0.004, −0.002 |
| 0.1:nca_true-empty:stable:frus | −0.014 | −0.017, −0.011 |
| 0.1:nca_true-empty:frontier:gain | +0.045 | −0.023, +0.113 |
| 0.1:nca_true-empty:frontier:regret | −0.001 | −0.003, +0.000 |
| 0.1:nca_true-empty:frontier:frus | −0.001 | −0.004, +0.002 |
| 0.1:nca_true-true_emb:target:gain | +0.028 | −0.066, +0.122 |
| 0.1:nca_true-true_emb:target:regret | +0.013 | +0.011, +0.014 |
| 0.1:nca_true-true_emb:target:frus | +0.009 | +0.006, +0.013 |
| 0.1:nca_true-true_emb:stable:gain | +0.015 | −0.069, +0.099 |
| 0.1:nca_true-true_emb:stable:regret | +0.005 | +0.004, +0.007 |
| 0.1:nca_true-true_emb:stable:frus | −0.002 | −0.005, +0.001 |
| 0.1:nca_true-true_emb:frontier:gain | +0.152 | +0.079, +0.225 |
| 0.1:nca_true-true_emb:frontier:regret | +0.011 | +0.009, +0.012 |
| 0.1:nca_true-true_emb:frontier:frus | +0.007 | +0.005, +0.009 |
| 0.1:nca-empty:target:gain | −0.108 | −0.189, −0.027 |
| 0.1:nca-empty:target:regret | −0.007 | −0.009, −0.006 |
| 0.1:nca-empty:target:frus | −0.013 | −0.016, −0.009 |
| 0.1:nca-empty:stable:gain | −0.121 | −0.201, −0.041 |
| 0.1:nca-empty:stable:regret | −0.009 | −0.010, −0.008 |
| 0.1:nca-empty:stable:frus | −0.015 | −0.018, −0.012 |
| 0.1:nca-empty:frontier:gain | −0.077 | −0.145, −0.009 |
| 0.1:nca-empty:frontier:regret | −0.010 | −0.012, −0.008 |
| 0.1:nca-empty:frontier:frus | −0.005 | −0.008, −0.002 |
| 1.0:true_emb-true_dp_emb:target:gain | −0.384 | −0.499, −0.269 |
| 1.0:true_emb-true_dp_emb:target:regret | +0.006 | +0.004, +0.009 |
| 1.0:true_emb-true_dp_emb:target:frus | −0.016 | −0.021, −0.012 |
| 1.0:true_emb-true_dp_emb:stable:gain | −0.138 | −0.253, −0.023 |
| 1.0:true_emb-true_dp_emb:stable:regret | +0.011 | +0.008, +0.013 |
| 1.0:true_emb-true_dp_emb:stable:frus | −0.016 | −0.021, −0.012 |
| 1.0:true_emb-true_dp_emb:frontier:gain | −0.208 | −0.306, −0.110 |
| 1.0:true_emb-true_dp_emb:frontier:regret | +0.010 | +0.007, +0.013 |
| 1.0:true_emb-true_dp_emb:frontier:frus | +0.002 | −0.002, +0.005 |
| 1.0:true_emb-empty:target:gain | −0.204 | −0.318, −0.090 |
| 1.0:true_emb-empty:target:regret | +0.013 | +0.011, +0.015 |
| 1.0:true_emb-empty:target:frus | −0.014 | −0.018, −0.009 |
| 1.0:true_emb-empty:stable:gain | −0.006 | −0.128, +0.116 |
| 1.0:true_emb-empty:stable:regret | +0.013 | +0.011, +0.015 |
| 1.0:true_emb-empty:stable:frus | −0.008 | −0.012, −0.004 |
| 1.0:true_emb-empty:frontier:gain | −0.118 | −0.205, −0.031 |
| 1.0:true_emb-empty:frontier:regret | +0.012 | +0.010, +0.015 |
| 1.0:true_emb-empty:frontier:frus | −0.001 | −0.005, +0.003 |
| 1.0:support_emb-empty:target:gain | +0.266 | +0.145, +0.387 |
| 1.0:support_emb-empty:target:regret | +0.006 | +0.004, +0.008 |
| 1.0:support_emb-empty:target:frus | −0.000 | −0.005, +0.004 |
| 1.0:support_emb-empty:stable:gain | +0.062 | −0.056, +0.180 |
| 1.0:support_emb-empty:stable:regret | +0.001 | −0.001, +0.003 |
| 1.0:support_emb-empty:stable:frus | +0.001 | −0.004, +0.005 |
| 1.0:support_emb-empty:frontier:gain | +0.176 | +0.078, +0.274 |
| 1.0:support_emb-empty:frontier:regret | +0.000 | −0.002, +0.002 |
| 1.0:support_emb-empty:frontier:frus | +0.001 | −0.002, +0.005 |
| 1.0:nca_true-empty:target:gain | −0.454 | −0.588, −0.320 |
| 1.0:nca_true-empty:target:regret | +0.005 | +0.002, +0.008 |
| 1.0:nca_true-empty:target:frus | +0.009 | +0.004, +0.015 |
| 1.0:nca_true-empty:stable:gain | −0.350 | −0.489, −0.211 |
| 1.0:nca_true-empty:stable:regret | +0.002 | −0.000, +0.005 |
| 1.0:nca_true-empty:stable:frus | +0.005 | +0.001, +0.010 |
| 1.0:nca_true-empty:frontier:gain | −0.226 | −0.324, −0.128 |
| 1.0:nca_true-empty:frontier:regret | −0.001 | −0.004, +0.002 |
| 1.0:nca_true-empty:frontier:frus | −0.006 | −0.010, −0.002 |
| 1.0:nca_true-true_emb:target:gain | −0.250 | −0.374, −0.126 |
| 1.0:nca_true-true_emb:target:regret | −0.008 | −0.011, −0.005 |
| 1.0:nca_true-true_emb:target:frus | +0.023 | +0.018, +0.028 |
| 1.0:nca_true-true_emb:stable:gain | −0.344 | −0.468, −0.220 |
| 1.0:nca_true-true_emb:stable:regret | −0.011 | −0.014, −0.008 |
| 1.0:nca_true-true_emb:stable:frus | +0.013 | +0.009, +0.018 |
| 1.0:nca_true-true_emb:frontier:gain | −0.108 | −0.200, −0.016 |
| 1.0:nca_true-true_emb:frontier:regret | −0.013 | −0.016, −0.010 |
| 1.0:nca_true-true_emb:frontier:frus | −0.005 | −0.009, −0.001 |

# S21 Runs of release 6.0.0

362 training runs (kt/results_r20/run_list_r20.csv lists each with its tag, subsample, seed, test AUC and run time). By condition:

| set | cond | frac | runs |
|---|---|---|---|
| runs_r20 | dp_emb | 0.1 | 5 |
| runs_r20 | empty | 0.1 | 5 |
| runs_r20 | empty | 1.0 | 1 |
| runs_r20 | empty_fa | 0.1 | 37 |
| runs_r20 | empty_fa | 1.0 | 11 |
| runs_r20 | er_emb | 0.1 | 5 |
| runs_r20 | expert_dp_emb | 0.1 | 5 |
| runs_r20 | expert_emb | 0.1 | 5 |
| runs_r20 | llm_dp_emb | 0.1 | 5 |
| runs_r20 | llm_emb | 0.1 | 5 |
| runs_r20 | map_dp_emb | 0.1 | 1 |
| runs_r20 | nca | 0.1 | 37 |
| runs_r20 | nca | 1.0 | 11 |
| runs_r20 | nca_dp | 0.1 | 15 |
| runs_r20 | nca_expert | 0.1 | 5 |
| runs_r20 | nca_llm | 0.1 | 5 |
| runs_r20 | nca_map | 0.1 | 1 |
| runs_r20 | nca_nogate | 0.1 | 11 |
| runs_r20 | nca_nomargin | 0.1 | 11 |
| runs_r20 | nca_nonull | 0.1 | 11 |
| runs_r20 | nca_static | 0.1 | 11 |
| runs_r20 | q_diff | 0.1 | 10 |
| runs_r20 | q_id | 0.1 | 10 |
| runs_r20 | q_kc | 0.1 | 10 |
| runs_r20 | q_qm | 0.1 | 10 |
| runs_r20 | q_qm_dp | 0.1 | 10 |
| runs_r20 | q_qm_g | 0.1 | 10 |
| runs_r20 | q_rqm | 0.1 | 10 |
| runs_r20 | st_dp | 0.1 | 5 |
| runs_r20 | st_er | 0.1 | 5 |
| runs_r20 | st_support | 0.1 | 5 |
| runs_r20 | support | 0.1 | 5 |
| runs_r20 | support | 1.0 | 1 |
| runs_r20 | support_emb | 0.1 | 5 |
| runs_r20 | uniform_emb | 0.1 | 5 |
| runs_r20_sim | dp_emb | 0.1 | 4 |
| runs_r20_sim | dp_emb | 1.0 | 1 |
| runs_r20_sim | empty | 0.1 | 4 |
| runs_r20_sim | empty | 1.0 | 2 |
| runs_r20_sim | er_emb | 0.1 | 4 |
| runs_r20_sim | er_emb | 1.0 | 1 |
| runs_r20_sim | nca | 0.1 | 4 |
| runs_r20_sim | nca | 1.0 | 1 |
| runs_r20_sim | nca_dp | 0.1 | 4 |
| runs_r20_sim | nca_dp | 1.0 | 1 |
| runs_r20_sim | nca_true | 0.1 | 4 |
| runs_r20_sim | nca_true | 1.0 | 2 |
| runs_r20_sim | st_true | 0.1 | 4 |
| runs_r20_sim | st_true_dp | 0.1 | 4 |
| runs_r20_sim | support_emb | 0.1 | 4 |
| runs_r20_sim | support_emb | 1.0 | 2 |
| runs_r20_sim | true_dp_emb | 0.1 | 4 |
| runs_r20_sim | true_dp_emb | 1.0 | 2 |
| runs_r20_sim | true_emb | 0.1 | 4 |
| runs_r20_sim | true_emb | 1.0 | 2 |
