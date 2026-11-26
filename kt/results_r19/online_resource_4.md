---
title: "Online Resource 4: analyses added in the revision (Sections S14 to S16)"
subtitle: "When does a concept graph help a knowledge tracer? Entry-point null-graph controls and decision stability for KC-level learner models"
author: "Hai Bang Truong, Saigon International University (truonghaibang@siu.edu.vn)"
---

Every value below is computed by the scripts of release 5.0.0 of the accompanying code (kt/src/r19_*.py) from the run files of kt/runs_r19 and of the earlier releases; the result files are kt/results_r19/*.json. Section numbers continue those of Online Resources 1 and 3.

# S14 Data, reproduction and graphs

## S14.1 Reproduction of the rebuilt corpora

Every corpus was rebuilt from the public copies listed in the README of release 5.0.0 with kt/src/prep.py (Junyi with prep_junyi_lowmem.py, XES3G5M with prep_r14.py) and matches Table 1 of the article in learners and capped interactions. The empty-graph control on subsample 0 at 10% was rerun on each rebuilt corpus (kt/runs_r19/repro) and compared with the archived run of the same condition, seed and code.

| Corpus | Archived run | Archived test AUC | Rerun test AUC | Identical |
|---|---|---|---|---|
| Algebra 2005 | runs_r9 | 0.7470181317 | 0.7470181317 | yes |
| ASSISTments 2009 | runs_r9 | 0.7376892190 | 0.7376892190 | yes |
| ASSISTments 2012 | runs_r9 | 0.7034439405 | 0.7034439405 | yes |
| ASSISTments 2015 | runs_r9 | 0.6998765737 | 0.6998765737 | yes |
| ASSISTments 2017 | runs_r9 | 0.7018387329 | 0.7018387329 | yes |
| Bridge to Alg. 2006 | runs_r9 | 0.7525425966 | 0.7525425966 | yes |
| EdNet KT1 | runs_r13 | 0.6715492508 | 0.6715492508 | yes |
| Eedi | runs_r9 | 0.7256459981 | 0.7256459981 | yes |
| Junyi Academy | runs_r9 | 0.7846231130 | 0.7846231130 | yes |
| Statics 2011 | runs_r9 | 0.7471066538 | 0.7471066538 | yes |
| XES3G5M | runs_r14 | 0.7289390227 | 0.7289390227 | yes |

Statics 2011 was also rerun before the new runs (0.7471066538). EdNet is compared with the rebuilt copy used since release 3.7.0 (runs_r13); against the original release-2.0.0 copy its test AUC differs by 6.5e-6 (kt/README.md, release 3.7.0).

## S14.2 Structure ablations of the eight graph-based tracers examined

**Table S14.1** What the experimental sections of the eight tracers named in Section 2.1 compare. A tick in the last column would mean a comparison of the graph with a random graph of the same size inside an otherwise unchanged network; none of the eight makes one.

| Tracer | Structure ablation reported | Random graph of the same size, network unchanged |
|---|---|---|
| GKT (Nakagawa et al. 2019) | dense graph against statistics-based and learned graphs, found close | no |
| GIKT (Yang et al. 2021) | number of GCN layers over the question–skill graph varied from 0 to 3 (0: embeddings indexed directly); related skills and related history removed from the interaction modules | no |
| SKT (Tong et al. 2020) | propagation over the prerequisite and similarity relations removed by module (author to verify against the ICDM paper) | no |
| RKT (Pandey and Srivastava 2020) | exercise-relation modeling removed, alone and with position encoding or forgetting; alternative computations of the relation matrix compared | no |
| DyGKT (Cheng et al. 2024) | multiset indicator removed; time encoders simplified or removed; question-difficulty module replaced by question- or concept-ID embeddings; no ablation of graph structure | no |
| PSI-KT (Zhou et al. 2024) | graph inference removed (each KC independent); inferred graph compared with expert and crowd annotations on Junyi | no |
| SINKT (Fu et al. 2024) | generated concept graph replaced by a transition graph built from the logs | no |
| HGKT (Tong et al. 2022) | support relations dropped together with the modules that read them | no |

## S14.3 Graph quality by training size

**Table S14.3** Support graph of every subsample against the graph from all training learners (*G*~100~). Edges: directed edges; recall: share of the edges of *G*~100~ in the subsample graph; precision: share of the subsample graph's edges in *G*~100~; Jaccard: of undirected pairs; reciprocal: reciprocal pairs in *G* and in DP(*G*).

| Corpus | Size | Subsample | Training learners | Edges | Edges of G100 | Recall | Precision | Jaccard | Reciprocal G | Reciprocal DP |
|---|---|---|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 5% | 0 | 86 | 19 | 557 | 0.025 | 0.737 | 0.032 | 0 | 0 |
| ASSISTments 2009 | 5% | 1 | 86 | 26 | 557 | 0.038 | 0.808 | 0.043 | 0 | 0 |
| ASSISTments 2009 | 5% | 2 | 86 | 26 | 557 | 0.038 | 0.808 | 0.045 | 0 | 0 |
| ASSISTments 2009 | 5% | 3 | 86 | 26 | 557 | 0.036 | 0.769 | 0.043 | 0 | 0 |
| ASSISTments 2009 | 5% | 4 | 86 | 22 | 557 | 0.031 | 0.773 | 0.036 | 0 | 0 |
| ASSISTments 2009 | 10% | 0 | 172 | 59 | 557 | 0.088 | 0.831 | 0.102 | 0 | 0 |
| ASSISTments 2009 | 10% | 1 | 172 | 80 | 557 | 0.126 | 0.875 | 0.140 | 0 | 0 |
| ASSISTments 2009 | 10% | 2 | 172 | 83 | 557 | 0.127 | 0.855 | 0.145 | 0 | 0 |
| ASSISTments 2009 | 10% | 3 | 172 | 67 | 557 | 0.106 | 0.881 | 0.114 | 0 | 0 |
| ASSISTments 2009 | 10% | 4 | 172 | 49 | 557 | 0.074 | 0.837 | 0.084 | 0 | 0 |
| ASSISTments 2009 | 25% | 0 | 430 | 178 | 557 | 0.287 | 0.899 | 0.310 | 0 | 0 |
| ASSISTments 2009 | 25% | 1 | 430 | 195 | 557 | 0.312 | 0.892 | 0.338 | 0 | 0 |
| ASSISTments 2009 | 25% | 2 | 430 | 176 | 557 | 0.260 | 0.824 | 0.302 | 0 | 0 |
| ASSISTments 2012 | 5% | 0 | 211 | 104 | 1925 | 0.040 | 0.740 | 0.053 | 0 | 0 |
| ASSISTments 2012 | 5% | 1 | 211 | 101 | 1925 | 0.036 | 0.683 | 0.052 | 0 | 0 |
| ASSISTments 2012 | 5% | 2 | 211 | 111 | 1925 | 0.037 | 0.649 | 0.057 | 0 | 0 |
| ASSISTments 2012 | 5% | 3 | 211 | 106 | 1925 | 0.039 | 0.717 | 0.055 | 0 | 0 |
| ASSISTments 2012 | 5% | 4 | 211 | 101 | 1925 | 0.041 | 0.782 | 0.051 | 0 | 0 |
| ASSISTments 2012 | 10% | 0 | 422 | 235 | 1925 | 0.097 | 0.791 | 0.119 | 0 | 0 |
| ASSISTments 2012 | 10% | 1 | 422 | 253 | 1925 | 0.104 | 0.791 | 0.130 | 0 | 0 |
| ASSISTments 2012 | 10% | 2 | 422 | 232 | 1925 | 0.092 | 0.767 | 0.116 | 0 | 0 |
| ASSISTments 2012 | 10% | 3 | 422 | 231 | 1925 | 0.097 | 0.810 | 0.118 | 0 | 0 |
| ASSISTments 2012 | 10% | 4 | 422 | 224 | 1925 | 0.092 | 0.790 | 0.116 | 0 | 0 |
| ASSISTments 2012 | 25% | 0 | 1056 | 662 | 1925 | 0.295 | 0.856 | 0.333 | 0 | 0 |
| ASSISTments 2012 | 25% | 1 | 1056 | 652 | 1925 | 0.295 | 0.870 | 0.330 | 0 | 0 |
| ASSISTments 2012 | 25% | 2 | 1056 | 668 | 1925 | 0.300 | 0.865 | 0.337 | 0 | 0 |
| ASSISTments 2015 | 5% | 0 | 209 | 19 | 988 | 0.018 | 0.947 | 0.019 | 0 | 0 |
| ASSISTments 2015 | 5% | 1 | 209 | 34 | 988 | 0.028 | 0.824 | 0.034 | 0 | 0 |
| ASSISTments 2015 | 5% | 2 | 209 | 27 | 988 | 0.023 | 0.852 | 0.027 | 0 | 0 |
| ASSISTments 2015 | 5% | 3 | 209 | 28 | 988 | 0.025 | 0.893 | 0.028 | 0 | 0 |
| ASSISTments 2015 | 5% | 4 | 209 | 27 | 988 | 0.020 | 0.741 | 0.027 | 0 | 0 |
| ASSISTments 2015 | 10% | 0 | 418 | 79 | 988 | 0.074 | 0.924 | 0.080 | 0 | 0 |
| ASSISTments 2015 | 10% | 1 | 418 | 83 | 988 | 0.070 | 0.831 | 0.084 | 0 | 0 |
| ASSISTments 2015 | 10% | 2 | 418 | 92 | 988 | 0.082 | 0.880 | 0.093 | 0 | 0 |
| ASSISTments 2015 | 10% | 3 | 418 | 87 | 988 | 0.079 | 0.897 | 0.088 | 0 | 0 |
| ASSISTments 2015 | 10% | 4 | 418 | 79 | 988 | 0.077 | 0.962 | 0.080 | 0 | 0 |
| ASSISTments 2015 | 25% | 0 | 1044 | 265 | 988 | 0.244 | 0.909 | 0.264 | 0 | 0 |
| ASSISTments 2015 | 25% | 1 | 1044 | 277 | 988 | 0.266 | 0.949 | 0.276 | 0 | 0 |
| ASSISTments 2015 | 25% | 2 | 1044 | 266 | 988 | 0.243 | 0.902 | 0.267 | 0 | 0 |
| ASSISTments 2017 | 5% | 0 | 58 | 75 | 1233 | 0.042 | 0.693 | 0.060 | 0 | 0 |
| ASSISTments 2017 | 5% | 1 | 58 | 81 | 1233 | 0.046 | 0.704 | 0.064 | 0 | 0 |
| ASSISTments 2017 | 5% | 2 | 58 | 54 | 1233 | 0.029 | 0.667 | 0.044 | 0 | 0 |
| ASSISTments 2017 | 5% | 3 | 58 | 75 | 1233 | 0.037 | 0.613 | 0.060 | 0 | 0 |
| ASSISTments 2017 | 5% | 4 | 58 | 71 | 1233 | 0.036 | 0.634 | 0.058 | 0 | 0 |
| ASSISTments 2017 | 10% | 0 | 115 | 182 | 1233 | 0.101 | 0.681 | 0.143 | 0 | 0 |
| ASSISTments 2017 | 10% | 1 | 115 | 220 | 1233 | 0.122 | 0.682 | 0.171 | 0 | 0 |
| ASSISTments 2017 | 10% | 2 | 115 | 192 | 1233 | 0.105 | 0.677 | 0.148 | 0 | 0 |
| ASSISTments 2017 | 10% | 3 | 115 | 181 | 1233 | 0.087 | 0.591 | 0.143 | 0 | 0 |
| ASSISTments 2017 | 10% | 4 | 115 | 158 | 1233 | 0.085 | 0.665 | 0.125 | 0 | 0 |
| ASSISTments 2017 | 25% | 0 | 288 | 514 | 1233 | 0.298 | 0.714 | 0.399 | 0 | 0 |
| ASSISTments 2017 | 25% | 1 | 288 | 497 | 1233 | 0.273 | 0.678 | 0.380 | 0 | 0 |
| ASSISTments 2017 | 25% | 2 | 288 | 496 | 1233 | 0.277 | 0.690 | 0.375 | 0 | 0 |
| Junyi Academy | 5% | 0 | 210 | 80 | 958 | 0.071 | 0.850 | 0.082 | 0 | 0 |
| Junyi Academy | 5% | 1 | 210 | 80 | 958 | 0.077 | 0.925 | 0.081 | 0 | 0 |
| Junyi Academy | 5% | 2 | 210 | 81 | 958 | 0.070 | 0.827 | 0.082 | 0 | 0 |
| Junyi Academy | 5% | 3 | 210 | 79 | 958 | 0.074 | 0.899 | 0.081 | 0 | 0 |
| Junyi Academy | 5% | 4 | 210 | 75 | 958 | 0.068 | 0.867 | 0.077 | 0 | 0 |
| Junyi Academy | 10% | 0 | 420 | 146 | 958 | 0.133 | 0.870 | 0.149 | 0 | 0 |
| Junyi Academy | 10% | 1 | 420 | 145 | 958 | 0.134 | 0.883 | 0.148 | 0 | 0 |
| Junyi Academy | 10% | 2 | 420 | 146 | 958 | 0.140 | 0.918 | 0.150 | 0 | 0 |
| Junyi Academy | 10% | 3 | 420 | 132 | 958 | 0.122 | 0.886 | 0.138 | 0 | 0 |
| Junyi Academy | 10% | 4 | 420 | 138 | 958 | 0.134 | 0.928 | 0.142 | 0 | 0 |
| Junyi Academy | 25% | 0 | 1049 | 312 | 958 | 0.287 | 0.881 | 0.316 | 0 | 0 |
| Junyi Academy | 25% | 1 | 1049 | 326 | 958 | 0.308 | 0.905 | 0.331 | 0 | 0 |
| Junyi Academy | 25% | 2 | 1049 | 328 | 958 | 0.311 | 0.909 | 0.333 | 0 | 0 |
| Algebra 2005 | 5% | 0 | 21 | 55 | 222 | 0.225 | 0.909 | 0.248 | 0 | 0 |
| Algebra 2005 | 5% | 1 | 21 | 61 | 222 | 0.261 | 0.951 | 0.269 | 0 | 0 |
| Algebra 2005 | 5% | 2 | 21 | 61 | 222 | 0.261 | 0.951 | 0.275 | 0 | 0 |
| Algebra 2005 | 5% | 3 | 21 | 75 | 222 | 0.333 | 0.987 | 0.338 | 0 | 0 |
| Algebra 2005 | 5% | 4 | 21 | 51 | 222 | 0.207 | 0.902 | 0.230 | 0 | 0 |
| Algebra 2005 | 10% | 0 | 41 | 98 | 222 | 0.423 | 0.959 | 0.441 | 0 | 0 |
| Algebra 2005 | 10% | 1 | 41 | 92 | 222 | 0.378 | 0.913 | 0.414 | 0 | 0 |
| Algebra 2005 | 10% | 2 | 41 | 95 | 222 | 0.419 | 0.979 | 0.428 | 0 | 0 |
| Algebra 2005 | 10% | 3 | 41 | 94 | 222 | 0.410 | 0.968 | 0.423 | 0 | 0 |
| Algebra 2005 | 10% | 4 | 41 | 91 | 222 | 0.401 | 0.978 | 0.410 | 0 | 0 |
| Algebra 2005 | 25% | 0 | 104 | 145 | 222 | 0.640 | 0.979 | 0.653 | 0 | 0 |
| Algebra 2005 | 25% | 1 | 104 | 145 | 222 | 0.635 | 0.972 | 0.646 | 0 | 0 |
| Algebra 2005 | 25% | 2 | 104 | 137 | 222 | 0.595 | 0.964 | 0.610 | 0 | 0 |
| Eedi | 5% | 0 | 170 | 166 | 664 | 0.181 | 0.723 | 0.244 | 0 | 0 |
| Eedi | 5% | 1 | 170 | 169 | 664 | 0.197 | 0.775 | 0.253 | 0 | 0 |
| Eedi | 5% | 2 | 170 | 175 | 664 | 0.212 | 0.806 | 0.260 | 0 | 0 |
| Eedi | 5% | 3 | 170 | 171 | 664 | 0.184 | 0.713 | 0.254 | 0 | 0 |
| Eedi | 5% | 4 | 170 | 175 | 664 | 0.190 | 0.720 | 0.260 | 0 | 0 |
| Eedi | 10% | 0 | 340 | 248 | 664 | 0.292 | 0.782 | 0.365 | 0 | 0 |
| Eedi | 10% | 1 | 340 | 237 | 664 | 0.277 | 0.776 | 0.353 | 0 | 0 |
| Eedi | 10% | 2 | 340 | 259 | 664 | 0.298 | 0.764 | 0.384 | 0 | 0 |
| Eedi | 10% | 3 | 340 | 258 | 664 | 0.288 | 0.740 | 0.378 | 0 | 0 |
| Eedi | 10% | 4 | 340 | 218 | 664 | 0.247 | 0.752 | 0.322 | 0 | 0 |
| Eedi | 25% | 0 | 851 | 375 | 664 | 0.449 | 0.795 | 0.532 | 0 | 0 |
| Eedi | 25% | 1 | 851 | 376 | 664 | 0.473 | 0.835 | 0.543 | 0 | 0 |
| Eedi | 25% | 2 | 851 | 363 | 664 | 0.423 | 0.774 | 0.528 | 0 | 0 |
| EdNet KT1 | 5% | 0 | 208 | 218 | 3378 | 0.044 | 0.688 | 0.064 | 0 | 0 |
| EdNet KT1 | 5% | 1 | 208 | 220 | 3378 | 0.047 | 0.727 | 0.064 | 0 | 0 |
| EdNet KT1 | 5% | 2 | 208 | 227 | 3378 | 0.046 | 0.687 | 0.067 | 0 | 0 |
| EdNet KT1 | 5% | 3 | 208 | 252 | 3378 | 0.051 | 0.687 | 0.074 | 0 | 0 |
| EdNet KT1 | 5% | 4 | 208 | 246 | 3378 | 0.052 | 0.715 | 0.072 | 0 | 0 |
| EdNet KT1 | 10% | 0 | 416 | 587 | 3378 | 0.114 | 0.656 | 0.169 | 0 | 0 |
| EdNet KT1 | 10% | 1 | 416 | 540 | 3378 | 0.110 | 0.687 | 0.157 | 0 | 0 |
| EdNet KT1 | 10% | 2 | 416 | 522 | 3378 | 0.109 | 0.703 | 0.152 | 0 | 0 |
| EdNet KT1 | 10% | 3 | 416 | 558 | 3378 | 0.115 | 0.695 | 0.163 | 0 | 0 |
| EdNet KT1 | 10% | 4 | 416 | 535 | 3378 | 0.103 | 0.650 | 0.155 | 0 | 0 |
| EdNet KT1 | 25% | 0 | 1039 | 1523 | 3378 | 0.318 | 0.705 | 0.422 | 0 | 0 |
| EdNet KT1 | 25% | 1 | 1039 | 1439 | 3378 | 0.303 | 0.711 | 0.402 | 0 | 0 |
| EdNet KT1 | 25% | 2 | 1039 | 1517 | 3378 | 0.319 | 0.710 | 0.422 | 0 | 0 |
| Bridge to Alg. 2006 | 5% | 0 | 38 | 91 | 327 | 0.272 | 0.978 | 0.278 | 0 | 0 |
| Bridge to Alg. 2006 | 5% | 1 | 38 | 85 | 327 | 0.254 | 0.976 | 0.260 | 0 | 0 |
| Bridge to Alg. 2006 | 5% | 2 | 38 | 101 | 327 | 0.303 | 0.980 | 0.309 | 0 | 0 |
| Bridge to Alg. 2006 | 5% | 3 | 38 | 78 | 327 | 0.235 | 0.987 | 0.239 | 0 | 0 |
| Bridge to Alg. 2006 | 5% | 4 | 38 | 101 | 327 | 0.300 | 0.970 | 0.309 | 0 | 0 |
| Bridge to Alg. 2006 | 10% | 0 | 77 | 112 | 327 | 0.343 | 1 | 0.343 | 0 | 0 |
| Bridge to Alg. 2006 | 10% | 1 | 77 | 139 | 327 | 0.410 | 0.964 | 0.421 | 0 | 0 |
| Bridge to Alg. 2006 | 10% | 2 | 77 | 148 | 327 | 0.446 | 0.986 | 0.453 | 0 | 0 |
| Bridge to Alg. 2006 | 10% | 3 | 77 | 152 | 327 | 0.450 | 0.967 | 0.465 | 0 | 0 |
| Bridge to Alg. 2006 | 10% | 4 | 77 | 141 | 327 | 0.425 | 0.986 | 0.431 | 0 | 0 |
| Bridge to Alg. 2006 | 25% | 0 | 192 | 209 | 327 | 0.624 | 0.976 | 0.634 | 0 | 0 |
| Bridge to Alg. 2006 | 25% | 1 | 192 | 207 | 327 | 0.630 | 0.995 | 0.633 | 0 | 0 |
| Bridge to Alg. 2006 | 25% | 2 | 192 | 197 | 327 | 0.596 | 0.990 | 0.602 | 0 | 0 |
| Statics 2011 | 5% | 0 | 10 | 17 | 116 | 0.138 | 0.941 | 0.147 | 0 | 0 |
| Statics 2011 | 5% | 1 | 10 | 20 | 116 | 0.155 | 0.900 | 0.172 | 0 | 0 |
| Statics 2011 | 5% | 2 | 10 | 22 | 116 | 0.172 | 0.909 | 0.190 | 0 | 0 |
| Statics 2011 | 5% | 3 | 10 | 25 | 116 | 0.198 | 0.920 | 0.216 | 0 | 0 |
| Statics 2011 | 5% | 4 | 10 | 20 | 116 | 0.138 | 0.800 | 0.172 | 0 | 0 |
| Statics 2011 | 10% | 0 | 21 | 39 | 116 | 0.336 | 1 | 0.336 | 0 | 0 |
| Statics 2011 | 10% | 1 | 21 | 49 | 116 | 0.414 | 0.980 | 0.422 | 0 | 0 |
| Statics 2011 | 10% | 2 | 21 | 50 | 116 | 0.397 | 0.920 | 0.431 | 0 | 0 |
| Statics 2011 | 10% | 3 | 21 | 42 | 116 | 0.345 | 0.952 | 0.362 | 0 | 0 |
| Statics 2011 | 10% | 4 | 21 | 42 | 116 | 0.353 | 0.976 | 0.362 | 0 | 0 |
| Statics 2011 | 25% | 0 | 52 | 62 | 116 | 0.526 | 0.984 | 0.534 | 0 | 0 |
| Statics 2011 | 25% | 1 | 52 | 58 | 116 | 0.500 | 1 | 0.500 | 0 | 0 |
| Statics 2011 | 25% | 2 | 52 | 67 | 116 | 0.569 | 0.985 | 0.578 | 0 | 0 |

# S15 Prediction

## S15.1 First-attempt references and averaging schemes

**Table S15.1** First-attempt AUC at 10% of the training learners, mean over the subsamples of the 32-unit design, by corpus and averaging scheme (Section 5.6 of the article). EdNet: no archived prediction file of the support graph on the rebuilt copy.

| Corpus | Units | KC rate, pooled | KC rate, per learner | KC rate, within KC | PFA-style, pooled | PFA-style, per learner | PFA-style, within KC | Probe ∅, pooled | Probe ∅, per learner | Probe ∅, within KC | Probe G (E), pooled | Probe G (E), per learner | Probe G (E), within KC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 2 | 0.6321 | 0.6388 | 0.5000 | 0.7029 | 0.6155 | 0.6660 | 0.7122 | 0.6095 | 0.6719 | 0.7204 | 0.6223 | 0.6771 |
| ASSISTments 2012 | 2 | 0.6073 | 0.6176 | 0.5000 | 0.6660 | 0.5786 | 0.6319 | 0.6658 | 0.5572 | 0.6304 | 0.6713 | 0.5770 | 0.6329 |
| ASSISTments 2015 | 2 | 0.6605 | 0.6542 | 0.5000 | 0.6893 | 0.6180 | 0.6173 | 0.6931 | 0.6259 | 0.6148 | 0.6984 | 0.6306 | 0.6201 |
| ASSISTments 2017 | 5 | 0.6322 | 0.6450 | 0.5000 | 0.7085 | 0.6209 | 0.6673 | 0.7089 | 0.6124 | 0.6673 | 0.7118 | 0.6138 | 0.6713 |
| Junyi Academy | 2 | 0.7592 | 0.7392 | 0.5000 | 0.7774 | 0.7268 | 0.6121 | 0.7530 | 0.6780 | 0.6098 | 0.7654 | 0.6910 | 0.6147 |
| Algebra 2005 | 5 | 0.6877 | 0.7230 | 0.5000 | 0.6987 | 0.6999 | 0.5988 | 0.7203 | 0.7018 | 0.6317 | 0.7209 | 0.7037 | 0.6237 |
| Eedi | 2 | 0.6021 | 0.6097 | 0.5000 | 0.7145 | 0.5834 | 0.6938 | 0.7265 | 0.5998 | 0.7036 | 0.7257 | 0.5930 | 0.7031 |
| EdNet KT1 | 2 | 0.6373 | 0.6368 | 0.5000 | 0.6630 | 0.5958 | 0.5945 | 0.6742 | 0.6017 | 0.6108 |  |  |  |
| Bridge to Alg. 2006 | 5 | 0.6478 | 0.6665 | 0.5000 | 0.6546 | 0.6320 | 0.6107 | 0.6989 | 0.6718 | 0.6425 | 0.7017 | 0.6763 | 0.6442 |
| Statics 2011 | 5 | 0.7083 | 0.7115 | 0.5000 | 0.7159 | 0.6981 | 0.6065 | 0.7481 | 0.7393 | 0.6159 | 0.7522 | 0.7371 | 0.6355 |

## S15.2 Primary comparison: interval methods, balanced design and seeds

Mixed: linear mixed model with a random intercept per corpus, REML (statsmodels mixedlm), Wald interval; Fit: "mixed", or "ols" where the random intercept was not estimable (one unit per corpus, or a singular fit), in which case ordinary least squares over the units is used. Fixed t: mean of corpus means with a t interval over corpora. Cluster bootstrap: corpora resampled, then subsamples within corpus, 10,000 draws. In parentheses: corpora with a positive difference.

| Term | Units | First att. | Mixed 95% | Fit | Fixed t 95% | Cluster bootstrap 95% | All pred. |
|---|---|---|---|---|---|---|---|
| 5%, update budget (32 units) | 32 | +0.0027 (7/10) | +0.0002, +0.0046 | mixed | -0.0002, +0.0055 | +0.0001, +0.0052 | -0.0001 (3/10) |
| 10%, 32 units (primary) | 32 | +0.0043 (9/10) | +0.0022, +0.0054 | mixed | +0.0016, +0.0070 | +0.0020, +0.0070 | +0.0021 (8/10) |
| 10%, 50 units (balanced) | 50 | +0.0047 (9/10) | +0.0022, +0.0072 | mixed | +0.0018, +0.0076 | +0.0022, +0.0074 | +0.0020 (8/10) |
| 10%, three seeds per unit (14 units) | 14 | +0.0045 (8/10) | +0.0014, +0.0075 | mixed | +0.0010, +0.0081 | +0.0017, +0.0076 | +0.0019 (9/10) |
| same units, seed 0 only | 14 | +0.0025 (8/10) | -0.0011, +0.0053 | mixed | -0.0010, +0.0060 | -0.0006, +0.0054 |  |

Smallest effect of interest from the control alone: δ~seed~ = 0.0039 (root mean square over units of the within-unit SD of the empty-graph control's first-attempt AUC across seeds 0, 100 and 200; median 0.0035).

Per corpus, three-seed gain on first attempts: Algebra 2005 -0.0007; ASSISTments 2009 +0.0115; ASSISTments 2012 +0.0044; ASSISTments 2015 +0.0074; ASSISTments 2017 +0.0031; Bridge to Alg. 2006 +0.0045; EdNet KT1 +0.0019; Eedi +0.0006; Junyi Academy +0.0138; Statics 2011 -0.0010.

## S15.3 Exploratory terms with false-discovery-rate adjustment

**Table S15.2** Every exploratory first-attempt term of Tables 3 and 4 of the earlier version and of Section 6.4 (results_r14, results_r16): mixed-model estimate and SE, two-sided *p* from the Wald statistic, Benjamini–Hochberg *q* over the 63 terms, and the equivalence class from the 90% interval at δ = δ~seed~ (B+/B−: beyond δ; N: nonzero, not beyond; S: nonzero, inside ±δ; E: equivalent to zero; U: unresolved). 23 terms have *p* < 0.05 against 3.2 expected by chance; 18 have *q* < 0.05.

| Term | Mean over corpora | Mixed estimate | SE | p | q (BH) | Class at δ_seed |
|---|---|---|---|---|---|---|
| gkt_cpu_ladder/gkt_er - gkt_uniform | -0.0562 | -0.0564 | 0.0087 | 0.0000 | 0.0000 | B- |
| gkt_clique/gkt_clique - gkt_empty | 0.0384 | 0.0384 | 0.0066 | 0.0000 | 0.0000 | B+ |
| gkt_cpu_ladder/gkt_uniform - gkt_empty | 0.0490 | 0.0492 | 0.0100 | 0.0000 | 0.0000 | B+ |
| state/st_support - st_dp | 0.0015 | 0.0017 | 0.0005 | 0.0004 | 0.0055 | S |
| probe_E_rv/rvec_emb - uniform_emb | 0.0042 | 0.0042 | 0.0012 | 0.0004 | 0.0055 | N |
| probe_E/support_emb - empty | 0.0043 | 0.0036 | 0.0011 | 0.0006 | 0.0061 | N |
| other constructions/emb DKT - empty | 0.0037 | 0.0035 | 0.0010 | 0.0007 | 0.0061 | N |
| curriculum/S/st_support - st_er | 0.0029 | 0.0032 | 0.0010 | 0.0009 | 0.0061 | N |
| capacity/tvec_emb - empty | 0.0060 | 0.0056 | 0.0017 | 0.0009 | 0.0061 | N |
| probe_E_rv/rvec_emb - rvecF_emb | 0.0034 | 0.0033 | 0.0010 | 0.0015 | 0.0092 | N |
| gkt_cpu_ladder/gkt_support - gkt_rewired | 0.0248 | 0.0253 | 0.0081 | 0.0017 | 0.0092 | B+ |
| state/st_support - st_empty | 0.0016 | 0.0020 | 0.0006 | 0.0018 | 0.0092 | S |
| probe_E_rv/rvec1_emb - uniform_emb | 0.0043 | 0.0039 | 0.0013 | 0.0020 | 0.0098 | N |
| probe_E/er_emb - uniform_emb | 0.0036 | 0.0033 | 0.0011 | 0.0024 | 0.0110 | N |
| capacity/empty_d128 - empty | 0.0035 | 0.0035 | 0.0012 | 0.0033 | 0.0134 | N |
| sakte_gpu/sakte_support - sakte_empty | 0.0027 | 0.0028 | 0.0009 | 0.0034 | 0.0134 | N |
| gkt_cpu_ladder/gkt_support - gkt_empty | 0.0273 | 0.0281 | 0.0109 | 0.0101 | 0.0374 | B+ |
| other constructions/emb corr - G | -0.0019 | -0.0018 | 0.0007 | 0.0123 | 0.0431 | S |
| probe_E/erR_emb - uniform_emb | 0.0035 | 0.0032 | 0.0013 | 0.0153 | 0.0509 | N |
| gkt_cpu_ladder/gkt_er - gkt_empty | -0.0071 | -0.0072 | 0.0030 | 0.0175 | 0.0550 | N |
| other constructions/emb corr - empty | 0.0023 | 0.0019 | 0.0009 | 0.0272 | 0.0792 | S |
| curriculum/S/st_support - st_unit | 0.0029 | 0.0035 | 0.0016 | 0.0277 | 0.0792 | N |
| gkt_cpu_ladder/gkt_support - gkt_uniform | -0.0218 | -0.0214 | 0.0109 | 0.0495 | 0.1355 | N |
| curriculum/S/st_support - st_clique | 0.0016 | 0.0015 | 0.0008 | 0.0726 | 0.1906 | S |
| probe_E/uniform_emb - empty | -0.0007 | -0.0007 | 0.0004 | 0.0760 | 0.1916 | S |
| other constructions/state DKT - ER | 0.0010 | 0.0011 | 0.0007 | 0.1119 | 0.2712 | E |
| other constructions/state DKT - G | -0.0009 | -0.0008 | 0.0006 | 0.1712 | 0.3921 | E |
| other constructions/state DKT - DP(DKT) | 0.0005 | 0.0006 | 0.0005 | 0.1743 | 0.3921 | E |
| other constructions/state DKT - empty | 0.0007 | 0.0010 | 0.0007 | 0.1932 | 0.4197 | E |
| probe_E_rv/er_emb - rvec_emb | -0.0006 | -0.0009 | 0.0008 | 0.2201 | 0.4384 | E |
| other constructions/emb DKT - ER | 0.0008 | 0.0010 | 0.0008 | 0.2252 | 0.4384 | E |
| state/st_er - st_uniform | -0.0007 | -0.0009 | 0.0007 | 0.2272 | 0.4384 | E |
| other constructions/state DP(DKT) - ER | 0.0005 | 0.0007 | 0.0005 | 0.2296 | 0.4384 | E |
| state/st_dp - st_er | 0.0003 | 0.0005 | 0.0004 | 0.2544 | 0.4714 | E |
| gkt_cpu_ladder/gkt_rewired - gkt_er | 0.0096 | 0.0102 | 0.0092 | 0.2703 | 0.4865 | U |
| curriculum/E/support_emb - clique_emb | 0.0012 | 0.0018 | 0.0017 | 0.2791 | 0.4885 | U |
| curriculum/E/support_emb - unit_emb | 0.0020 | 0.0020 | 0.0019 | 0.2903 | 0.4943 | U |
| other constructions/state corr - G | -0.0018 | -0.0015 | 0.0016 | 0.3407 | 0.5648 | U |
| state/st_uniform - st_empty | 0.0004 | 0.0007 | 0.0007 | 0.3600 | 0.5815 | E |
| curriculum/E/clique_emb - empty | 0.0023 | 0.0022 | 0.0027 | 0.4124 | 0.6257 | U |
| curriculum/S/st_unit - st_empty | -0.0006 | -0.0008 | 0.0010 | 0.4246 | 0.6257 | E |
| probe_E/support_emb - dp_emb | 0.0006 | 0.0008 | 0.0010 | 0.4266 | 0.6257 | E |
| curriculum/S/st_clique - st_empty | 0.0007 | 0.0009 | 0.0011 | 0.4367 | 0.6257 | E |
| curriculum/E/support_emb - er_emb | 0.0016 | 0.0015 | 0.0019 | 0.4370 | 0.6257 | U |
| curriculum/E/unit_emb - empty | 0.0015 | 0.0012 | 0.0017 | 0.4830 | 0.6763 | U |
| gkt_clique/gkt_clique - gkt_uniform | -0.0034 | -0.0034 | 0.0052 | 0.5067 | 0.6939 | U |
| curriculum/E/unit_emb - er_emb | -0.0004 | -0.0006 | 0.0010 | 0.5460 | 0.7318 | E |
| sakte_gpu/sakte_er - sakte_uniform | 0.0006 | 0.0010 | 0.0018 | 0.5832 | 0.7655 | U |
| other constructions/emb corr - ER | -0.0005 | -0.0006 | 0.0011 | 0.6061 | 0.7741 | E |
| state/st_er - st_empty | -0.0002 | -0.0001 | 0.0003 | 0.6144 | 0.7741 | E |
| sakte_gpu/sakte_uniform - sakte_empty | 0.0007 | 0.0009 | 0.0020 | 0.6660 | 0.8096 | U |
| capacity/support_emb - empty_d128 | 0.0008 | 0.0005 | 0.0013 | 0.6774 | 0.8096 | E |
| gkt_clique/gkt_clique - gkt_support | -0.0023 | -0.0023 | 0.0059 | 0.6973 | 0.8096 | U |
| probe_E_rv/rvecF_emb - empty | 0.0001 | 0.0003 | 0.0007 | 0.7049 | 0.8096 | E |
| probe_E_rv/support_emb - rvec_emb | 0.0008 | 0.0006 | 0.0016 | 0.7147 | 0.8096 | E |
| sakte_gpu/sakte_support - sakte_er | 0.0014 | 0.0008 | 0.0023 | 0.7196 | 0.8096 | U |
| probe_E/dp_emb - er_emb | 0.0009 | 0.0003 | 0.0011 | 0.7578 | 0.8376 | E |
| curriculum/S/st_unit - st_er | -0.0000 | -0.0003 | 0.0010 | 0.7827 | 0.8447 | E |
| other constructions/state corr - ER | 0.0001 | 0.0004 | 0.0017 | 0.7911 | 0.8447 | E |
| other constructions/state corr - empty | -0.0001 | 0.0003 | 0.0017 | 0.8425 | 0.8846 | E |
| other constructions/emb DKT - G | -0.0006 | -0.0001 | 0.0011 | 0.9085 | 0.9383 | E |
| probe_E/er_emb - erR_emb | 0.0000 | 0.0001 | 0.0010 | 0.9505 | 0.9658 | E |
| capacity/rvec_emb - empty_d128 | 0.0000 | -0.0000 | 0.0016 | 0.9936 | 0.9936 | E |

## S15.4 Transfer of structure

**Table S15.4** Graph built from all training learners (*G*~100~) given to the probe trained on 10% of them; first attempts unless stated.

| Term | Units | First att. | Mixed 95% | Fit | Fixed t 95% | Cluster bootstrap 95% | All pred. |
|---|---|---|---|---|---|---|---|
| g100_emb - empty | 10 | +0.0016 (8/10) | -0.0021, +0.0052 | ols | -0.0026, +0.0058 | -0.0023, +0.0047 | +0.0026 (10/10) |
| g100_emb - support_emb | 10 | -0.0011 (3/10) | -0.0026, +0.0005 | ols | -0.0028, +0.0007 | -0.0026, +0.0003 | -0.0008 (3/10) |
| g100_emb - dp100_emb | 10 | +0.0007 (6/10) | -0.0016, +0.0030 | ols | -0.0020, +0.0034 | -0.0014, +0.0030 | +0.0008 (7/10) |
| dp100_emb - empty | 10 | +0.0009 (6/10) | -0.0027, +0.0045 | ols | -0.0033, +0.0051 | -0.0026, +0.0043 | +0.0019 (8/10) |
| st_g100 - st_empty | 10 | +0.0014 (6/10) | -0.0014, +0.0041 | ols | -0.0018, +0.0045 | -0.0010, +0.0041 | +0.0002 (7/10) |
| st_g100 - st_support | 10 | -0.0005 (3/10) | -0.0013, +0.0004 | ols | -0.0015, +0.0005 | -0.0012, +0.0004 | -0.0003 (5/10) |
| st_g100 - st_dp100 | 10 | +0.0017 (6/10) | -0.0010, +0.0044 | ols | -0.0014, +0.0048 | -0.0008, +0.0043 | +0.0000 (7/10) |
| st_dp100 - st_empty | 10 | -0.0003 (5/10) | -0.0023, +0.0016 | ols | -0.0026, +0.0019 | -0.0023, +0.0014 | +0.0002 (6/10) |
| support_emb - empty | 20 | +0.0031 (7/10) | -0.0001, +0.0062 | mixed | -0.0006, +0.0067 | -0.0002, +0.0063 | +0.0026 (8/10) |
| st_support - st_empty | 20 | +0.0016 (8/10) | +0.0002, +0.0031 | mixed | +0.0001, +0.0032 | +0.0001, +0.0034 | +0.0004 (6/10) |

Overlap of the subsample graph with *G*~100~ on the units of Table S15.4:

| Corpus | Edges of G | Edges of G100 | Recall | Precision | Jaccard |
|---|---|---|---|---|---|
| Algebra 2005 | 98 | 222 | 0.423 | 0.959 | 0.441 |
| ASSISTments 2009 | 59 | 557 | 0.088 | 0.831 | 0.102 |
| ASSISTments 2012 | 235 | 1,925 | 0.097 | 0.791 | 0.119 |
| ASSISTments 2015 | 79 | 988 | 0.074 | 0.924 | 0.080 |
| ASSISTments 2017 | 182 | 1,233 | 0.101 | 0.681 | 0.143 |
| Bridge to Alg. 2006 | 112 | 327 | 0.343 | 1 | 0.343 |
| EdNet KT1 | 587 | 3,378 | 0.114 | 0.656 | 0.169 |
| Eedi | 248 | 664 | 0.292 | 0.782 | 0.365 |
| Junyi Academy | 146 | 958 | 0.133 | 0.870 | 0.149 |
| Statics 2011 | 39 | 116 | 0.336 | 1 | 0.336 |

## S15.5 PDKT: the graph in the training loss

**Table S15.5** Null-graph ladder inside PDKT (Eq. 5, λ = 1, at most 512 edges per update), 10%, update budget. pdkt_empty is DKT; the last row compares DKT with the empty-graph probe.

| Term | Units | First att. | Mixed 95% | Fit | Fixed t 95% | Cluster bootstrap 95% | All pred. |
|---|---|---|---|---|---|---|---|
| pdkt_er - pdkt_empty | 10 | -0.0068 (1/10) | -0.0124, -0.0011 | ols | -0.0133, -0.0002 | -0.0123, -0.0016 | +0.0014 (6/10) |
| pdkt_dp - pdkt_er | 10 | -0.0005 (4/10) | -0.0073, +0.0063 | ols | -0.0084, +0.0073 | -0.0069, +0.0060 | +0.0012 (4/10) |
| pdkt_support - pdkt_dp | 10 | +0.0033 (8/10) | -0.0016, +0.0083 | ols | -0.0024, +0.0090 | -0.0014, +0.0080 | +0.0002 (7/10) |
| pdkt_support - pdkt_empty | 10 | -0.0040 (3/10) | -0.0086, +0.0007 | ols | -0.0093, +0.0014 | -0.0084, +0.0003 | +0.0028 (6/10) |
| pdkt_support - pdkt_er | 10 | +0.0028 (7/10) | -0.0027, +0.0083 | ols | -0.0036, +0.0092 | -0.0026, +0.0080 | +0.0014 (8/10) |
| pdkt_empty - empty | 10 | -0.0112 (3/10) | -0.0226, +0.0002 | ols | -0.0243, +0.0020 | -0.0221, -0.0004 | -0.0154 (1/10) |

## S15.6 Longer histories

**Table S15.6** support_emb - empty, larger corpora (this release), embedding refinement, 10%; units: ASSISTments 2012 s0, Eedi s0.

| Term | Units | First att. | Mixed 95% | Fit | Fixed t 95% | Cluster bootstrap 95% | All pred. |
|---|---|---|---|---|---|---|---|
| 500 interactions per learner | 2 | -0.0000 (1/2) | -0.0015, +0.0014 | ols | -0.0095, +0.0094 | -0.0008, +0.0007 |  |
| 200 interactions, same units | 2 | +0.0024 (2/2) | -0.0020, +0.0068 | ols | -0.0263, +0.0311 | +0.0001, +0.0047 |  |

**Table S15.6** support_emb - empty, smallest corpora (release 3.11.0), embedding refinement, 10%; units: Algebra 2005 s0, Algebra 2005 s1, Algebra 2005 s2, Algebra 2005 s3, Algebra 2005 s4, Statics 2011 s0, Statics 2011 s1, Statics 2011 s2, Statics 2011 s3, Statics 2011 s4.

| Term | Units | First att. | Mixed 95% | Fit | Fixed t 95% | Cluster bootstrap 95% | All pred. |
|---|---|---|---|---|---|---|---|
| 500 interactions per learner | 10 | +0.0003 (1/2) | -0.0047, +0.0053 | mixed | -0.0065, +0.0071 | -0.0043, +0.0048 |  |
| 200 interactions, same units | 10 | +0.0023 (2/2) | -0.0027, +0.0074 | mixed | -0.0199, +0.0245 | -0.0020, +0.0078 |  |

## S15.7 XES3G5M without multi-KC questions

H1 (support graph at both entry points against the empty graph, 10%, update budget) on XES3G5M with every answer to a multi-KC question dropped (data/xes3g5m1.parquet, prep_r19.py; same learners), against the first-KC rule of the replication on the same subsamples. First attempts: subsample 0: -0.0006 without multi-KC questions, +0.0003 with the first-KC rule; subsample 1: +0.0013 without multi-KC questions, +0.0017 with the first-KC rule. All predictions: subsample 0: -0.0006 and -0.0002; subsample 1: +0.0020 and +0.0021.

# S16 Next-KC decisions

## S16.1 Agreement by corpus and rule

**Table S16.1** Share of decisions in which two models choose the same next KC, 25% of the training learners, subsample 0; mean over the pairs of seeds 0, 100, 200. Near-ties: mean number of other candidates within 0.01 of the best value of the control (seed 0) under the target rule.

| Corpus | Candidates (median) | Near-ties | C seeds | target | C seeds | target_tb0.01 | C seeds | target_tb0.02 | C seeds | easiest | G-C crossed | target | G-C crossed | target_tb0.01 | G-C crossed | target_tb0.02 | G-C crossed | easiest | G-C shared | target | G-C shared | target_tb0.01 | G-C shared | target_tb0.02 | G-C shared | easiest |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ASSISTments 2009 | 106 | 3 | 0.018 | 0.044 | 0.064 | 0.149 | 0.026 | 0.046 | 0.068 | 0.200 | 0.069 | 0.116 | 0.171 | 0.466 |
| ASSISTments 2012 | 252 | 10 | 0.007 | 0.033 | 0.054 | 0.045 | 0.009 | 0.033 | 0.061 | 0.067 | 0.020 | 0.064 | 0.108 | 0.339 |
| ASSISTments 2015 | 96 | 3 | 0.019 | 0.038 | 0.058 | 0.102 | 0.020 | 0.039 | 0.064 | 0.097 | 0.024 | 0.067 | 0.114 | 0.333 |
| ASSISTments 2017 | 70 | 0.989 | 0.141 | 0.156 | 0.178 | 0.230 | 0.164 | 0.172 | 0.193 | 0.281 | 0.294 | 0.307 | 0.345 | 0.547 |
| Junyi Academy | 685 | 21 | 0.003 | 0.017 | 0.034 | 0.052 | 0.002 | 0.018 | 0.036 | 0.065 | 0.003 | 0.032 | 0.061 | 0.243 |
| Algebra 2005 | 102 | 2 | 0.014 | 0.024 | 0.046 | 0.742 | 0.016 | 0.036 | 0.062 | 0.756 | 0.022 | 0.059 | 0.103 | 0.863 |
| Eedi | 51 | 2 | 0.179 | 0.210 | 0.264 | 0.721 | 0.151 | 0.183 | 0.230 | 0.565 | 0.219 | 0.275 | 0.341 | 0.655 |
| EdNet KT1 | 123 | 4 | 0.065 | 0.104 | 0.148 | 0.388 | 0.068 | 0.114 | 0.177 | 0.347 | 0.114 | 0.179 | 0.253 | 0.538 |
| Bridge to Alg. 2006 | 485 | 11 | 0.004 | 0.017 | 0.037 | 0.005 | 0.002 | 0.020 | 0.037 | 0.021 | 0.005 | 0.031 | 0.053 | 0.100 |
| Statics 2011 | 73 | 2 | 0.016 | 0.041 | 0.058 | 0.137 | 0.021 | 0.035 | 0.050 | 0.066 | 0.040 | 0.036 | 0.068 | 0.197 |

## S16.2 Regret against first-attempt AUC

**Table S16.2** Regression of regret on first-attempt AUC over every decision model (5%, 25%, 100%; corpus fixed effects; standard errors clustered by corpus).

| Simulator | rule | slope | lo | hi | p | n | auc_sd_within | auc_range_within_median | mean_regret |
|---|---|---|---|---|---|---|---|---|
| rasch|target | -0.2721 | -0.4465 | -0.0977 | 0.0022 | 280 | 0.0195 | 0.0739 | 0.1261 |
| rasch|easiest | -0.9056 | -2 | -0.0673 | 0.0342 | 280 | 0.0195 | 0.0739 | 0.1130 |
| rasch|target_tb0.01 | -0.2665 | -0.4286 | -0.1045 | 0.0013 | 280 | 0.0195 | 0.0739 | 0.1293 |
| rasch|target_tb0.02 | -0.2574 | -0.4470 | -0.0677 | 0.0078 | 280 | 0.0195 | 0.0739 | 0.1304 |
| pfa|target | -0.0737 | -0.1977 | 0.0502 | 0.2435 | 280 | 0.0195 | 0.0739 | 0.1120 |
| pfa|easiest | -0.7177 | -1 | -0.1744 | 0.0096 | 280 | 0.0195 | 0.0739 | 0.0910 |
| pfa|target_tb0.01 | -0.0257 | -0.1729 | 0.1216 | 0.7326 | 280 | 0.0195 | 0.0739 | 0.1108 |
| pfa|target_tb0.02 | -0.0253 | -0.1918 | 0.1412 | 0.7658 | 280 | 0.0195 | 0.0739 | 0.1099 |

**Table S16.3** Mean regret by corpus, simulator and training size of the decision models.

| Corpus | Simulator | Training % | Models | First-att. AUC | Regret, target | Regret, stabilized 0.01 | Regret, easiest |
|---|---|---|---|---|---|---|---|
| Algebra 2005 | pfa | 5 | 3 | 0.6802 | 0.1252 | 0.1317 | 0.0613 |
| Algebra 2005 | pfa | 25 | 16 | 0.7465 | 0.1209 | 0.1270 | 0.0374 |
| Algebra 2005 | pfa | 100 | 9 | 0.7821 | 0.1269 | 0.1394 | 0.0508 |
| Algebra 2005 | rasch | 5 | 3 | 0.6802 | 0.1464 | 0.1563 | 0.0906 |
| Algebra 2005 | rasch | 25 | 16 | 0.7465 | 0.1341 | 0.1405 | 0.0576 |
| Algebra 2005 | rasch | 100 | 9 | 0.7821 | 0.1337 | 0.1427 | 0.0766 |
| ASSISTments 2009 | pfa | 5 | 3 | 0.6887 | 0.1337 | 0.1242 | 0.2125 |
| ASSISTments 2009 | pfa | 25 | 16 | 0.7414 | 0.1188 | 0.1185 | 0.1218 |
| ASSISTments 2009 | pfa | 100 | 9 | 0.7721 | 0.1178 | 0.1181 | 0.1263 |
| ASSISTments 2009 | rasch | 5 | 3 | 0.6887 | 0.1612 | 0.1413 | 0.2058 |
| ASSISTments 2009 | rasch | 25 | 16 | 0.7414 | 0.1268 | 0.1227 | 0.1264 |
| ASSISTments 2009 | rasch | 100 | 9 | 0.7721 | 0.1231 | 0.1206 | 0.1339 |
| ASSISTments 2012 | pfa | 5 | 3 | 0.6550 | 0.1125 | 0.1126 | 0.1517 |
| ASSISTments 2012 | pfa | 25 | 16 | 0.6876 | 0.1065 | 0.1032 | 0.0950 |
| ASSISTments 2012 | pfa | 100 | 9 | 0.7121 | 0.1058 | 0.1046 | 0.0927 |
| ASSISTments 2012 | rasch | 5 | 3 | 0.6550 | 0.1493 | 0.1551 | 0.1645 |
| ASSISTments 2012 | rasch | 25 | 16 | 0.6876 | 0.1345 | 0.1421 | 0.0916 |
| ASSISTments 2012 | rasch | 100 | 9 | 0.7121 | 0.1310 | 0.1333 | 0.0927 |
| ASSISTments 2015 | pfa | 5 | 3 | 0.6706 | 0.0925 | 0.0979 | 0.0257 |
| ASSISTments 2015 | pfa | 25 | 16 | 0.7144 | 0.0983 | 0.1013 | 0.0232 |
| ASSISTments 2015 | pfa | 100 | 9 | 0.7344 | 0.0974 | 0.0982 | 0.0151 |
| ASSISTments 2015 | rasch | 5 | 3 | 0.6706 | 0.1215 | 0.1287 | 0.0368 |
| ASSISTments 2015 | rasch | 25 | 16 | 0.7144 | 0.1191 | 0.1264 | 0.0323 |
| ASSISTments 2015 | rasch | 100 | 9 | 0.7344 | 0.1163 | 0.1197 | 0.0257 |
| ASSISTments 2017 | pfa | 5 | 3 | 0.6937 | 0.1451 | 0.1390 | 0.2424 |
| ASSISTments 2017 | pfa | 25 | 16 | 0.7188 | 0.1109 | 0.1090 | 0.1973 |
| ASSISTments 2017 | pfa | 100 | 9 | 0.7353 | 0.0984 | 0.0949 | 0.1432 |
| ASSISTments 2017 | rasch | 5 | 3 | 0.6937 | 0.1465 | 0.1443 | 0.1906 |
| ASSISTments 2017 | rasch | 25 | 16 | 0.7188 | 0.1074 | 0.1085 | 0.1512 |
| ASSISTments 2017 | rasch | 100 | 9 | 0.7353 | 0.0965 | 0.0949 | 0.1062 |
| Bridge to Alg. 2006 | pfa | 5 | 3 | 0.6869 | 0.1520 | 0.1394 | 0.1314 |
| Bridge to Alg. 2006 | pfa | 25 | 16 | 0.7223 | 0.1521 | 0.1386 | 0.1083 |
| Bridge to Alg. 2006 | pfa | 100 | 9 | 0.7541 | 0.1532 | 0.1398 | 0.1103 |
| Bridge to Alg. 2006 | rasch | 5 | 3 | 0.6869 | 0.1438 | 0.1751 | 0.3056 |
| Bridge to Alg. 2006 | rasch | 25 | 16 | 0.7223 | 0.1408 | 0.1660 | 0.2499 |
| Bridge to Alg. 2006 | rasch | 100 | 9 | 0.7541 | 0.1401 | 0.1636 | 0.2438 |
| EdNet KT1 | pfa | 5 | 3 | 0.6673 | 0.0893 | 0.0822 | 0.1698 |
| EdNet KT1 | pfa | 25 | 16 | 0.6826 | 0.0618 | 0.0604 | 0.0563 |
| EdNet KT1 | pfa | 100 | 9 | 0.6918 | 0.0688 | 0.0649 | 0.0483 |
| EdNet KT1 | rasch | 5 | 3 | 0.6673 | 0.1066 | 0.1072 | 0.1544 |
| EdNet KT1 | rasch | 25 | 16 | 0.6826 | 0.0875 | 0.0881 | 0.0541 |
| EdNet KT1 | rasch | 100 | 9 | 0.6918 | 0.0862 | 0.0859 | 0.0448 |
| Eedi | pfa | 5 | 3 | 0.7149 | 0.1011 | 0.0996 | 0.1028 |
| Eedi | pfa | 25 | 16 | 0.7337 | 0.0753 | 0.0733 | 0.0360 |
| Eedi | pfa | 100 | 9 | 0.7432 | 0.0804 | 0.0800 | 0.0386 |
| Eedi | rasch | 5 | 3 | 0.7149 | 0.1369 | 0.1261 | 0.1289 |
| Eedi | rasch | 25 | 16 | 0.7337 | 0.0968 | 0.0895 | 0.0429 |
| Eedi | rasch | 100 | 9 | 0.7432 | 0.0987 | 0.0933 | 0.0446 |
| Junyi Academy | pfa | 5 | 3 | 0.7369 | 0.1416 | 0.1328 | 0.1671 |
| Junyi Academy | pfa | 25 | 16 | 0.7851 | 0.1459 | 0.1449 | 0.0440 |
| Junyi Academy | pfa | 100 | 9 | 0.8112 | 0.1502 | 0.1573 | 0.0364 |
| Junyi Academy | rasch | 5 | 3 | 0.7369 | 0.1939 | 0.1911 | 0.3085 |
| Junyi Academy | rasch | 25 | 16 | 0.7851 | 0.1811 | 0.1724 | 0.0608 |
| Junyi Academy | rasch | 100 | 9 | 0.8112 | 0.1572 | 0.1535 | 0.0541 |
| Statics 2011 | pfa | 5 | 3 | 0.7188 | 0.1186 | 0.1268 | 0.2003 |
| Statics 2011 | pfa | 25 | 16 | 0.7660 | 0.1135 | 0.1120 | 0.1430 |
| Statics 2011 | pfa | 100 | 9 | 0.7829 | 0.1186 | 0.1207 | 0.1480 |
| Statics 2011 | rasch | 5 | 3 | 0.7188 | 0.1337 | 0.1467 | 0.2384 |
| Statics 2011 | rasch | 25 | 16 | 0.7660 | 0.1254 | 0.1303 | 0.1830 |
| Statics 2011 | rasch | 100 | 9 | 0.7829 | 0.1322 | 0.1382 | 0.2184 |

Weights of the learning simulator (log(1 + successes), log(1 + failures), smoothed proportion correct) by corpus: ASSISTments 2009 +0.551, -0.563, +1.123; ASSISTments 2012 +0.509, -0.547, +0.807; ASSISTments 2015 +0.441, -0.457, +0.507; ASSISTments 2017 +0.143, -0.098, +3.056; Junyi Academy +0.228, -0.324, +1.777; Algebra 2005 +0.672, -0.738, +0.472; Eedi +0.775, -0.859, +0.009; EdNet KT1 +0.490, -0.381, +0.668; Bridge to Alg. 2006 +0.562, -0.621, +0.875; Statics 2011 +0.510, -0.464, +1.828.
