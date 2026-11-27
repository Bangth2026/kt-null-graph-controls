[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24492979.svg)](https://doi.org/10.5281/zenodo.24492979)

# Null-graph controls for concept graphs in knowledge tracing

Code, run files and analysis scripts for

> Hai Bang Truong. *Do log-built concept graphs help KC-level learner models?
> Entry-point null graphs, a positive control and decision-level tests.* Manuscript under revision for User Modeling and User-Adapted Interaction (UMUAI).

The study fixes a KC-level probe tracer, changes only the graph it receives and measures the graph against an empty-graph
control that keeps every other parameter, a ladder of null graphs (uniform mixing, Erdős–Rényi, degree-preserving and
unit-preserving rewiring), fixed random codes per KC and capacity controls, on ten public corpora, with a replication on two
held-out corpora under a pre-specified plan (`kt/prereg/`). Release 5.0.0 adds the runs and analyses of the UMUAI revision:
a graph built from all training learners given to a tracer trained on 10% of them (transfer of structure), the null-graph
ladder inside PDKT (graph in the training loss), three training seeds per subsample, a 500-interaction cap on two larger
corpora, XES3G5M without multi-KC questions, simple first-attempt references (KC rate, PFA-style model) with pooled,
per-learner and within-KC AUC, graph quality by training size, and next-KC decisions under a Rasch and a learning
simulator, with a stabilized target rule (tolerance band and curriculum tie-break).

## Layout

Everything is under `kt/` so that the paths cited in the manuscript and its Supplementary Material (`kt/src/...`,
`kt/runs_r*/...`, `kt/results_r*/...`) hold as written.

| Path | Content |
|---|---|
| `kt/src/` | data preparation (`prep.py`, `prep_junyi_lowmem.py`, `prep_r14.py`), models (`models.py`), the probe and its controls (`r5_controls.py` to `r16_controls.py`), analyses (`r*_analyse.py`, `r15_regret.py`, `r16_decision.py`, `r17_between_corpora.py`), figures |
| `kt/runs*/`, `kt/r2/` | one JSON file per training run (and prediction files where kept) |
| `kt/results*/` | analysis outputs behind every table and figure |
| `kt/src/r19_*.py`, `kt/runs_r19/`, `kt/results_r19/` | release 5.0.0 (UMUAI revision): runs, analyses and their outputs |
| `kt/src/r20_*.py`, `kt/src/prep_r20.py`, `kt/runs_r20/`, `kt/runs_r20_sim/`, `kt/results_r20/` | release 6.0.0 (second UMUAI version): NCA-KT, item level, DBE-KT22, simulated worlds and closed loop |
| `kt/src/r21_*.py`, `kt/runs_r21_sim/`, `kt/runs_r21/`, `kt/results_r21/`, `kt/graphs_r21/` | release 7.0.0 (third UMUAI version): positive control in the simulated worlds, gated decisions, XES3G5M KC routes, power, seed ensembles, KC-rate rule |
| `kt/src/r22_*.py`, `kt/runs_r22_sim/`, `kt/runs_r22/`, `kt/results_r22/` | release 8.0.0 (fifth UMUAI version): third simulated world, MIRT simulator, chance-corrected agreement and restricted candidate sets, closed loop with 24 tracers per world, partial Q-matrix perturbation, t-based power |
| `kt/prereg/` | the analysis plan of the replication and its SHA-256 checksum |
| `kt/README.md` | every release, script and run set, in the order in which they were made |
| `tests/test_kt_graphs.py` | unit tests of the graph constructions and null graphs |

The probe is called FKG-KT in file names, after its first release; Supplementary Table S7.0 maps condition names to the
symbols of the manuscript.

## Running

```bash
pip install -r requirements.txt torch==2.15.0 --extra-index-url https://download.pytorch.org/whl/cpu
export KT_ROOT=$PWD/kt KT_RAW=/path/to/raw NT=1
python kt/src/prep.py statics      # one corpus per call; sources in kt/README.md (release 3.7.0)
python kt/src/r16_controls.py --data statics11 --sub 0 --cond empty   # reproduces runs_r6/statics11__f10__s0__empty__steps.json
PYTHONPATH=. pytest tests/test_kt_graphs.py
```

The scripts expect the checkout at `/home/claude/kt` (as when the runs were made) or `KT_ROOT` set; data files are not
redistributed.

## Release 2.0.0 scripts

A few scripts of the first release (`kt/src/graphs.py` beyond `load`, `kt/src/nulltests.py`, `kt/r2/src/`, `kt/r2/pyktp/`)
built weighted graphs and community partitions with a separate graph package that is not part of this repository and is not
used by any current analysis; their imports are optional, and their run files are kept. Every number of the current
manuscript is produced by scripts that run without it.

## Provenance

Releases 2.0.0 to 3.14.0 of this code were developed in the directory `kt/` of the author's repository
`fuzzy-kg-modularity` and archived on Zenodo with it. Version 4.0.0 moves the code of this study to its own repository and
Zenodo record, as a copy of `kt/` without the commit history of the earlier repository; the content of `kt/` is unchanged except for the optional imports above, and the release-2.0.0-era file
`kt/theory/`, which belongs to another study, is not included.

## Archive

Zenodo: https://doi.org/10.5281/zenodo.24492979 (all versions); version 4.0.0: https://doi.org/10.5281/zenodo.24492980. Version 4.1.0 adds `kt/src/r18_*.py` and
`kt/results_r18/` (no new runs). Version 5.0.0 adds `kt/src/r19_*.py`, `kt/src/prep_r19.py`, `kt/runs_r19/` and `kt/results_r19/`
(new runs; see `kt/README.md`, release 5.0.0). Version 6.0.0 adds `kt/src/r20_*.py`, `kt/src/prep_r20.py`, `kt/runs_r20/`, `kt/runs_r20_sim/` and `kt/results_r20/` (new runs; see `kt/README.md`, release 6.0.0). Version 7.0.0 adds `kt/src/r21_*.py`, `kt/runs_r21_sim/`, `kt/runs_r21/`, `kt/results_r21/` and `kt/graphs_r21/` (new runs; see `kt/README.md`, release 7.0.0). Version 8.0.0 adds `kt/src/r22_*.py`, `kt/runs_r22_sim/`, `kt/runs_r22/` and `kt/results_r22/` (new runs; see `kt/README.md`, release 8.0.0).

## Raw data used to rebuild the corpora (releases 5.0.0 to 8.0.0)

Public copies on Hugging Face: `Lucy9999/assist2009` (`skill_builder_data.csv`, used as `skill_builder_data_corrected.csv`),
`Lucy9999/assist2012` (zip of the 2012-2013 file), `Lucy9999/assist2015`, `Lucy9999/assist2017`
(`Released Full Dataset/anonymized_full_release_competition_dataset.csv`), `Lucy9999/junyi2015`, `Lucy9999/AL2005`,
`Lucy9999/BD2006`, `Lucy9999/NIPS34` (`data.zip`), `Lucy9999/statics2011`, `Lucy9999/EdNet-KT1`, `Unggi/ednet5000_raw_data`
(`ednet_content/questions.csv`), `Atomi/XES3G5M_interaction_sequences` and, from release 6.0.0, `Unggi/dbe-kt22_raw_data` (DBE-KT22). Release 7.0.0 uses the KC routes of the original XES3G5M release (`metadata/questions.json`, `kc_routes_map.json`; github.com/ai4ed/XES3G5M). Every rebuilt corpus matches Table 1 of the
manuscript, and the empty-graph control on subsample 0 reproduces its archived test AUC to the last digit (`kt/runs_r19/repro/`).

## Citation

See `CITATION.cff`. License: MIT (code); the corpora keep their own licenses.
