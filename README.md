# Null-graph controls for concept graphs in knowledge tracing

Code, run files and analysis scripts for

> Hai Bang Truong. *When Does a Concept Graph Help a Knowledge Tracer? Null-Graph Controls by Entry Point for KC-Level First
> Attempts.* Manuscript prepared for IEEE Transactions on Learning Technologies.

The study fixes a KC-level probe tracer, changes only the graph it receives and measures the graph against an empty-graph
control that keeps every other parameter, a ladder of null graphs (uniform mixing, Erdős–Rényi, degree-preserving and
unit-preserving rewiring), fixed random codes per KC and capacity controls, on ten public corpora, with a replication on two
held-out corpora under a pre-specified plan (`kt/prereg/`).

## Layout

Everything is under `kt/` so that the paths cited in the manuscript and its Supplementary Material (`kt/src/...`,
`kt/runs_r*/...`, `kt/results_r*/...`) hold as written.

| Path | Content |
|---|---|
| `kt/src/` | data preparation (`prep.py`, `prep_junyi_lowmem.py`, `prep_r14.py`), models (`models.py`), the probe and its controls (`r5_controls.py` to `r16_controls.py`), analyses (`r*_analyse.py`, `r15_regret.py`, `r16_decision.py`, `r17_between_corpora.py`), figures |
| `kt/runs*/`, `kt/r2/` | one JSON file per training run (and prediction files where kept) |
| `kt/results*/` | analysis outputs behind every table and figure |
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
Zenodo record; the content of `kt/` is unchanged except for the optional imports above, and the release-2.0.0-era file
`kt/theory/`, which belongs to another study, is not included.

## Archive

Zenodo: https://doi.org/10.5281/zenodo.24492767 (version 4.0.0).

## Citation

See `CITATION.cff`. License: MIT (code); the corpora keep their own licenses.
