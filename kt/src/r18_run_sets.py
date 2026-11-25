"""Release 4.1.0: inventory of the 2,025 runs that precede the run sets of Supplementary Table S6.10, and a manifest of
every run file in the archive (no new runs).

Sets (Section V-B of the article):
  learning curve  kt/r2/runs, the 957 runs with src = R2 in kt/r2/results/r2_runs.csv (release 2.0.0)
  controls        kt/runs_r5, 566 runs of the controls under the epoch budget
  update budget   kt/runs_r6, 502 runs under the update budget of Section VI-B
Writes results_r18/run_sets_r18.csv (one row per set, model and training fraction) and results_r18/run_manifest.csv
(one row per run file of the archive with its folder, corpus, fraction, subsample, condition and whether it is counted
among the 3,908 runs of the article).
"""
import os, glob, json
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r18'; os.makedirs(OUT, exist_ok=True)


def subs(df):
    per = {c: ' '.join(str(int(s)) for s in sorted(g['sub'].unique())) for c, g in df.groupby('data')}
    if len(per) == 10 and len(set(per.values())) == 1: return 'all ten corpora: ' + next(iter(per.values()))
    return '; '.join(f'{c}: {v}' for c, v in per.items())


def seeds(df):
    return 'subsample' if (df['seed'] == df['sub']).all() else ' '.join(str(int(x)) for x in sorted(df['seed'].unique()))


rows = []
r2 = pd.read_csv(f'{REPO}/r2/results/r2_runs.csv'); r2 = r2[r2.src == 'R2'].copy()
r2['model'] = r2.cfg.str.split('/').str[0].replace({'P': 'FKG-KT (probe)'})
for (m, fr), g in r2.groupby(['model', 'frac']):
    rows.append(dict(set='learning curve (kt/r2/runs)', release='2.0.0', model=m, fraction=fr, device='', seeds=seeds(g),
                     conditions=' '.join(sorted(g.cfg.unique())), subsamples=subs(g), runs=len(g)))
for folder, lab in (('runs_r5', 'controls, epoch budget (kt/runs_r5)'), ('runs_r6', 'releases 3.1.0–3.4.0 (kt/runs_r6)')):
    recs = []
    for f in glob.glob(f'{REPO}/{folder}/*.json'):
        d = json.load(open(f))
        if 'test_auc' in d: recs.append(dict(data=d['data'], frac=d['frac'], sub=d['sub'], seed=d.get('seed'), cond=d.get('cond'), budget=d.get('budget', 'epochs'), device=d.get('device', '')))
    df = pd.DataFrame(recs)
    for (fr, b), g in df.groupby(['frac', 'budget']):
        rows.append(dict(set=lab + ('' if folder == 'runs_r5' else f', {b}'), release=('2.1.0' if folder == 'runs_r5' else '3.1.0–3.4.0'), model='FKG-KT (probe)', fraction=fr, device=' '.join(sorted(set(g.device) - {''})), seeds=seeds(g),
                         conditions=' '.join(sorted(g.cond.unique())), subsamples=subs(g), runs=len(g)))
sets = pd.DataFrame(rows); sets.to_csv(f'{OUT}/run_sets_r18.csv', index=False)
print(sets.groupby('set').runs.sum(), sets.runs.sum())

# manifest of every run file
man = []
for f in sorted(glob.glob(f'{REPO}/**/*.json', recursive=True)):
    rel = os.path.relpath(f, REPO); folder = os.path.dirname(rel)
    if folder.startswith('results') or '/results' in folder: continue
    try: d = json.load(open(f))
    except Exception: continue
    if not (isinstance(d, dict) and 'test_auc' in d): continue
    man.append(dict(folder=folder, file=os.path.basename(rel), data=d.get('data'), frac=d.get('frac'), sub=d.get('sub'),
                    condition=d.get('cond', d.get('cfg', d.get('graph'))), device=d.get('device', ''),
                    test_auc=d.get('test_auc'), test_auc_first=d.get('test_auc_first', d.get('test_auc_newkc'))))
man = pd.DataFrame(man)
man.to_csv(f'{OUT}/run_manifest.csv', index=False)
print(man.folder.value_counts().sort_index())
