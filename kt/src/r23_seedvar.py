"""Release 9.0.0: does the graph reduce seed variance? (reviewer comment R6 of the second round)

For every unit of the seed ensembles of r21_stats.ensembles (three training seeds each of the empty-graph control and of
the probe with the support graph in the embedding refinement, 10% of the training learners), on first attempts at a KC:
  auc_sd     standard deviation over the three seeds of the first-attempt AUC
  pred_sd    mean over first-attempt test predictions of the standard deviation over seeds of the predicted probability
for the control and for the graph. If the graph acts partly as a variance-reducing regularizer, both should be lower
with the graph. Corpus means first, then the mean over corpora; Wilcoxon signed-rank over corpora.
Output: results_r23/seed_variance.json
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r21_stats import U20, pred_of, REPO
from r14_common import load_runs
OUT = f'{REPO}/results_r23'; os.makedirs(OUT, exist_ok=True)

SEEDS = {}
for path in (f'{REPO}/runs_r6', f'{REPO}/runs_r9', f'{REPO}/runs_r19'):
    for k, v in load_runs([path]).items():
        if k[1] == 0.1 and k[3] in ('support', 'empty') and k[4] in (0, 1, 2): SEEDS.setdefault(k, v)
rows = []
for c, s in U20:
    ks = {m: [(c, 0.1, s, m, r, 0.002) for r in (0, 1, 2)] for m in ('support', 'empty')}
    if not all(k in SEEDS for v in ks.values() for k in v): continue
    P = {m: [pred_of(SEEDS[k]) for k in kk] for m, kk in ks.items()}
    if any(a is None for v in P.values() for a in v): continue
    y = P['empty'][0]['y']; new = P['empty'][0]['new'].astype(bool)
    if not all(len(a['y']) == len(y) and (a['y'] == y).all() for v in P.values() for a in v): continue
    r = dict(data=c, sub=s)
    for m in P:
        A = np.stack([a['p'][new] for a in P[m]])
        r[f'{m}_auc_sd'] = float(np.std([roc_auc_score(y[new], a) for a in A], ddof=1)); r[f'{m}_pred_sd'] = float(A.std(0, ddof=1).mean())
    rows.append(r)
d = pd.DataFrame(rows); cm = d.groupby('data').mean(numeric_only=True)
out = dict(units=len(d), corpora=int(d.data.nunique()), per_unit=rows)
for k in ('auc_sd', 'pred_sd'):
    diff = cm[f'support_{k}'] - cm[f'empty_{k}']
    out[k] = dict(control=float(cm[f'empty_{k}'].mean()), graph=float(cm[f'support_{k}'].mean()), diff=float(diff.mean()),
                  lower_with_graph=int((diff < 0).sum()), p=float(wilcoxon(diff).pvalue))
json.dump(out, open(f'{OUT}/seed_variance.json', 'w'), indent=1)
print({k: v for k, v in out.items() if k != 'per_unit'})
