"""Release 4.1.0: reference points for the agreement of next-KC choices (no new runs).

Same decision models and pairs as the crossed-seed analysis of r14_analyse.py (25% of the training learners, update
budget, subsample 0; empty-graph control and support graph at both entry points, seeds 0, 100 and 200; pairs with
different seeds), same decisions (up to 400 test learners at five positions; candidates are the KCs the learner has not
yet attempted) and the same two rules (target: predicted probability closest to 0.65; easiest: highest probability).

Added for every pair and decision:
  chance      agreement of two independent uniform choices among the candidates, 1 / (number of candidates);
  top5        the KC chosen by model A is among the five best candidates of model B under the same rule;
  near_0.01   model B rates the KC chosen by model A within 0.01 of its own best value under the same rule (for the target
              rule, | p_B(choice_A) - 0.65 | - min | p_B - 0.65 | <= 0.01; for the easiest rule, max p_B - p_B(choice_A)
              <= 0.01), averaged over both directions of the pair;
  ties_0.01   number of candidates that model B rates within 0.01 of its own best value under the target rule.
Pairs: graph vs control with crossed seeds (G_j vs E_k, j != k) and control vs control (E_j vs E_k, j != k).
Means per corpus, then over corpora. Writes results_r18/r18_decision_agreement.json.
"""
import os, glob, json, itertools
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
OUT = f'{REPO}/results_r18'; os.makedirs(OUT, exist_ok=True)
TGT = 0.65
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']


def path(c, m, r):
    tag = f'{c}__f25__s0__{m}' + (f'__rep{r}' if r else '') + '__steps_cand.npz'
    for d in ('runs_r13', 'runs_r14'):
        p = f'{REPO}/{d}/{tag}'
        if os.path.exists(p): return p
    return None


def pair_stats(A, B, seen):
    ok = ~np.isnan(A[..., 0].astype(np.float32)); rows = []
    for i, t in zip(*np.nonzero(ok)):
        cand = ~seen[i, t]
        if cand.sum() < 2 or np.isnan(B[i, t, 0]): continue
        idx = np.flatnonzero(cand); a = A[i, t, idx].astype(np.float64); b = B[i, t, idx].astype(np.float64)
        n = len(idx); k = min(5, n); r = {'n_cand': n, 'chance': 1.0 / n}
        for rule in ('target', 'easiest'):
            if rule == 'target': va, vb = -np.abs(a - TGT), -np.abs(b - TGT)
            else: va, vb = a, b
            ca, cb = int(np.argmax(va)), int(np.argmax(vb))
            r[f'{rule}_same'] = float(ca == cb)
            r[f'{rule}_top5'] = 0.5 * (float(ca in set(np.argsort(-vb)[:k])) + float(cb in set(np.argsort(-va)[:k])))
            r[f'{rule}_near_0.01'] = 0.5 * (float(vb.max() - vb[ca] <= 0.01) + float(va.max() - va[cb] <= 0.01))
            if rule == 'target': r['ties_0.01'] = 0.5 * (float((vb.max() - vb <= 0.01).sum()) + float((va.max() - va <= 0.01).sum()))
        rows.append(r)
    return pd.DataFrame(rows)


res = {}
for lab, (ma, mb) in {'G_j vs E_k (j != k)': ('support', 'empty'), 'E_j vs E_k (j != k)': ('empty', 'empty')}.items():
    rows = []
    for c in CORPORA:
        for j, k in itertools.product(range(3), range(3)):
            if j == k or (ma == mb and j >= k): continue
            pa, pb = path(c, ma, j), path(c, mb, k)
            if not pa or not pb: continue
            A, B = np.load(pa), np.load(pb)
            if A['score'].shape != B['score'].shape or not (A['seen'] == B['seen']).all(): continue
            d = pair_stats(A['score'], B['score'], A['seen']).mean().to_dict(); d.update(data=c, pair=f'{j}-{k}'); rows.append(d)
    df = pd.DataFrame(rows); per = df.groupby('data').mean(numeric_only=True)
    res[lab] = dict(corpora=int(len(per)), pairs=int(len(df)), **{m: float(per[m].mean()) for m in per.columns},
                    median_n_cand=float(per['n_cand'].median()), per_corpus=per.round(4).to_dict(orient='index'))
json.dump(res, open(f'{OUT}/r18_decision_agreement.json', 'w'), indent=1)
for lab, o in res.items():
    print(lab, {k: round(v, 4) for k, v in o.items() if isinstance(v, float)}, o['corpora'], o['pairs'])
