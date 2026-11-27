"""Release 7.0.0: graph statistics without training (reviewer comments 4.1, 4.7.6 and minor 7).

1. Edges of the support graph G100 of all training learners, recomputed on every corpus (Table 1).
2. Sensitivity of the support graph to the support threshold s of Eq. (2) (s = 5, 10, 20) at 10% of the training
   learners (five subsamples): edges, and recall and precision against G100 built with s = 10.
   On the support: an ordered pair (a, b) is an edge iff T(a->b) + T(b->a) >= s and T(a->b) > T(b->a); the minimum
   with pref(a, b) in Eq. (2) changes the graded value but never whether a pair is an edge (pref > 0 whenever
   T(a->b) > 0), which is checked here on every graph.
3. Recovery of the true graph G* of the simulated worlds by the support graph at 5%, 10%, 25% and 100% of the
   training learners (subsamples 0-2): recall, precision, and the share of its edges that are reversed edges of G*.
Output: results_r21/graphs.json
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import load, transition_support
import r14_controls as R14, r20_controls as R20
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r21'; os.makedirs(OUT, exist_ok=True)
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
HELD = ['xes3g5m', 'eedi12']


def support(tr, K, s=10):
    mu = transition_support(tr, K, s); A = (mu > 0)
    T = np.zeros((K, K))
    for _, g in tr.groupby('user', sort=False):
        q = g.kc.values; r = q[np.r_[True, q[1:] != q[:-1]]]
        if len(r) > 1: np.add.at(T, (r[:-1], r[1:]), 1)
    B = ((T + T.T) >= s) & (T > T.T); np.fill_diagonal(B, False)
    assert (A == B).all(), 'support differs from the dominant-direction rule'
    return A


def main():
    out = {'g100': {}, 's_sensitivity': {}, 'sim_recovery': {}}
    for c in CORPORA + HELD:
        if not os.path.exists(f'{REPO}/data/{c}.parquet'): continue
        df = load(c, 1.0, 0); K = int(df.kc.max()) + 1; G = support(df[df.split == 'train'], K)
        out['g100'][c] = dict(edges=int(G.sum()), K=K, train_learners=int(df[df.split == 'train'].user.nunique()),
                              train_learners_10=int(round(0.1 * df[df.split == 'train'].user.nunique())))
        if c in HELD: continue
        rows = []
        for sub in range(5):
            d = load(c, 0.1, sub); tr = d[d.split == 'train']
            for s in (5, 10, 20):
                A = support(tr, K, s); tp = (A & G).sum()
                rows.append(dict(sub=sub, s=s, edges=int(A.sum()), recall=float(tp / max(G.sum(), 1)), precision=float(tp / max(A.sum(), 1))))
        out['s_sensitivity'][c] = rows
        print(c, out['g100'][c], flush=True)
    for w in ('sim_a09', 'sim_dbe'):
        res = []
        for frac in (0.05, 0.1, 0.25, 1.0):
            for sub in ((0, 1, 2) if frac < 1 else (0,)):
                d = load(w, frac, sub); K = int(d.kc.max()) + 1; names = R14.kc_names(d, K)
                Gs = R20.named_graph(f'{REPO}/data/{w}_true.json', names) > 0; A = support(d[d.split == 'train'], K)
                tp = (A & Gs).sum(); rev = (A & Gs.T).sum()
                res.append(dict(frac=frac, sub=sub, edges=int(A.sum()), true_edges=int(Gs.sum()), recall=float(tp / Gs.sum()),
                                precision=float(tp / max(A.sum(), 1)), reversed=float(rev / max(A.sum(), 1))))
        out['sim_recovery'][w] = res; print(w, res[-1], flush=True)
    json.dump(out, open(f'{OUT}/graphs.json', 'w'), indent=1)


if __name__ == '__main__': main()
