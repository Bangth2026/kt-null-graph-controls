"""Release 5.0.0: analyses without training runs, added in the revision for UMUAI.

1. Simple first-attempt references (reviewer comment M6). For every unit (corpus, subsample) of the 32-unit design at 10%
   of the training learners, on exactly the test predictions of the archived probe runs:
     kc_rate  the proportion correct of the target KC among the training learners of the subsample (prior 0.5 weight 5)
     pfa      a logistic model fitted on all next-attempt predictions of the training learners of the subsample, with a
              fixed effect per KC and the learner's running performance: log(1 + successes) and log(1 + failures) on
              other KCs, the smoothed proportion correct on other KCs, and log(1 + successes), log(1 + failures) on the
              target KC (zero at a first attempt) (a PFA/IRT-style learner model; L2, C = 1)
   and the empty-graph probe and the support graph in the embedding refinement (runs_r9). AUC on first attempts is
   computed three ways: pooled over all first-attempt predictions (as in the manuscript), averaged over test learners
   (learners with both outcomes among their first attempts), and averaged over KCs, weighted by the number of first
   attempts (KCs with both outcomes): the last is the within-KC AUC, which no between-KC difficulty can raise.
2. Graph quality by training size (M7, minor comments 4 and 5): for subsamples at 5%, 10% and 25% (the subsamples of
   the study), the number of edges of the support graph, its overlap with the graph built from all training learners
   (Jaccard of undirected pairs; recall: share of the 100% graph's directed edges present; precision: share of the
   subsample graph's edges present in the 100% graph), and the number of reciprocal pairs in DP(G).
3. Uniform mixing (minor comment 2): the input of uniform mixing to KC k is (S - E_k)/(K - 1) with S the sum of the
   rows; two KCs differ by (E_j - E_k)/(K - 1). The relative size of that difference is reported per corpus for the
   trained tables of the empty-graph control is not stored, so it is reported for the initial table (N(0, 1) rows).
Writes results_r19/baselines.json and results_r19/graph_quality.json.
"""
import os, sys, json, numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import load, transition_support, rewire
warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r19'; os.makedirs(OUT, exist_ok=True)
SMALL = ['statics11', 'algebra05', 'bridge06', 'assist17']
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
U10 = [(c, s) for c in SMALL for s in range(5)] + [(c, s) for c in CORPORA if c not in SMALL for s in range(2)]


def features(df, split):
    """One row per next-attempt prediction (positions 1.. of each learner's first 200 interactions), in the order of
    train.seqs + r13_controls.evaluate (learners in order of first appearance, positions in order)."""
    rows = []
    d = df[df.split == split]
    for ui, (u, g) in enumerate(d.groupby('user', sort=False)):
        q = g.kc.values[:200]; r = g.correct.values[:200]
        if len(q) < 2: continue
        ns = nf = 0; ks = {}; kf = {}; seen = set()
        for t in range(len(q)):
            k = int(q[t])
            if t >= 1:
                s_k = ks.get(k, 0); f_k = kf.get(k, 0); so = ns - s_k; fo = nf - f_k
                rows.append((ui, t, k, int(r[t]), k not in seen, np.log1p(so), np.log1p(fo), (so + 1) / (so + fo + 2),
                             np.log1p(s_k), np.log1p(f_k)))
            seen.add(k)
            if r[t]: ns += 1; ks[k] = ks.get(k, 0) + 1
            else: nf += 1; kf[k] = kf.get(k, 0) + 1
    return pd.DataFrame(rows, columns=['uid', 'pos', 'kc', 'y', 'first', 'ls', 'lf', 'pc', 'lsk', 'lfk'])


def aucs(y, p, uid, kc):
    out = {'pooled': float(roc_auc_score(y, p)) if len(set(y)) > 1 else None}
    for name, grp in (('per_learner', uid), ('within_kc', kc)):
        v = []; w = []
        for gk in np.unique(grp):
            mk = grp == gk
            if mk.sum() >= 2 and 0 < y[mk].sum() < mk.sum():
                v.append(roc_auc_score(y[mk], p[mk])); w.append(mk.sum())
        out[name] = float(np.average(v, weights=w if name == 'within_kc' else None)) if v else None
    return out


def baselines():
    res = []
    for c, s in U10:
        df = load(c, 0.1, s); K = int(df.kc.max()) + 1
        tr = features(df, 'train'); te = features(df, 'test')
        rate = tr.groupby('kc').y.agg(['sum', 'count']); kr = np.full(K, np.nan)
        kr[rate.index.values] = (rate['sum'] + 2.5) / (rate['count'] + 5); kr[np.isnan(kr)] = (tr.y.sum() + 2.5) / (len(tr) + 5)
        Xcols = ['ls', 'lf', 'pc', 'lsk', 'lfk']
        def X(d):
            oh = np.zeros((len(d), K), np.float32); oh[np.arange(len(d)), d.kc.values] = 1
            return np.hstack([oh, d[Xcols].values.astype(np.float32)])
        lr = LogisticRegression(C=1.0, max_iter=500).fit(X(tr), tr.y.values)
        p_pfa = lr.predict_proba(X(te))[:, 1]; p_kc = kr[te.kc.values]
        row = dict(data=c, sub=s)
        fm = te['first'].values; y = te.y.values[fm]; uid = te.uid.values[fm]; kk = te.kc.values[fm]
        for name, p in (('kc_rate', p_kc), ('pfa', p_pfa)):
            row[name] = aucs(y, p[fm], uid, kk); row[name + '_all'] = float(roc_auc_score(te.y.values, p))
        for name, cond in (('empty', 'empty'), ('support_emb', 'support_emb')):
            for d in ('runs_r9', 'runs_r13', 'runs_r16', 'runs_r10/base'):
                f = f'{REPO}/{d}/{c}__f10__s{s}__{cond}__steps_pred.npz'
                if not os.path.exists(f): continue
                Z = np.load(f)
                if len(Z['y']) == len(te) and (Z['y'] == te.y.values).all() and (Z['kc'] == te.kc.values).all() and (Z['new'] == fm).all():
                    row[name] = aucs(y, Z['p'][fm], uid, kk); row[name + '_all'] = float(roc_auc_score(te.y.values, Z['p'])); row[name + '_src'] = d; break
        res.append(row); print(c, s, {k: round(row[k]['pooled'], 4) for k in ('kc_rate', 'pfa', 'empty', 'support_emb') if k in row}, flush=True)
    json.dump(res, open(f'{OUT}/baselines.json', 'w'), indent=1)
    return res


def graph_quality():
    out = []
    for c in CORPORA:
        full = load(c, 1.0, 0); K = int(full.kc.max()) + 1
        A100 = (transition_support(full[full.split == 'train'], K) > 0).astype(np.float32); B100 = (A100 + A100.T) > 0
        for frac, subs in ((0.05, 5), (0.1, 5), (0.25, 3)):
            for s in range(subs):
                df = load(c, frac, s); Kd = int(df.kc.max()) + 1
                A = (transition_support(df[df.split == 'train'], Kd) > 0).astype(np.float32)[:K, :K]
                if A.shape[0] < K: A = np.pad(A, ((0, K - A.shape[0]), (0, K - A.shape[0])))
                B = (A + A.T) > 0
                dp, _ = rewire(A, seed=1000 + s)
                out.append(dict(data=c, frac=frac, sub=s, n_train=int(df[df.split == 'train'].user.nunique()),
                                edges=int(A.sum()), edges_100=int(A100.sum()),
                                jaccard=float((B & B100).sum() / max(1, (B | B100).sum())),
                                recall=float((A * A100).sum() / max(1, A100.sum())),
                                precision=float((A * A100).sum() / max(1, A.sum())),
                                reciprocal_G=int(((A * A.T) > 0).sum() // 2), reciprocal_DP=int(((dp * dp.T) > 0).sum() // 2)))
                print(out[-1], flush=True)
    json.dump(out, open(f'{OUT}/graph_quality.json', 'w'), indent=1)


if __name__ == '__main__':
    which = sys.argv[1:] or ['baselines', 'graph_quality']
    if 'graph_quality' in which: graph_quality()
    if 'baselines' in which: baselines()
