"""Release 7.0.0: statistics without training for the third UMUAI version.

1. Power of the comparisons read as negative or unresolved (reviewer comment 4.4): for every term, the smallest effect
   detectable with 80% power by a two-sided test at 5%, MDE = (z_0.975 + z_0.80) SE, with SE the standard error of
   the interval the article reports (mixed model, or OLS with one unit per corpus).
2. Seed ensembles on prediction (question 2): on the 14 three-seed units at 10%, the predictions of the three seeds
   of the control and of the support graph are averaged per test prediction, and the first-attempt AUC of the two
   ensembles is compared; also the single-seed differences and the control's seed spread.
3. Table 9 on one set of corpora (reviewer comment 4.7.1): every row on the nine corpora that have prediction files for
   the support graph in the embedding refinement.
4. Regret against AUC at one training size (reviewer comment 4.7.3): at 25% of the training learners (subsample 0,
   seeds 0, 100, 200), the first-attempt AUC gain of the graph among the decision models themselves, the regret change
   the Rasch slope predicts from it, and the regret change measured directly.
Output: results_r21/stats_more.json
"""
import os, sys, json, glob, numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r14_common import load_runs, CORPORA
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r21'; os.makedirs(OUT, exist_ok=True)
Z = norm.ppf(0.975) + norm.ppf(0.80)
U20 = [(c, s) for c in CORPORA for s in (0, 1)]


def power():
    S = json.load(open(f'{REPO}/results_r19/stats.json')); out = {}
    for t in S['fdr']['terms']:
        out[t['term']] = dict(est=t['est'], se=t['se'], mde80=Z * t['se'])
    for k in ('g100_emb - empty | U20', 'g100_emb - support_emb | U20', 'g100_emb - dp100_emb | U20',
              'st_g100 - st_empty | U20', 'st_g100 - st_support | U20', 'st_g100 - st_dp100 | U20'):
        m = S['transfer'][k]['first']['mixed']; out['transfer/' + k] = dict(est=m['est'], se=m['se'], mde80=Z * m['se'])
    for k, v in S['pdkt'].items():
        m = v['first']['mixed']; out['pdkt/' + k] = dict(est=m['est'], se=m['se'], mde80=Z * m['se'])
    m = S['primary_32']['first']['mixed']; out['primary/support - empty'] = dict(est=m['est'], se=m['se'], mde80=Z * m['se'])
    return out


def pred_of(v):
    """Prediction file of a run: its own, or an archived copy of the same run (same tag) whose first-attempt AUC
    equals the run's to 1e-9."""
    r, base = v; tag = os.path.basename(base)
    for f in [base + '_pred.npz'] + sorted(glob.glob(f'{REPO}/runs_*/{tag}_pred.npz')) + sorted(glob.glob(f'{REPO}/runs_*/*/{tag}_pred.npz')):
        if not os.path.exists(f): continue
        z = np.load(f); new = z['new'].astype(bool)
        if abs(roc_auc_score(z['y'][new], z['p'][new]) - r['test_auc_first']) < 1e-9: return z
    return None


def ensembles():
    SEEDS = {}
    for path in (f'{REPO}/runs_r6', f'{REPO}/runs_r9', f'{REPO}/runs_r19'):
        for k, v in load_runs([path]).items():
            if k[1] == 0.1 and k[3] in ('support', 'empty') and k[4] in (0, 1, 2): SEEDS.setdefault(k, v)
    rows = []
    for c, s in U20:
        ks = {m: [(c, 0.1, s, m, r, 0.002) for r in (0, 1, 2)] for m in ('support', 'empty')}
        if not all(k in SEEDS for v in ks.values() for k in v): continue
        P = {}; ok = True
        for m, kk in ks.items():
            arr = [pred_of(SEEDS[k]) for k in kk]
            if any(a is None for a in arr): ok = False; break
            if not all(len(a['y']) == len(arr[0]['y']) and (a['y'] == arr[0]['y']).all() for a in arr): ok = False; break
            P[m] = arr
        if not ok or len(P['support'][0]['y']) != len(P['empty'][0]['y']): print('skip', c, s); continue
        y = P['empty'][0]['y']; new = P['empty'][0]['new'].astype(bool)
        auc = lambda p: roc_auc_score(y[new], p[new])
        single = {m: [auc(a['p']) for a in P[m]] for m in P}
        ens = {m: auc(np.mean([a['p'] for a in P[m]], 0)) for m in P}
        rows.append(dict(data=c, sub=s, single_gain=float(np.mean(single['support']) - np.mean(single['empty'])),
                         ens_gain=float(ens['support'] - ens['empty']), ens_control_minus_single=float(ens['empty'] - np.mean(single['empty'])),
                         ens_graph_minus_single=float(ens['support'] - np.mean(single['support'])),
                         control_seed_sd=float(np.std(single['empty'], ddof=1))))
    d = pd.DataFrame(rows); c = d.groupby('data').mean(numeric_only=True)
    return dict(units=len(d), corpora=int(d.data.nunique()), single_gain=float(c.single_gain.mean()), ens_gain=float(c.ens_gain.mean()),
                ens_gain_pos=int((c.ens_gain > 0).sum()), ens_control_minus_single=float(c.ens_control_minus_single.mean()),
                ens_graph_minus_single=float(c.ens_graph_minus_single.mean()), per_unit=rows)


def table9():
    b = json.load(open(f'{REPO}/results_r19/baselines.json')); rows = []
    for x in b:
        if 'support_emb' not in x: continue
        r = dict(data=x['data'], sub=x['sub'])
        for m in ('kc_rate', 'pfa', 'empty', 'support_emb'):
            for s in ('pooled', 'per_learner', 'within_kc'): r[f'{m}:{s}'] = x[m][s]
            r[f'{m}:all'] = x[f'{m}_all']
        rows.append(r)
    d = pd.DataFrame(rows); c = d.groupby('data').mean(numeric_only=True).drop(columns='sub')
    out = dict(corpora=int(d.data.nunique()), units=len(d), means={k: float(v) for k, v in c.mean().items()})
    for s in ('pooled', 'per_learner', 'within_kc', 'all'):
        g = c[f'support_emb:{s}'] - c[f'empty:{s}']; out[f'gain:{s}'] = dict(mean=float(g.mean()), pos=int((g > 0).sum()))
    return out


def regret_25():
    R = pd.read_csv(f'{REPO}/results_r19/decisions_rows.csv'); D = json.load(open(f'{REPO}/results_r19/decisions.json'))
    s = R[(R.frac == 25) & (R['sub'] == 0) & R.model.isin(['empty', 'support'])]
    out = {}
    for sl, g in s.groupby('sim'):
        m = g.groupby(['data', 'model']).mean(numeric_only=True).unstack('model')
        dauc = (m['auc_first']['support'] - m['auc_first']['empty']); dreg = (m['regret_target']['support'] - m['regret_target']['empty'])
        sd = g[g.model == 'empty'].groupby('data').regret_target.std(ddof=1)
        slope = D['regret_vs_auc'][f'{sl}|target']['slope']
        out[sl] = dict(auc_gain_25=float(dauc.mean()), auc_gain_pos=int((dauc > 0).sum()), predicted_regret_change=float(slope * dauc.mean()),
                       measured_regret_change=float(dreg.mean()), measured_lower=int((dreg < 0).sum()), seed_sd=float(sd.mean()),
                       slope=slope, auc_for_one_sd=float(sd.mean() / abs(slope)))
    return out


if __name__ == '__main__':
    out = dict(power=power(), ensembles=ensembles(), table9=table9(), regret_25=regret_25())
    json.dump(out, open(f'{OUT}/stats_more.json', 'w'), indent=1)
    e = out['ensembles']; print({k: v for k, v in e.items() if k != 'per_unit'}); print(out['regret_25']); print(out['table9'])
    for k, v in out['power'].items():
        if any(x in k for x in ('support_emb - dp_emb', 'dp_emb - er_emb', 'transfer', 'pdkt', 'st_support - st_dp')): print(k, {a: round(b, 4) for a, b in v.items()})
