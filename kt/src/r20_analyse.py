"""Release 6.0.0: analyses of the runs of the second UMUAI version (no training).

Reads runs_r20 (and runs_r20_sim), the archived reference runs of earlier releases (r14_common.reference_runs,
runs_r13/runs_r14 for 100%) and results_r20/closed_loop_<world>.json, and writes results_r20/analysis.json.
Every comparison is a paired difference of test AUC (first attempts at a KC, all predictions; first attempts at an
item for the item-level runs) on the same subsample and seed, summarised as in r14_common.summ (mean over corpora
of the subsample means, corpora with a positive difference, Wilcoxon over corpora, mixed-model and bootstrap
intervals). Exploratory throughout; Benjamini-Hochberg q over the first-attempt terms of this release.
"""
import os, json, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import r14_common as C
REPO = C.REPO; OUT = f'{REPO}/results_r20'; os.makedirs(OUT, exist_ok=True)
C10 = C.CORPORA; ALL = C10 + ['dbe_kt22']


def runs():
    R = C.load_runs([f'{REPO}/runs_r20'])
    _, N13, REF = C.reference_runs()
    for k, v in REF.items(): R.setdefault(k, v)
    for k, v in C.load_runs([f'{REPO}/runs_r13', f'{REPO}/runs_r14']).items():
        if k[1] == 1.0 or k[3] == 'map_emb': R.setdefault(k, v)
    return R


def term(R, units, a, b, frac=0.1, cols=('test_auc_first', 'test_auc'), boot=True):
    o = {}
    for col in cols:
        rows = C.pairs(R, units, a, b, frac=frac, col=col)
        o[col] = C.summ(rows, do_boot=boot)
    return o


def bh(ps):
    k = [i for i, p in enumerate(ps) if p is not None]; m = len(k); q = [None] * len(ps)
    order = sorted(k, key=lambda i: ps[i]); run = 1.0
    for r in range(m - 1, -1, -1):
        i = order[r]; run = min(run, ps[i] * m / (r + 1)); q[i] = run
    return q


def main():
    R = runs(); A = {}
    U0 = [(c, 0) for c in ALL]
    U32 = C.U10 + [('dbe_kt22', s) for s in range(5)]
    # ---- NCA-KT
    nca = {}
    nca['nca-empty'] = term(R, U32, 'nca', 'empty')
    nca['nca-support_emb'] = term(R, U32, 'nca', 'support_emb')
    nca['nca-nca_dp (edges)'] = term(R, U32, 'nca', 'nca_dp')
    nca['support_emb-empty (same units)'] = term(R, U32, 'support_emb', 'empty')
    nca['support_emb-dp_emb (same units)'] = term(R, U32, 'support_emb', 'dp_emb')
    for ab in ('nca_static', 'nca_nonull', 'nca_nogate', 'nca_nomargin'):
        nca[f'nca-{ab}'] = term(R, U0, 'nca', ab)
    nca['nca_nonull-empty'] = term(R, U0, 'nca_nonull', 'empty')
    nca['empty_fa-empty'] = term(R, U32, 'empty_fa', 'empty')
    nca['nca-empty_fa'] = term(R, U32, 'nca', 'empty_fa')
    nca['nca_dp-empty_fa'] = term(R, U32, 'nca_dp', 'empty_fa')
    nca['nca_nomargin-empty'] = term(R, U0, 'nca_nomargin', 'empty')
    nca['support_emb-empty (unit 0)'] = term(R, U0, 'support_emb', 'empty')
    nca['nca_nogate-empty'] = term(R, U0, 'nca_nogate', 'empty')
    U100 = [(c, 0) for c in ALL]
    nca['100: nca-empty'] = term(R, U100, 'nca', 'empty', frac=1.0)
    nca['100: nca-nca_dp'] = term(R, U100, 'nca', 'nca_dp', frac=1.0)
    nca['100: nca-support'] = term(R, U100, 'nca', 'support', frac=1.0)
    nca['100: support-empty'] = term(R, U100, 'support', 'empty', frac=1.0)
    nca['100: empty_fa-empty'] = term(R, U100, 'empty_fa', 'empty', frac=1.0)
    nca['100: nca-empty_fa'] = term(R, U100, 'nca', 'empty_fa', frac=1.0)
    al = []
    for (c, fr, s, cond, rep, lr), (r, _) in R.items():
        if cond.startswith('nca') and 'alpha_mean' in r:
            al.append(dict(data=c, frac=fr, sub=s, cond=cond, alpha=r['alpha_mean'], first=r['alpha_first'], repeat=r['alpha_repeat']))
    if al:
        d = pd.DataFrame(al); nca['alpha'] = d.groupby(['cond', 'frac'])[['alpha', 'first', 'repeat']].mean().reset_index().to_dict('records')
        nca['alpha_by_corpus_10'] = d[(d.cond == 'nca') & (d.frac == 0.1)].groupby('data')[['alpha', 'first', 'repeat']].mean().to_dict('index')
    A['nca'] = nca
    # ---- DBE-KT22: new corpus, authored and generated graphs
    UD = [('dbe_kt22', s) for s in range(5)]
    dbe = {}
    for a, b in [('support', 'empty'), ('support_emb', 'empty'), ('uniform_emb', 'empty'), ('er_emb', 'uniform_emb'), ('dp_emb', 'er_emb'),
                 ('support_emb', 'dp_emb'), ('st_support', 'empty'), ('st_support', 'st_dp'), ('st_er', 'empty'),
                 ('expert_emb', 'empty'), ('expert_emb', 'expert_dp_emb'), ('expert_emb', 'support_emb'),
                 ('llm_emb', 'empty'), ('llm_emb', 'llm_dp_emb'), ('llm_emb', 'support_emb'), ('llm_emb', 'expert_emb'),
                 ('nca', 'empty'), ('nca', 'nca_dp'), ('nca_expert', 'empty'), ('nca_llm', 'empty'), ('nca_expert', 'nca')]:
        rows = {col: C.pairs(R, UD, a, b, col=col) for col in ('test_auc_first', 'test_auc')}
        dbe[f'{a}-{b}'] = {col: (dict(units=len(v), mean=float(np.mean([x['d'] for x in v])), pos=int(sum(x['d'] > 0 for x in v)),
                                      t=_tint([x['d'] for x in v])) if v else None) for col, v in rows.items()}
    dbe['100: support-empty'] = term(R, [('dbe_kt22', 0)], 'support', 'empty', frac=1.0, boot=False)
    A['dbe'] = dbe
    # junyi map in NCA
    A['junyi_map'] = {'nca_map-empty': term(R, [('junyi', 0)], 'nca_map', 'empty', boot=False),
                      'map_emb-map_dp_emb': term(R, [('junyi', 0)], 'map_emb', 'map_dp_emb', boot=False),
                      'nca-empty': term(R, [('junyi', 0)], 'nca', 'empty', boot=False)}
    # ---- item level
    QD = ['statics11', 'assist17', 'dbe_kt22', 'eedi', 'assist09', 'ednet', 'assist12']
    UQ = [(c, 0) for c in QD] + [(c, 1) for c in ('statics11', 'assist17', 'dbe_kt22')]
    q = {}
    cols = ('test_auc_first', 'test_auc_first_item', 'test_auc')
    for a, b in [('q_kc', 'empty'), ('q_id', 'q_kc'), ('q_qm', 'q_id'), ('q_qm', 'q_kc'), ('q_qm', 'q_rqm'), ('q_rqm', 'q_id'),
                 ('q_qm_g', 'q_qm'), ('q_qm_g', 'q_qm_dp'), ('q_qm_dp', 'q_qm'), ('q_diff', 'q_kc'), ('q_qm', 'q_diff')]:
        q[f'{a}-{b}'] = term(R, UQ, a, b, cols=cols if a != 'q_kc' or b != 'empty' else ('test_auc_first', 'test_auc'))
    lev = []
    for (c, fr, s, cond, rep, lr), (r, _) in R.items():
        if cond.startswith('q_') and fr == 0.1:
            lev.append(dict(data=c, sub=s, cond=cond, first=r['test_auc_first'], first_item=r.get('test_auc_first_item'), all=r['test_auc'],
                            unseen=r.get('test_items_unseen_share')))
    if lev: q['levels'] = pd.DataFrame(lev).groupby(['data', 'cond'])[['first', 'first_item', 'all', 'unseen']].mean().reset_index().to_dict('records')
    A['item'] = q
    # ---- BH over first-attempt terms of this release (corpus-level Wilcoxon p where available)
    keys, ps = [], []
    for blk in ('nca', 'item'):
        for k, v in A[blk].items():
            if isinstance(v, dict) and 'test_auc_first' in v and v['test_auc_first'] and 'p_wilcoxon' in v['test_auc_first']:
                keys.append(f'{blk}:{k}'); ps.append(v['test_auc_first']['p_wilcoxon'])
    A['bh'] = dict(zip(keys, bh(ps)))
    # ---- simulated worlds
    A['sim'] = sim()
    json.dump(A, open(f'{OUT}/analysis.json', 'w'), indent=1, default=float)
    print('written', f'{OUT}/analysis.json')


def _tint(v):
    v = np.asarray(v, float)
    if len(v) < 2: return None
    from scipy.stats import t
    se = v.std(ddof=1) / np.sqrt(len(v)); h = t.ppf(.975, len(v) - 1) * se
    return [float(v.mean() - h), float(v.mean() + h)]


def sim():
    S = {}
    Rs = {}
    for f in glob.glob(f'{REPO}/runs_r20_sim/*.json'):
        r = json.load(open(f)); Rs[(r['data'], round(r['frac'], 2), r['cond'], r['rep'])] = r
    for w in ('sim_dbe', 'sim_a09'):
        o = {'auc': {}, 'terms': {}}
        for fr in (0.1, 1.0):
            for cond in ('empty', 'support_emb', 'er_emb', 'dp_emb', 'true_emb', 'true_dp_emb', 'nca', 'nca_dp', 'nca_true', 'st_true', 'st_true_dp'):
                v = [Rs[(w, fr, cond, rep)] for rep in (0, 1) if (w, fr, cond, rep) in Rs]
                if v: o['auc'][f'{fr}:{cond}'] = dict(first=float(np.mean([x['test_auc_first'] for x in v])), all=float(np.mean([x['test_auc'] for x in v])), n=len(v))
            for a, b in [('true_emb', 'true_dp_emb'), ('true_emb', 'empty'), ('support_emb', 'empty'), ('support_emb', 'dp_emb'), ('er_emb', 'empty'),
                         ('true_emb', 'support_emb'), ('nca_true', 'empty'), ('nca', 'nca_dp'), ('nca_true', 'true_emb'), ('nca', 'empty'),
                         ('st_true', 'st_true_dp'), ('st_true', 'empty')]:
                d = [(Rs[(w, fr, a, rep)]['test_auc_first'] - Rs[(w, fr, b, rep)]['test_auc_first'], Rs[(w, fr, a, rep)]['test_auc'] - Rs[(w, fr, b, rep)]['test_auc'])
                     for rep in (0, 1) if (w, fr, a, rep) in Rs and (w, fr, b, rep) in Rs]
                if d: o['terms'][f'{fr}:{a}-{b}'] = dict(first=float(np.mean([x[0] for x in d])), all=float(np.mean([x[1] for x in d])),
                                                          reps=[x[0] for x in d])
        p = f'{OUT}/closed_loop_{w}.json'
        if os.path.exists(p):
            L = json.load(open(p)); cl = {'reference': {k: {m: v[m] for m in ('gain', 'ttm5', 'reached5', 'frustration', 'success', 'regret')} for k, v in L['reference'].items()}}
            by = {}
            for tag, rec in L['models'].items():
                key = (rec['frac'], rec['cond'])
                by.setdefault(key, []).append(rec)
            rows = {}
            for (fr, cond), recs in by.items():
                for pol in ('target', 'stable', 'frontier'):
                    rows[f'{fr}:{cond}:{pol}'] = {m: float(np.mean([r[pol][m] for r in recs])) for m in ('gain', 'ttm5', 'reached5', 'frustration', 'success', 'regret')}
                    rows[f'{fr}:{cond}:{pol}']['n_seeds'] = len(recs)
                    if len(recs) == 2:
                        a, b = (np.array(r[pol + '_chosen']) for r in recs)
                        rows[f'{fr}:{cond}:{pol}']['first_choice_agreement'] = float((a[:, 0] == b[:, 0]).mean())
                        rows[f'{fr}:{cond}:{pol}']['jaccard10'] = float(np.mean([len(set(x) & set(y)) / max(1, len(set(x) | set(y))) for x, y in zip(a, b)]))
                rows[f'{fr}:{cond}:max_score_check'] = max(r['score_check_max_abs_diff'] for r in recs)
            # paired differences per learner between conditions (seed-averaged), target rule
            def per(fr, cond, pol, m):
                recs = by.get((fr, cond), [])
                return np.mean([np.array(r[pol]['per_learner'][m], float) for r in recs], 0) if recs else None
            pairs = {}
            for fr in (0.1, 1.0):
                for a, b in [('true_emb', 'true_dp_emb'), ('true_emb', 'empty'), ('support_emb', 'empty'), ('nca_true', 'empty'), ('nca_true', 'true_emb'), ('nca', 'empty')]:
                    for pol in ('target', 'stable', 'frontier'):
                        for m in ('gain', 'regret', 'frus'):
                            x, y = per(fr, a, pol, m), per(fr, b, pol, m)
                            if x is None or y is None: continue
                            d = x - y; se = d.std(ddof=1) / np.sqrt(len(d))
                            pairs[f'{fr}:{a}-{b}:{pol}:{m}'] = dict(mean=float(d.mean()), lo=float(d.mean() - 1.96 * se), hi=float(d.mean() + 1.96 * se))
            cl['policies'] = rows; cl['pairs'] = pairs
            # seed-to-seed spread of the control as a yardstick
            o['closed_loop'] = cl
        S[w] = o
    return S


if __name__ == '__main__': main()
