"""Release 5.0.0: statistical re-analyses for the revision for UMUAI (reviewer comments M1, M2, M3, M10). Reads run files.

 1. Primary comparison (support graph at both entry points against the empty-graph control, 10%, update budget):
    (a) the 32-unit estimate of the manuscript, (b) the balanced 50-unit design (five subsamples on every corpus), each
    with four intervals: the mixed model of the manuscript (random intercept per corpus, REML, Wald 95%), a fixed-effect
    estimate (mean of corpus means, t interval over corpora), a corpus-clustered bootstrap (corpora resampled, then
    subsamples within corpus; 10,000 draws) and the Wilcoxon test over corpus means.
 2. Three training seeds per subsample (seeds 0, 100, 200) on subsamples 0 and 1 of every corpus (20 units): the gain
    averaged over seeds within each unit, so that its interval carries seed variance, and the seed SD of the control.
 3. A smallest effect of interest that does not use the graph effect: delta_seed, the root mean square over units of the
    within-unit SD of the control's first-attempt AUC across three seeds. Every ladder term of the manuscript is classed
    against it as well as against 0.001/0.002/0.003.
 4. False-discovery-rate sensitivity: every exploratory first-attempt term of Tables 3 and 4 (mixed-model estimate and SE
    from results_r14 and results_r16), Benjamini-Hochberg over the family.
 5. Transfer of structure (runs_r19): the graph built from all training learners, given to the tracer at 10%, against the
    empty-graph control, against the subsample's own graph, and against its degree-preserving rewiring, in the embedding
    refinement and in the state channel.
 6. PDKT (runs_r19): the null-graph ladder with the graph in the training loss.
 7. 500-interaction cap on two larger corpora (runs_r19, __len500).
Writes results_r19/stats.json.
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r14_common import load_runs, load_len, pairs, mixed, reference_runs, ladder_runs, SMALL, CORPORA, U10
warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r19'; os.makedirs(OUT, exist_ok=True)
U20 = [(c, s) for c in CORPORA for s in (0, 1)]; U50 = [(c, s) for c in CORPORA for s in range(5)]
DELTAS = (0.001, 0.002, 0.003)


def klass(lo90, hi90, d):
    if lo90 > d: return 'B+'
    if hi90 < -d: return 'B-'
    if lo90 > 0 or hi90 < 0: return 'S' if (lo90 > -d and hi90 < d) else 'N'
    if lo90 > -d and hi90 < d: return 'E'
    return 'U'


def describe(rows, B=10000, seed=0, deltas=DELTAS):
    if not rows: return None
    df = pd.DataFrame(rows).dropna(subset=['d']); per = df.groupby('data').d.mean(); n = len(per)
    o = dict(units=int(len(df)), corpora=int(n), mean=float(per.mean()), median=float(per.median()), pos=int((per > 0).sum()),
             per_corpus={k: float(v) for k, v in per.items()})
    if n >= 4: o['p_wilcoxon'] = float(wilcoxon(per.values).pvalue)
    m = mixed(df)
    if m: o['mixed'] = {k: m[k] for k in ('est', 'se', 'lo', 'hi', 'lo90', 'hi90', 'kind')}
    se = per.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    if n > 1:
        q = tdist.ppf(.975, n - 1); q9 = tdist.ppf(.95, n - 1)
        o['fixed_t'] = dict(lo=float(per.mean() - q * se), hi=float(per.mean() + q * se), lo90=float(per.mean() - q9 * se), hi90=float(per.mean() + q9 * se))
    rng = np.random.default_rng(seed); g = {c: v.d.values for c, v in df.groupby('data')}; cs = list(g); bs = []
    for _ in range(B):
        pick = rng.choice(n, n); bs.append(np.mean([rng.choice(g[cs[i]], len(g[cs[i]])).mean() for i in pick]))
    o['cluster_boot'] = dict(lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), lo90=float(np.quantile(bs, .05)), hi90=float(np.quantile(bs, .95)))
    if 'mixed' in o:
        o['class'] = {str(d): klass(o['mixed']['lo90'], o['mixed']['hi90'], d) for d in deltas}
    return o


def both(R, units, a, b, **kw):
    return {'first': describe(pairs(R, units, a, b, col='test_auc_first', **kw)), 'all': describe(pairs(R, units, a, b, col='test_auc', **kw))}


def cross_pairs(Ra, Rb, units, a, b, col='test_auc_first'):
    rows = []
    for c, s in units:
        ka = (c, 0.1, s, a, 0, 0.002); kb = (c, 0.1, s, b, 0, 0.002)
        if ka in Ra and kb in Rb: rows.append(dict(data=c, sub=s, d=Ra[ka][0][col] - Rb[kb][0][col]))
    return rows


def main():
    res = {}
    R6 = load_runs([f'{REPO}/runs_r6']); R9 = load_runs([f'{REPO}/runs_r9']); R19 = load_runs([f'{REPO}/runs_r19'])
    # 1. primary
    res['primary_32'] = both(R6, U10, 'support', 'empty')
    U5 = [(c, s_) for c in CORPORA for s_ in range(5)]
    res['primary_5pct'] = both(R6, U5, 'support', 'empty', frac=0.05)
    R6b = dict(load_runs([f'{REPO}/runs_r16'])); R6b.update(R6)
    res['primary_50'] = both(R6b, U50, 'support', 'empty')
    # 2. three seeds
    SEEDS = {}
    for path in (f'{REPO}/runs_r6', f'{REPO}/runs_r9', f'{REPO}/runs_r19'):
        for k, v in load_runs([path]).items():
            if k[1] == 0.1 and k[3] in ('support', 'empty') and k[4] in (0, 1, 2): SEEDS.setdefault(k, v)
    rows = {'first': [], 'all': []}; ctrl_sd = []; units3 = []
    for c, s in U20:
        ks = [((c, 0.1, s, 'support', r, 0.002), (c, 0.1, s, 'empty', r, 0.002)) for r in (0, 1, 2)]
        if not all(a in SEEDS and b in SEEDS for a, b in ks): continue
        units3.append((c, s))
        for col, key in (('test_auc_first', 'first'), ('test_auc', 'all')):
            rows[key].append(dict(data=c, sub=s, d=float(np.mean([SEEDS[a][0][col] - SEEDS[b][0][col] for a, b in ks]))))
        ctrl_sd.append(np.std([SEEDS[b][0]['test_auc_first'] for _, b in ks], ddof=1))
    d_seed = float(np.sqrt(np.mean(np.square(ctrl_sd)))) if ctrl_sd else None
    res['three_seeds'] = dict(units=len(units3), first=describe(rows['first'], deltas=DELTAS + ((d_seed,) if d_seed else ())),
                              all=describe(rows['all']), control_seed_sd_rms=d_seed,
                              control_seed_sd_median=float(np.median(ctrl_sd)) if ctrl_sd else None)
    # single-seed estimate on the same units, for comparison
    res['three_seeds']['seed0_same_units'] = describe(pairs(R6, units3, 'support', 'empty'))
    # 3./4. FDR over exploratory terms
    terms = []
    def walk(x, path):
        if isinstance(x, dict):
            if 'first' in x and isinstance(x['first'], dict) and isinstance(x['first'].get('mixed'), float):
                f = x['first']; terms.append(dict(term=path, est=f['mixed'], se=f['se'], lo90=f['ci90'][0], hi90=f['ci90'][1], mean=f.get('mean')))
                return
            if 'first' in x and isinstance(x['first'], dict) and isinstance(x['first'].get('mixed'), dict):
                m = x['first']['mixed']; terms.append(dict(term=path, est=m['est'], se=m['se'], lo90=m['lo90'], hi90=m['hi90'], mean=x['first'].get('mean')))
                return
            for k, v in x.items(): walk(v, f'{path}/{k}')
    r14 = json.load(open(f'{REPO}/results_r14/r14_results.json')); r16 = json.load(open(f'{REPO}/results_r16/r16_results.json'))
    for key in ('equivalence', 'curriculum', 'gkt_cpu_ladder', 'gkt_clique'): walk(r14[key], f'r14/{key}')
    walk(r16['other_constructions'], 'r16/other_constructions')
    terms = [t for t in terms if '/emb G - ' not in t['term'] and '/state G - ' not in t['term']]
    T = pd.DataFrame(terms).drop_duplicates('term')
    T['z'] = T.est / T.se; T['p'] = 2 * (1 - norm.cdf(T.z.abs()))
    o = T.p.rank(method='first').astype(int).values; m = len(T); ps = T.p.values; order = np.argsort(ps); q = np.empty(m); run = 1.0
    for rank_i, i in reversed(list(enumerate(order, start=1))):
        run = min(run, ps[i] * m / rank_i); q[i] = run
    T['q_bh'] = q
    if d_seed:
        T['class_dseed'] = [klass(a, b, d_seed) for a, b in zip(T.lo90, T.hi90)]
    res['fdr'] = dict(n_terms=int(m), n_p05=int((T.p < 0.05).sum()), n_q05=int((T.q_bh < 0.05).sum()), n_q10=int((T.q_bh < 0.10).sum()),
                      expected_false_at_p05=float(0.05 * m), terms=T.round(6).to_dict(orient='records'))
    # 5. transfer
    REF = {}
    for path in (f'{REPO}/runs_r9', f'{REPO}/runs_r13', f'{REPO}/runs_r16', f'{REPO}/runs_r10/base', f'{REPO}/runs_r11/base'):
        for k, v in load_runs([path]).items():
            if k[1] == 0.1 and k[4] == 0: REF.setdefault(k, v)
    # EdNet: the comparisons are made within the rebuilt data (runs_r13 / runs_r16 / runs_r19), as in r14_common
    for k, v in load_runs([f'{REPO}/runs_r13']).items():
        if k[0] == 'ednet' and k[1] == 0.1 and k[4] == 0: REF[k] = v
    ALL = dict(REF); ALL.update(R19)
    tr = {}
    for a, b in (('g100_emb', 'empty'), ('g100_emb', 'support_emb'), ('g100_emb', 'dp100_emb'), ('dp100_emb', 'empty'),
                 ('st_g100', 'st_empty'), ('st_g100', 'st_support'), ('st_g100', 'st_dp100'), ('st_dp100', 'st_empty'),
                 ('support_emb', 'empty'), ('st_support', 'st_empty')):
        for units, lab in ((U20, 'U20'), (U10, 'U32')):
            tr[f'{a} - {b} | {lab}'] = both(ALL, units, a, b)
    ov = [dict(data=v[0]['data'], sub=v[0]['sub'], jaccard=v[0].get('jaccard_G_G100_undirected'), recall=v[0].get('recall_G100_in_G_directed'),
               precision=v[0].get('precision_G_in_G100_directed'), e100=v[0].get('edges_g100'), e=v[0]['edges'])
          for k, v in R19.items() if k[3] == 'g100_emb']
    tr['overlap'] = pd.DataFrame(ov).groupby('data').mean(numeric_only=True).round(4).to_dict(orient='index') if ov else {}
    res['transfer'] = tr
    # 6. PDKT ladder
    pd_ = {}
    for a, b in (('pdkt_er', 'pdkt_empty'), ('pdkt_dp', 'pdkt_er'), ('pdkt_support', 'pdkt_dp'), ('pdkt_support', 'pdkt_empty'),
                 ('pdkt_support', 'pdkt_er'), ('pdkt_empty', 'empty')):
        pd_[f'{a} - {b}'] = both(ALL, U20, a, b)
    res['pdkt'] = pd_
    # 7. cap 500
    L5 = load_len([f'{REPO}/runs_r19', f'{REPO}/runs_r14'], 500)
    cap = {}
    a, b = 'support_emb', 'empty'
    for lab, corp in (('larger corpora (this release)', ('assist12', 'eedi')), ('smallest corpora (release 3.11.0)', ('algebra05', 'statics11'))):
        rows5 = pairs(L5, [(c, s) for c in corp for s in range(5)], a, b)
        rows2 = pairs(ALL, [(r['data'], r['sub']) for r in rows5], a, b)
        cap[f'{a} - {b}, {lab}'] = dict(len500=describe(rows5, B=2000), len200=describe(rows2, B=2000), units=[(r['data'], r['sub']) for r in rows5],
                                        per_unit=[dict(data=r5['data'], sub=r5['sub'], len500=r5['d'], len200=r2['d']) for r5, r2 in zip(rows5, rows2)])
    res['cap500'] = cap
    # 8. XES3G5M without multi-KC questions (H1 of the replication, subsamples 0 and 1)
    R14r = load_runs([f'{REPO}/runs_r14']); xs = []
    for s_ in (0, 1):
        k = lambda c, cond: (c, 0.1, s_, cond, 0, 0.002)
        if all(x in R19 for x in (k('xes3g5m1', 'support'), k('xes3g5m1', 'empty'))) and all(x in R14r for x in (k('xes3g5m', 'support'), k('xes3g5m', 'empty'))):
            xs.append(dict(sub=s_, single_first=R19[k('xes3g5m1', 'support')][0]['test_auc_first'] - R19[k('xes3g5m1', 'empty')][0]['test_auc_first'],
                           single_all=R19[k('xes3g5m1', 'support')][0]['test_auc'] - R19[k('xes3g5m1', 'empty')][0]['test_auc'],
                           first_kc_first=R14r[k('xes3g5m', 'support')][0]['test_auc_first'] - R14r[k('xes3g5m', 'empty')][0]['test_auc_first'],
                           first_kc_all=R14r[k('xes3g5m', 'support')][0]['test_auc'] - R14r[k('xes3g5m', 'empty')][0]['test_auc']))
    if xs:
        res['xes_single_kc'] = xs
        res['xes_single_kc_text'] = ('H1 (support graph at both entry points against the empty graph, 10%, update budget) on XES3G5M with every answer to a multi-KC question dropped (data/xes3g5m1.parquet, prep_r19.py; same learners), against the first-KC rule of the replication on the same subsamples. First attempts: '
                                     + '; '.join(f"subsample {x['sub']}: {x['single_first']:+.4f} without multi-KC questions, {x['first_kc_first']:+.4f} with the first-KC rule" for x in xs)
                                     + '. All predictions: ' + '; '.join(f"subsample {x['sub']}: {x['single_all']:+.4f} and {x['first_kc_all']:+.4f}" for x in xs) + '.')
    json.dump(res, open(f'{OUT}/stats.json', 'w'), indent=1, default=float)

    def show(lab, o):
        if not o: print(lab, None); return
        m = o.get('mixed', {}); b = o['cluster_boot']; f = o.get('fixed_t', {})
        print(f"{lab:42s} mean {o['mean']:+.4f} med {o['median']:+.4f} pos {o['pos']}/{o['corpora']} units {o['units']} | mixed [{m.get('lo', np.nan):+.4f},{m.get('hi', np.nan):+.4f}] "
              f"t [{f.get('lo', np.nan):+.4f},{f.get('hi', np.nan):+.4f}] boot [{b['lo']:+.4f},{b['hi']:+.4f}] p {o.get('p_wilcoxon', np.nan):.3f} cls {o.get('class')}")
    for k in ('primary_5pct', 'primary_32', 'primary_50'): show(k, res[k]['first']); show(k + ' all', res[k]['all'])
    show('three_seeds', res['three_seeds']['first']); show('seed0 same units', res['three_seeds']['seed0_same_units'])
    print('delta_seed', d_seed, 'median', res['three_seeds']['control_seed_sd_median'])
    print('FDR', {k: v for k, v in res['fdr'].items() if k != 'terms'})
    for k, v in res['transfer'].items():
        if k != 'overlap': show(k, v['first'])
    print(res['transfer']['overlap'])
    for k, v in res['pdkt'].items(): show(k, v['first'])
    for k, v in res['cap500'].items(): show(k + ' 500', v['len500']); show(k + ' 200', v['len200'])


if __name__ == '__main__': main()
