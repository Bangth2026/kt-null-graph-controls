"""Release 8.0.0: statistics without training for the fourth UMUAI version (reviewer comments M1, M7, minor 9).

1. Primary comparison at 10% (update budget, 32 units) with the graph at both entry points and at (E) only, per unit and
   per corpus (forest plot, Fig. 6), with mixed-model, t (over corpus means) and corpus-clustered bootstrap intervals.
2. Every term of the null-graph ladders with both estimators side by side: the mean over corpora of subsample means and
   the mixed-model estimate, its 95% interval, a t interval over corpus means, the cluster bootstrap interval of
   release 4.x, and a t-based minimum detectable effect, MDE = (t_{0.975, n-1} + t_{0.80, n-1}) SE with n the number of
   corpora (the normal-quantile MDE of release 7.0.0 is about 12% smaller with ten corpora).
Output: results_r22/stats_r22.json, results_r22/fig6_forest.png
"""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_common as C
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r22'; os.makedirs(OUT, exist_ok=True)
CORP = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
SMALL = ('algebra05', 'statics11', 'assist17', 'bridge06')
U32 = [(c, s) for c in CORP for s in (range(5) if c in SMALL else range(2))]


def tmde(se, n): return float((tdist.ppf(0.975, n - 1) + tdist.ppf(0.80, n - 1)) * se)


def tci(vals):
    v = np.asarray(vals, float); n = len(v); m = v.mean(); se = v.std(ddof=1) / np.sqrt(n); q = tdist.ppf(0.975, n - 1)
    return [float(m - q * se), float(m + q * se)]


def boot(df, B=10000, seed=0):
    rng = np.random.default_rng(seed); cs = df.data.unique(); g = {c: df[df.data == c].d.values for c in cs}; out = []
    for _ in range(B):
        pick = rng.choice(cs, len(cs)); out.append(np.mean([g[c][rng.integers(0, len(g[c]), len(g[c]))].mean() for c in pick]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def term(rows):
    df = pd.DataFrame(rows); cm = df.groupby('data').d.mean(); mx = C.mixed(df)
    return dict(units=len(df), corpora=int(cm.size), mean=float(cm.mean()), pos=int((cm > 0).sum()), per_corpus=cm.to_dict(),
                per_unit=[(r.data, int(r.sub), float(r.d)) for r in df.itertuples()],
                mixed=dict(est=float(mx['est']), se=float(mx['se']), lo=float(mx['lo']), hi=float(mx['hi'])) if mx else None,
                t=tci(cm.values), boot=boot(df), mde_t=tmde(mx['se'], cm.size) if mx else None)


def main():
    ARCH, N13, REF = C.reference_runs(); L = C.ladder_runs(); R = dict(REF); R.update({k: v for k, v in L.items() if k not in R})
    out = {}
    for name, a, b in (('both: support - empty', 'support', 'empty'), ('E: support_emb - empty', 'support_emb', 'empty'),
                       ('E: support_emb - dp_emb', 'support_emb', 'dp_emb'), ('E: dp_emb - er_emb', 'dp_emb', 'er_emb'),
                       ('E: er_emb - uniform_emb', 'er_emb', 'uniform_emb'), ('E: uniform_emb - empty', 'uniform_emb', 'empty')):
        for col in ('test_auc_first', 'test_auc'):
            rows = C.pairs(R, U32, a, b, col=col)
            if rows: out[f'{name} | {col}'] = term(rows)
    # every ladder term of release 4.x with a t-based MDE
    E = json.load(open(f'{REPO}/results_r14/r14_results.json'))
    lad = {}
    for blk in ('probe_E', 'state', 'sakte_gpu'):
        for k, v in E['equivalence'][blk].items():
            f = v['first']; mx = f['mixed']
            lad[f'{blk}/{k}'] = dict(mean=f['mean'], pos=f['pos'], n=f['n'], est=mx['est'], se=mx['se'], lo=mx['lo'], hi=mx['hi'],
                                     t=tci(list(f['per_corpus'].values())), boot=f.get('boot'), mde_t=tmde(mx['se'], f['n']),
                                     all_mean=v['all']['mean'], all_pos=v['all']['pos'])
    for k, v in E['gkt_cpu_ladder'].items():
        f = v['first']; mx = f['mixed']
        lad[f'gkt/{k}'] = dict(mean=f['mean'], pos=f['pos'], n=f['n'], est=mx['est'], se=mx['se'], lo=mx['lo'], hi=mx['hi'],
                               t=tci(list(f['per_corpus'].values())), boot=f.get('boot'), mde_t=tmde(mx['se'], f['n']),
                               all_mean=v['all']['mean'], all_pos=v['all']['pos'])
    S19 = json.load(open(f'{REPO}/results_r19/stats.json'))
    for k, v in S19['pdkt'].items():
        f = v['first']; mx = f['mixed']
        lad[f'pdkt/{k}'] = dict(mean=f['mean'], pos=f['pos'], n=f.get('units', 10), est=mx['est'], se=mx['se'], lo=mx['lo'], hi=mx['hi'],
                                mde_t=tmde(mx['se'], 10), all_mean=v['all']['mean'], all_pos=v['all']['pos'])
    for k, v in S19['transfer'].items():
        if 'first' not in v: continue
        f = v['first']; mx = f['mixed']
        lad[f'transfer/{k}'] = dict(mean=f['mean'], pos=f['pos'], n=10, est=mx['est'], se=mx['se'], lo=mx['lo'], hi=mx['hi'], mde_t=tmde(mx['se'], 10))
    O16 = json.load(open(f'{REPO}/results_r16/r16_results.json'))['other_constructions']
    for k, v in O16.items():
        if not isinstance(v, dict) or 'first' not in v or 'mean' not in v['first']: continue
        f = v['first']
        lad[f'other/{k}'] = dict(mean=f['mean'], pos=f['pos'], n=f['corpora'], est=f['mixed'], se=f['se'], lo=f['ci95'][0], hi=f['ci95'][1],
                                 t=tci(list(f['per_corpus'].values())), boot=f.get('boot95'), mde_t=tmde(f['se'], f['corpora']),
                                 all_mean=v['all']['mean'], all_pos=v['all']['pos'])
    out['ladder'] = lad
    json.dump(out, open(f'{OUT}/stats_r22.json', 'w'), indent=1)
    fig(out)
    for k in ('both: support - empty | test_auc_first', 'E: support_emb - empty | test_auc_first', 'E: support_emb - dp_emb | test_auc_first'):
        v = out[k]; print(k, round(v['mean'], 4), v['pos'], v['units'], {a: [round(x, 4) for x in v[a]] for a in ('t', 'boot')}, round(v['mixed']['est'], 4), round(v['mde_t'], 4))


def fig(out):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
    NAME = dict(assist09='ASSISTments 2009', assist12='ASSISTments 2012', assist15='ASSISTments 2015', assist17='ASSISTments 2017', junyi='Junyi Academy',
                algebra05='Algebra 2005', eedi='Eedi', ednet='EdNet KT1', bridge06='Bridge to Alg. 2006', statics11='Statics 2011')
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, (k, title) in zip(axs, (('both: support - empty | test_auc_first', '(a) first attempts'), ('both: support - empty | test_auc', '(b) all predictions'))):
        v = out[k]; order = sorted(CORP, key=lambda c: v['per_corpus'][c])
        for i, c in enumerate(order):
            u = [d for (cc, s, d) in v['per_unit'] if cc == c]
            ax.scatter(u, [i] * len(u), s=8, color='0.6', marker='o', zorder=2)
            ax.scatter([np.mean(u)], [i], s=22, color='#1f5fa8', marker='D', zorder=3)
        lo, hi = v['mixed']['lo'], v['mixed']['hi']; m = v['mixed']['est']
        ax.axvspan(lo, hi, color='#c4581b', alpha=0.15, lw=0); ax.axvline(m, color='#c4581b', lw=1); ax.axvline(0, color='0.3', lw=0.6)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([NAME[c] for c in order]); ax.set_title(title, loc='left', fontsize=7.5)
        ax.set_xlabel('AUC difference, support graph − empty graph'); ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.tight_layout(); fig.savefig(f'{OUT}/fig6_forest.png', dpi=300); plt.close(fig)


if __name__ == '__main__': main()
