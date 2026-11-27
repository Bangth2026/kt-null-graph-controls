"""Release 9.0.0: analyses of the third UMUAI version's second revision (no training).

1. results_r23/positive_control.json: as r22_analyse.positive_control, with subsamples 3 and 4 of sim_dbe_mix
   (runs_r23_sim: empty, true_emb, true_dp_emb, support_emb, dp_emb at 5%, 10%, 25%), so that sim_dbe_mix has five
   subsamples per size for the terms of Table 2. For each world a pooled estimate over the three sizes: the mean of
   the three size means, with a stratified standard error sqrt(sum_s var_s / n_s) / 3 and a t interval with
   df = (number of cells) - 3. Pooling over sizes is not pooling over worlds.
2. results_r23/junyi_map.json: the published Junyi prerequisite map against its degree-preserving rewiring (map_dp_emb,
   subsamples 1-4 in runs_r23, subsample 0 in runs_r20) and against the empty graph and the support graph, five
   subsamples at 10%, first attempts; t interval over subsamples and t-based MDE.
"""
import os, sys, json, glob, numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21_analyse as A
from r14_common import load_runs
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r23'; os.makedirs(OUT, exist_ok=True)
WORLDS = ('sim_dbe', 'sim_dbe_mix', 'sim_a09')
DIRS = [f'{REPO}/runs_r20_sim', f'{REPO}/runs_r21_sim', f'{REPO}/runs_r22_sim', f'{REPO}/runs_r23_sim']


def tsumm(v):
    v = np.asarray(v, float); n = len(v)
    if n < 2: return dict(mean=float(v.mean()), n=n, pos=int((v > 0).sum()), ci=None, values=v.tolist())
    se = v.std(ddof=1) / np.sqrt(n); q = tdist.ppf(0.975, n - 1)
    return dict(mean=float(v.mean()), n=n, pos=int((v > 0).sum()), se=float(se), ci=[float(v.mean() - q * se), float(v.mean() + q * se)],
                mde80=float((q + tdist.ppf(0.8, n - 1)) * se), values=v.tolist())


def positive_control():
    out = {'by_world': {}, 'over_sizes': {}}
    D = {w: A.runs(DIRS, w) for w in WORLDS}
    for w, R in D.items():
        out['by_world'][w] = {}; cells = {}
        for f in (5, 10, 25, 100):
            subs = sorted({k[1] for k in R if k[0] == f}); res = {}
            for a, b in A.TERMS:
                for col, sfx in (('test_auc_first', ''), ('test_auc', ':all')):
                    v = [R[(f, s, a)][col] - R[(f, s, b)][col] for s in subs if (f, s, a) in R and (f, s, b) in R]
                    if v:
                        res[f'{a}-{b}{sfx}'] = tsumm(v)
                        if f != 100: cells.setdefault(f'{a}-{b}{sfx}', {})[f] = v
            if res: out['by_world'][w][str(f)] = res
        pooled = {}
        for term, by in cells.items():
            if len(by) < 3: continue
            means = [np.mean(v) for v in by.values()]; var = [np.var(v, ddof=1) / len(v) for v in by.values()]
            n = sum(len(v) for v in by.values()); se = np.sqrt(sum(var)) / 3; q = tdist.ppf(0.975, n - 3); m = float(np.mean(means))
            pooled[term] = dict(mean=m, se=float(se), ci=[m - q * se, m + q * se], cells=n, pos=int(sum((np.array(v) > 0).sum() for v in by.values())),
                                per_size={str(f): float(np.mean(v)) for f, v in by.items()}, n_per_size={str(f): len(v) for f, v in by.items()})
        out['over_sizes'][w] = pooled
    out['n_runs'] = {w: len(R) for w, R in D.items()}; out['n_runs_r23'] = len(glob.glob(f'{REPO}/runs_r23_sim/*__steps.json'))
    json.dump(out, open(f'{OUT}/positive_control.json', 'w'), indent=1)
    for w in WORLDS:
        v = out['over_sizes'][w].get('true_emb-true_dp_emb'); va = out['over_sizes'][w].get('true_emb-true_dp_emb:all')
        print(w, {f: (round(x['true_emb-true_dp_emb']['mean'], 4), x['true_emb-true_dp_emb']['n']) for f, x in out['by_world'][w].items() if 'true_emb-true_dp_emb' in x},
              'over sizes', round(v['mean'], 4), [round(x, 4) for x in v['ci']], v['pos'], v['cells'], '| all', round(va['mean'], 4), [round(x, 4) for x in va['ci']])
    return out


def junyi_map():
    R = {}
    for d in sorted(glob.glob(f'{REPO}/runs*')):
        if not os.path.isdir(d): continue
        for f in glob.glob(f'{d}/junyi__f10__s*__steps.json'):
            j = json.load(open(f))
            if j.get('budget', 'steps') != 'steps' or int(j.get('rep', 0) or 0) != 0 or j.get('l2', 0) not in (0, 0.0, None): continue
            if int(j.get('maxlen', 200) or 200) != 200 or round(float(j.get('lr', 0.002)), 4) != 0.002: continue
            R.setdefault((int(j['sub']), j['cond']), j)
    out = {}
    for a, b in (('map_emb', 'map_dp_emb'), ('map_emb', 'empty'), ('map_emb', 'support_emb'), ('map_dp_emb', 'empty')):
        for col in ('test_auc_first', 'test_auc'):
            v = [R[(s, a)][col] - R[(s, b)][col] for s in range(5) if (s, a) in R and (s, b) in R]
            if v: out[f'{a}-{b}|{col}'] = tsumm(v)
    json.dump(out, open(f'{OUT}/junyi_map.json', 'w'), indent=1)
    print({k: (round(v['mean'], 4), v['n'], v['pos'], v['ci'] and [round(x, 4) for x in v['ci']], round(v.get('mde80', 0), 4)) for k, v in out.items()})
    return out


if __name__ == '__main__':
    what = sys.argv[1:] or ['pc', 'map']
    if 'pc' in what: positive_control()
    if 'map' in what: junyi_map()
