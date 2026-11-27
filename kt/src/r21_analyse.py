"""Release 7.0.0: analyses of the runs of the third UMUAI version (no training).

1. Positive control (results_r21/positive_control.json): in the simulated worlds, every term of the ladder on the
   true graph G* (empty, uni_emb = U, true_er_emb = ER(G*), true_dp_emb = DP(G*), true_emb = G*) and on the log-built
   support graph G (er_emb, dp_emb, support_emb), on first attempts and all predictions, by world and training size
   (subsamples 0-2 at 5%, 10%, 25%; subsample 0 at 100%, from runs_r20_sim). Pooled over the two worlds the unit is
   (world, subsample): mean, units with a positive difference, t interval over the units, and the MDE of the same term.
2. XES3G5M routes (results_r21/xes_routes.json): route_emb against route_dp_emb, empty and support_emb (runs_r14), at
   10% on five subsamples; t interval over subsamples.
"""
import os, sys, json, glob, numpy as np
from scipy.stats import t as tdist, norm
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r21'; os.makedirs(OUT, exist_ok=True)
Z = norm.ppf(0.975) + norm.ppf(0.80)
TERMS = [('uni_emb', 'empty'), ('true_er_emb', 'uni_emb'), ('true_dp_emb', 'true_er_emb'), ('true_emb', 'true_dp_emb'), ('true_emb', 'empty'),
         ('er_emb', 'uni_emb'), ('dp_emb', 'er_emb'), ('support_emb', 'dp_emb'), ('support_emb', 'empty'), ('true_emb', 'support_emb')]


def runs(dirs, data):
    R = {}
    for d in dirs:
        for f in glob.glob(f'{d}/{data}__*__steps.json'):
            j = json.load(open(f))
            if int(j.get('rep', 0) or 0) != 0: continue
            R[(int(round(j['frac'] * 100)), int(j['sub']), j['cond'])] = j
    return R


def summ(v):
    v = np.asarray(v, float); n = len(v)
    if n == 0: return None
    m = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    ci = [float(m - tdist.ppf(0.975, n - 1) * se), float(m + tdist.ppf(0.975, n - 1) * se)] if n > 1 else None
    return dict(mean=m, n=n, pos=int((v > 0).sum()), se=se, ci=ci, mde80=(Z * se if se else None), values=v.tolist())


def positive_control():
    out = {'by_world': {}, 'pooled': {}}
    D = {w: runs([f'{REPO}/runs_r20_sim', f'{REPO}/runs_r21_sim'], w) for w in ('sim_dbe', 'sim_a09')}
    for w, R in D.items():
        out['by_world'][w] = {}
        for f in (5, 10, 25, 100):
            subs = sorted({k[1] for k in R if k[0] == f})
            res = {}
            for a, b in TERMS:
                for col, sfx in (('test_auc_first', ''), ('test_auc', ':all')):
                    v = [R[(f, s, a)][col] - R[(f, s, b)][col] for s in subs if (f, s, a) in R and (f, s, b) in R]
                    if v: res[f'{a}-{b}{sfx}'] = summ(v)
            res['auc'] = {c: float(np.mean([R[(f, s, c)]['test_auc_first'] for s in subs if (f, s, c) in R]))
                          for c in {k[2] for k in R if k[0] == f}}
            out['by_world'][w][str(f)] = res
    for f in (5, 10, 25, 100):
        res = {}
        for a, b in TERMS:
            for col, sfx in (('test_auc_first', ''), ('test_auc', ':all')):
                v = []
                for w, R in D.items():
                    subs = sorted({k[1] for k in R if k[0] == f})
                    v += [R[(f, s, a)][col] - R[(f, s, b)][col] for s in subs if (f, s, a) in R and (f, s, b) in R]
                if v: res[f'{a}-{b}{sfx}'] = summ(v)
        out['pooled'][str(f)] = res
    out['n_runs'] = {w: len(R) for w, R in D.items()}
    out['n_runs_r21'] = len(glob.glob(f'{REPO}/runs_r21_sim/*__steps.json'))
    json.dump(out, open(f'{OUT}/positive_control.json', 'w'), indent=1)
    return out


def xes_routes():
    R = runs([f'{REPO}/runs_r14', f'{REPO}/runs_r21'], 'xes3g5m'); out = {}
    for a, b in (('route_emb', 'route_dp_emb'), ('route_emb', 'empty'), ('route_emb', 'support_emb'), ('support_emb', 'empty'), ('route_dp_emb', 'empty')):
        for col, sfx in (('test_auc_first', ''), ('test_auc', ':all')):
            v = [R[(10, s, a)][col] - R[(10, s, b)][col] for s in range(5) if (10, s, a) in R and (10, s, b) in R]
            if v: out[f'{a}-{b}{sfx}'] = summ(v)
    e = [R[(10, s, 'route_emb')]['edges_cond'] for s in range(5) if (10, s, 'route_emb') in R]
    out['route_edges'] = float(np.mean(e)) if e else None
    rp = glob.glob(f'{REPO}/runs_r21/repro/xes3g5m__f10__s0__empty__steps.json')
    if rp:
        a = json.load(open(rp[0]))['test_auc_first']; b = R[(10, 0, 'empty')]['test_auc_first']
        out['repro_empty_s0'] = dict(rerun=a, archived=b, abs_diff=abs(a - b))
    out['n_runs'] = len(glob.glob(f'{REPO}/runs_r21/xes3g5m__*__steps.json'))
    json.dump(out, open(f'{OUT}/xes_routes.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    pc = positive_control()
    for f in ('5', '10', '25', '100'):
        print(f, {k: (round(v['mean'], 4), v['pos'], v['n']) for k, v in pc['pooled'].get(f, {}).items() if ':all' not in k})
    print(json.dumps({k: ((round(v['mean'], 4), v['pos']) if 'mean' in v else v) if isinstance(v, dict) else v for k, v in xes_routes().items()}))
