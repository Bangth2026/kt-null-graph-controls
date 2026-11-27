"""Release 8.0.0: positive control by world (reviewer comment M2), with sim_dbe_mix, and the partial Q-matrix
perturbation (reviewer comment M8). No training.

1. results_r22/positive_control.json: every ladder term by world and training size (5%, 10%, 25% on subsamples 0-2;
   100% on subsample 0 for sim_dbe and sim_a09), with a t interval over the subsamples of each world, and pooled over
   the three worlds (nine units) for reference only.
2. results_r22/q_partial.json: item-level probe, real Q-matrix against Q-matrices with 10% and 20% of the questions'
   KCs permuted (q_pqm10, q_pqm20) and against the fully permuted Q-matrix (q_rqm, release 6.0.0), on first attempts at
   a KC and at a question; ten units of seven corpora; t interval over corpus means.
"""
import os, sys, json, glob, numpy as np
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r21_analyse as A
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r22'; os.makedirs(OUT, exist_ok=True)
WORLDS = ('sim_dbe', 'sim_dbe_mix', 'sim_a09')
DIRS = [f'{REPO}/runs_r20_sim', f'{REPO}/runs_r21_sim', f'{REPO}/runs_r22_sim']


def positive_control():
    out = {'by_world': {}, 'pooled': {}}
    D = {w: A.runs(DIRS, w) for w in WORLDS}
    for w, R in D.items():
        out['by_world'][w] = {}
        for f in (5, 10, 25, 100):
            subs = sorted({k[1] for k in R if k[0] == f}); res = {}
            for a, b in A.TERMS:
                for col, sfx in (('test_auc_first', ''), ('test_auc', ':all')):
                    v = [R[(f, s, a)][col] - R[(f, s, b)][col] for s in subs if (f, s, a) in R and (f, s, b) in R]
                    if v: res[f'{a}-{b}{sfx}'] = A.summ(v)
            if res: out['by_world'][w][str(f)] = res
    for f in (5, 10, 25):
        res = {}
        for a, b in A.TERMS:
            v = []
            for w, R in D.items():
                v += [R[(f, s, a)]['test_auc_first'] - R[(f, s, b)]['test_auc_first'] for s in range(3) if (f, s, a) in R and (f, s, b) in R]
            if v: res[f'{a}-{b}'] = A.summ(v)
        out['pooled'][str(f)] = res
    out['n_runs'] = {w: len(R) for w, R in D.items()}
    out['n_runs_r22'] = len(glob.glob(f'{REPO}/runs_r22_sim/*__steps.json'))
    json.dump(out, open(f'{OUT}/positive_control.json', 'w'), indent=1)
    return out


def q_partial():
    R = {}
    for d in (f'{REPO}/runs_r20', f'{REPO}/runs_r22'):
        for f in glob.glob(f'{d}/*__f10__s*__q_*__steps.json'):
            j = json.load(open(f)); R[(j['data'], int(j['sub']), j['cond'])] = j
    units = sorted({(k[0], k[1]) for k in R if k[2] == 'q_pqm10'})
    out = {'units': units}
    for a, b in (('q_qm', 'q_pqm10'), ('q_qm', 'q_pqm20'), ('q_qm', 'q_rqm'), ('q_pqm10', 'q_pqm20')):
        for col in ('test_auc_first', 'test_auc_first_item', 'test_auc'):
            per = {}
            for c, s in units:
                if (c, s, a) in R and (c, s, b) in R and R[(c, s, a)].get(col) is not None:
                    per.setdefault(c, []).append(R[(c, s, a)][col] - R[(c, s, b)][col])
            if not per: continue
            cm = np.array([np.mean(v) for v in per.values()]); n = len(cm); se = cm.std(ddof=1) / np.sqrt(n); q = tdist.ppf(0.975, n - 1)
            out[f'{a}-{b}|{col}'] = dict(mean=float(cm.mean()), pos=int((cm > 0).sum()), n=n, ci=[float(cm.mean() - q * se), float(cm.mean() + q * se)],
                                         per_corpus={c: float(np.mean(v)) for c, v in per.items()})
    rp = glob.glob(f'{REPO}/runs_r22/repro/statics11__f10__s0__q_qm__steps.json')
    if rp:
        a = json.load(open(rp[0]))['test_auc_first']; b = R[('statics11', 0, 'q_qm')]['test_auc_first']; out['repro'] = dict(rerun=a, archived=b, abs_diff=abs(a - b))
    json.dump(out, open(f'{OUT}/q_partial.json', 'w'), indent=1)
    return out


if __name__ == '__main__':
    pc = positive_control()
    for w in WORLDS:
        for f in ('5', '10', '25', '100'):
            v = pc['by_world'][w].get(f, {}).get('true_emb-true_dp_emb')
            if v: print(w, f, round(v['mean'], 4), v['pos'], v['n'], v['ci'] and [round(x, 4) for x in v['ci']])
    if glob.glob(f'{REPO}/runs_r22/*q_pqm*json'): print(json.dumps({k: (round(v['mean'], 4), v['pos'], v['n']) for k, v in q_partial().items() if isinstance(v, dict) and 'mean' in v}))
