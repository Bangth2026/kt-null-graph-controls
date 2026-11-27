"""Release 8.0.0: decisions in the simulated worlds with training variability (reviewer comment M5).

As r21_sim.py (same learners, warm-up, horizon, gates, base rules and outcomes), with three changes:
  - tracers: the probe at 10% and 25% of the training learners on subsamples 0, 1 and 2 (not only subsample 0), given
    the empty graph, G, G* or DP(G*); 24 tracers per world. Intervals for an effect are t intervals over the six
    (subsample, size) cells, each cell averaging the per-learner difference over its learners and graph inputs, so they
    include training variability;
  - all references are reported: oracle (the KC whose TRUE success probability is closest to 0.65, which is regret-optimal
    by definition), oracle_frontier (the unmastered KC with the largest true probability of being learned now), the
    curriculum walk and random choice;
  - forgetting calibration: the expected number of KCs forgotten over the 40 steps by a learner who practised nothing,
    sum over mastered KCs after warm-up of 1 - (1 - F)^40.
Worlds: sim_dbe, sim_a09 (release 6.0.0) and sim_dbe_mix (r22_sim.py).
Output: results_r22/decisions_sim_<world>.json
"""
import os, sys, json, numpy as np, torch
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_sim as S, r21_sim as X
ROOT = S.ROOT; OUT = f'{ROOT}/results_r22'; os.makedirs(OUT, exist_ok=True)
RUNS = [f'{ROOT}/runs_r20_sim', f'{ROOT}/runs_r21_sim', f'{ROOT}/runs_r22_sim']
CONDS = ('empty', 'support_emb', 'true_emb', 'true_dp_emb')


def main(world):
    W = S.load_world(world); K = W['K']; G = X.gate_graphs(world, W); rate = X.kc_rate(world, K)
    L = S.Learners(W, S.N_TEST, seed=777)
    for t in range(S.WARM): ch = L.walk_choice(); L.step(ch, walk=True)
    m_warm = L.m.sum(1); forget = float((m_warm * (1 - (1 - S.FORGET) ** S.H)).mean())
    out = dict(world=world, K=K, edges_true=int(W['A'].sum()), edges_g100=int(G['g100'].sum()),
               g100_vs_true=dict(recall=float(((G['g100'] > 0) & (W['A'] > 0)).sum() / W['A'].sum()),
                                 precision=float(((G['g100'] > 0) & (W['A'] > 0)).sum() / max(G['g100'].sum(), 1))),
               mastered_after_warmup=float(m_warm.mean()), expected_forgotten_without_practice=forget, reference={}, kcrate={}, models={})
    for p in ('oracle', 'oracle_frontier', 'curriculum', 'random'):
        res, _ = S.run_loop(W, p); res.pop('per_learner', None); out['reference'][p] = res
    for g in X.GATES:
        r = X.run(W, 'kcrate', G[g], rate=rate); r.pop('first', None); out['kcrate'][g] = r
    for frac in (10, 25):
        for sub in (0, 1, 2):
            for cond in CONDS:
                tag = f'{world}__f{frac}__s{sub}__{cond}__steps'; f = None
                for d in RUNS:
                    if os.path.exists(f'{d}/{tag}.pt'): f = f'{d}/{tag}.pt'
                if f is None: print('missing', tag); continue
                m = torch.load(f, weights_only=False); js = json.load(open(f[:-3] + '.json'))
                rec = dict(cond=cond, frac=frac / 100, sub=sub, test_auc_first=js['test_auc_first'])
                for base in ('target', 'frontier'):
                    for g in X.GATES:
                        r = X.run(W, base, G[g], m=m); r.pop('first', None); rec[f'{base}|{g}'] = r
                out['models'][tag] = rec
            print(world, frac, sub, flush=True)
            json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))
    json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))


def effect(out, a, b, conds=CONDS, what='gain', inp=None):
    """Mean effect and t interval over the six (subsample, size) cells. a, b: policy keys; with inp=(c1, c0) the effect
    of the tracer input c1 against c0 under policy a."""
    cells = {}
    for r in out['models'].values():
        key = (r['frac'], r['sub'])
        if inp is None:
            if r['cond'] not in conds: continue
            d = np.mean(r[a]['per_learner'][what]) - np.mean(r[b]['per_learner'][what])
            cells.setdefault(key, []).append(d)
    if inp is not None:
        by = {(r['frac'], r['sub'], r['cond']): r for r in out['models'].values()}
        for (fr, sb, c) in by:
            if c == inp[0] and (fr, sb, inp[1]) in by:
                cells[(fr, sb)] = [np.mean(by[(fr, sb, c)][a]['per_learner'][what]) - np.mean(by[(fr, sb, inp[1])][a]['per_learner'][what])]
    v = np.array([np.mean(x) for x in cells.values()]); n = len(v)
    se = v.std(ddof=1) / np.sqrt(n); q = tdist.ppf(0.975, n - 1)
    return dict(mean=float(v.mean()), lo=float(v.mean() - q * se), hi=float(v.mean() + q * se), pos=int((v > 0).sum()), n=n)


if __name__ == '__main__':
    main(sys.argv[1])
