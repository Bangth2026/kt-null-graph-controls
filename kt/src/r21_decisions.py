"""Release 7.0.0: further next-KC decision analyses on the ten corpora (reviewer comments 4.2, 4.3 and minor 11;
questions 2, 3 and 5). No training: the candidate scores, simulators and rules are those of r19_decisions.py.

1. KC-rate rule: the candidate score of every KC is its proportion correct among the training learners of the
   subsample (prior 0.5 with weight 5, as r19_baselines.py), at 5%, 25% and 100% (subsample 0). It is scored by the
   same rules and simulators as the neural decision models, so its regret can be set beside theirs.
2. Set-level agreement: for two models at a decision point, the Jaccard overlap of their near-tie sets (candidates
   within e of their best |p - tau|, e = 0.01 and 0.02) and the overlap of their five best candidates under the target
   rule (|top-5 A & top-5 B| / 5).
3. Sensitivity to the target tau (0.50, 0.60, 0.65, 0.70, 0.80): agreement of the target rule between control seeds and
   between graph and control with seeds crossed, and the change in regret by the graph under both simulators.
4. Seed ensembles at 25%: the mean of the candidate scores of the three seeds of the control and of the graph model;
   agreement of the two ensembles and their regret, against single-seed pairs.
Output: results_r21/decisions_more.json
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r19_decisions as D
from r5_controls import load
from r14_controls import first_encounter_rank
REPO = D.REPO; OUT = f'{REPO}/results_r21'; os.makedirs(OUT, exist_ok=True)
TAUS = (0.5, 0.6, 0.65, 0.7, 0.8)


def kc_rate_scores(c, frac, K, n, L):
    df = load(c, frac / 100, 0); tr = df[df.split == 'train']
    g = tr.groupby('kc').correct.agg(['sum', 'count']).reindex(range(K)).fillna(0)
    r = ((g['sum'] + 2.5) / (g['count'] + 5)).values.astype(np.float32)
    return np.broadcast_to(r, (n, L, K)).copy()


def target_choice(a, cnd, tau):
    idx = np.flatnonzero(cnd); return idx[np.argmin(np.abs(a[idx] - tau))]


def sets(a, cnd, tau, e):
    idx = np.flatnonzero(cnd); d = np.abs(a[idx] - tau)
    near = set(idx[d <= d.min() + e]); top5 = set(idx[np.argsort(d, kind='stable')[:5]]); return near, top5


def valid_points(score, seen):
    ok = ~np.isnan(score[..., 0].astype(np.float32))
    return [(i, t) for i, t in zip(*np.nonzero(ok)) if (~seen[i, t]).sum() >= 2]


def regret_tau(sc, sim, seen, pts, tau):
    v = []
    for i, t in pts:
        cnd = ~seen[i, t]; s = sim[i, t].astype(np.float64)
        if np.isnan(s[cnd]).any(): continue
        k = target_choice(sc[i, t].astype(np.float32), cnd, tau); d = np.abs(s - tau); v.append(d[k] - d[cnd].min())
    return float(np.mean(v))


def main():
    out = {'per_corpus': {}}
    for c in D.CORPORA:
        M = D.models_of(c)
        ref = np.load(M[(25, 0, 'support', 0)] + '_cand.npz'); seen = ref['seen']; n, L, K = seen.shape
        sc = {k: np.load(p + '_cand.npz')['score'][..., :K] for k, p in M.items()}
        df25 = load(c, 0.25, 0); rank = first_encounter_rank(df25[df25.split == 'train'], max(K, int(df25.kc.max()) + 1))[:K]
        rank = np.where(np.isfinite(rank), rank, 10.0)
        sims = {'rasch': D.rasch(c, n, K)[..., :K], 'pfa': D.pfa(c, n, K)[0][..., :K]}
        pts = valid_points(sc[(25, 0, 'support', 0)], seen)
        res = {}
        # 1. KC-rate rule against the neural models of the same training size
        kr = {}
        for frac in (5, 25, 100):
            S = kc_rate_scores(c, frac, K, n, L); dec = D.decisions(S, seen, rank)
            neur = [k for k in M if k[0] == frac and k[1] == 0]
            kr[frac] = {sl: dict(kc_rate=D.regret(dec, sim, seen),
                                 neural={f'{k[2]}|{k[3]}': D.regret(D.decisions(sc[k], seen, rank), sim, seen) for k in neur})
                        for sl, sim in sims.items()}
        res['kc_rate_rule'] = kr
        # 2-4. agreement on sets, tau, ensembles (25%, subsample 0)
        CC = [(25, 0, 'empty', r) for r in (0, 1, 2)]; GG = [(25, 0, 'support', r) for r in (0, 1, 2)]
        pairs = {'control_seeds': [(CC[i], CC[j]) for i, j in ((0, 1), (0, 2), (1, 2))],
                 'graph_control_crossed': [(GG[i], CC[j]) for i in range(3) for j in range(3) if i != j],
                 'graph_control_shared': [(GG[i], CC[i]) for i in range(3)]}
        ens = {'control': np.mean([sc[k] for k in CC], 0), 'graph': np.mean([sc[k] for k in GG], 0)}
        def pair_stats(A, B, tau=0.65):
            same = []; jn1 = []; jn2 = []; t5 = []
            for i, t in pts:
                cnd = ~seen[i, t]; a = A[i, t].astype(np.float32); b = B[i, t].astype(np.float32)
                same.append(target_choice(a, cnd, tau) == target_choice(b, cnd, tau))
                if tau == 0.65:
                    for e, J in ((0.01, jn1), (0.02, jn2)):
                        na, ta = sets(a, cnd, tau, e); nb, tb = sets(b, cnd, tau, e)
                        J.append(len(na & nb) / len(na | nb))
                    t5.append(len(ta & tb) / 5)
            o = dict(target=float(np.mean(same)))
            if tau == 0.65: o.update(jaccard_near_0p01=float(np.mean(jn1)), jaccard_near_0p02=float(np.mean(jn2)), top5=float(np.mean(t5)))
            return o
        ag = {}
        for name, pp in pairs.items():
            ag[name] = pd.DataFrame([pair_stats(sc[a], sc[b]) for a, b in pp]).mean().to_dict()
        ag['ensembles_graph_control'] = pair_stats(ens['graph'], ens['control'])
        ag['ensemble_vs_member_control'] = pd.DataFrame([pair_stats(ens['control'], sc[k]) for k in CC]).mean().to_dict()
        res['agreement_sets'] = ag
        tau = {}
        for tv in TAUS:
            tau[tv] = dict(control_seeds=float(np.mean([pair_stats(sc[a], sc[b], tv)['target'] for a, b in pairs['control_seeds']])),
                           graph_control_crossed=float(np.mean([pair_stats(sc[a], sc[b], tv)['target'] for a, b in pairs['graph_control_crossed']])),
                           **{f'regret_change_{sl}': float(np.mean([regret_tau(sc[g], sim, seen, pts, tv) for g in GG]) -
                                                         np.mean([regret_tau(sc[k], sim, seen, pts, tv) for k in CC]))
                              for sl, sim in sims.items()},
                           **{f'seed_sd_{sl}': float(np.std([regret_tau(sc[k], sim, seen, pts, tv) for k in CC], ddof=1)) for sl, sim in sims.items()})
        res['tau'] = tau
        res['ensemble_regret'] = {sl: dict(control=regret_tau(ens['control'], sim, seen, pts, 0.65),
                                           graph=regret_tau(ens['graph'], sim, seen, pts, 0.65),
                                           control_members=[regret_tau(sc[k], sim, seen, pts, 0.65) for k in CC],
                                           graph_members=[regret_tau(sc[k], sim, seen, pts, 0.65) for k in GG]) for sl, sim in sims.items()}
        out['per_corpus'][c] = res
        print(c, {k: round(v['target'], 3) for k, v in ag.items()}, flush=True)
        json.dump(out, open(f'{OUT}/decisions_more.json', 'w'), indent=1, default=float)
    json.dump(out, open(f'{OUT}/decisions_more.json', 'w'), indent=1, default=float)


if __name__ == '__main__': main()
