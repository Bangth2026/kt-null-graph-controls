"""Release 7.0.0: decisions in the simulated worlds with a known prerequisite graph (reviewer comment 4.2).

The worlds, learners, warm-up, horizon and outcomes are those of r20_sim.py (prerequisite-gated BKT with forgetting,
500 fresh learners, 20 warm-up steps of the curriculum walk, H = 40 adaptive steps, common random numbers). What is new
is that a graph can act on the decision itself, as a gate on the candidate set, and that a model-free rule is added.

Gate. Before a base rule chooses, the candidates are restricted to the KCs whose prerequisites in a gate graph are all
"passed" by the learner, where a prerequisite is passed when the learner's last two answers on it were correct (a
model-free mastery criterion of the kind tutors use to unlock content). KCs without prerequisites in the gate graph
are always open. If no KC is open the rule chooses among all KCs. Gate graphs:
    none      no gate
    true      G*, the true prerequisite graph of the world
    dp        DP(G*), a degree-preserving rewiring of G* (seed 1000): same in- and out-degrees, wrong edges
    g100      the support graph (Eq. 2) of all training learners of the world's logs, the graph a platform could build
Base rules:
    target    the KC whose predicted P(correct) is closest to 0.65 (tracer), as r20_sim.py
    frontier  the KC with the highest predicted P among those predicted below 0.8 (tracer), as r20_sim.py
    kcrate    among KCs the learner has not attempted yet (all KCs once every KC is attempted), the KC whose proportion
              correct on first attempts among the training learners of subsample 0 at 10% is closest to 0.65; no tracer
Tracers: the probe at 10% and 25% of the training learners, subsample 0, given the empty graph, the support graph G,
G* or DP(G*) in the embedding refinement (runs_r20_sim, runs_r21_sim).
References (no tracer): oracle, oracle_frontier, curriculum, random, as r20_sim.py.
Output: results_r21/decisions_sim_<world>.json
Usage: python r21_sim.py <world>
"""
import sys, os, json, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_sim as S
from r20_sim import Learners, score_all, check_scores, TARGET, MASTERED, H, WARM, N_TEST, RHO
from r5_controls import load, rewire
import r19_controls as R19
ROOT = S.ROOT; OUT = f'{ROOT}/results_r21'; os.makedirs(OUT, exist_ok=True)
RUNS = [f'{ROOT}/runs_r20_sim', f'{ROOT}/runs_r21_sim']
GATES = ('none', 'true', 'dp', 'g100')


def gate_graphs(world, W):
    K = W['K']; A = (W['A'] > 0).astype(np.float32)
    dp = rewire(A, seed=1000)[0].astype(np.float32)
    df = load(world, 1.0, 0)
    a = argparse.Namespace(data=world, frac=1.0, sub=0, cond='support')
    g100, _ = R19.support_of(df, a, K)
    return {'none': None, 'true': A, 'dp': dp, 'g100': g100[:K, :K]}


def kc_rate(world, K):
    df = load(world, 0.1, 0); tr = df[df.split == 'train']
    f = tr.drop_duplicates(['user', 'kc']); r = f.groupby('kc').correct.mean().reindex(range(K))
    return r.fillna(f.correct.mean()).values


def passed(hq, hr, t, K):
    """[n, K] bool: the learner's last two answers on the KC (within the first t steps) were correct."""
    n = hq.shape[0]; last = np.full((n, K), -1); prev = np.full((n, K), -1)
    for s in range(t):
        k = hq[:, s]; r = hr[:, s]; idx = np.arange(n)
        prev[idx, k] = last[idx, k]; last[idx, k] = r
    return (last == 1) & (prev == 1)


def attempted(hq, t, K):
    n = hq.shape[0]; a = np.zeros((n, K), bool); a[np.arange(n)[:, None], hq[:, :t]] = True; return a


def open_set(G, P):
    if G is None: return np.ones_like(P)
    Gb = G > 0; npre = Gb.sum(0)                                   # prerequisites of each KC (column sums)
    unmet = (~P).astype(np.int32) @ Gb.astype(np.int32)              # [n, K]: number of unmet prerequisites
    return unmet == 0


def run(W, base, gate, m=None, rate=None, seed=777):
    L = Learners(W, N_TEST, seed=seed); K = W['K']
    hq = np.zeros((N_TEST, WARM + H), int); hr = np.zeros((N_TEST, WARM + H), int)
    for t in range(WARM):
        ch = L.walk_choice(); hq[:, t] = ch; hr[:, t] = L.step(ch, walk=True)
    m0 = L.m.sum(1).copy(); ttm = np.full(N_TEST, H + 1); fails = np.zeros(N_TEST, int); frus = np.zeros(N_TEST)
    succ = np.zeros(N_TEST); reg = np.zeros(N_TEST); first = None; opened = np.zeros(N_TEST); blocked_true = np.zeros(N_TEST)
    Atrue = W['A'] > 0
    for t in range(H):
        T = WARM + t
        Ok = open_set(gate, passed(hq, hr, T, K))
        none = ~Ok.any(1); Ok[none] = True; opened += Ok.mean(1)
        if base == 'kcrate':
            att = attempted(hq, T, K); cand = Ok & ~att; cand[~cand.any(1)] = Ok[~cand.any(1)]
            d = np.where(cand, np.abs(rate[None, :] - TARGET), np.inf); ch = d.argmin(1)
        else:
            Sc = np.concatenate([score_all(m, hq[i:i + 100, :T], hr[i:i + 100, :T]) for i in range(0, N_TEST, 100)])[:, :K]
            if Sc.shape[1] < K:                                            # KCs absent from the training logs: never candidates
                Sc = np.concatenate([Sc, np.full((N_TEST, K - Sc.shape[1]), np.nan)], 1)
                Ok = Ok & ~np.isnan(Sc); Ok[~Ok.any(1)] = ~np.isnan(Sc[~Ok.any(1)])
            if base == 'target':
                ch = np.where(Ok, np.abs(Sc - TARGET), np.inf).argmin(1)
            elif base == 'frontier':
                X = np.where(Ok & (np.nan_to_num(Sc, nan=1.0) < MASTERED), Sc, -np.inf); ch = X.argmax(1); nn = ~np.isfinite(X).any(1)
                ch[nn] = np.where(Ok[nn], np.nan_to_num(Sc[nn], nan=np.inf), np.inf).argmin(1)
            else:
                raise ValueError(base)
        # was the chosen KC blocked by the TRUE state (some true prerequisite not mastered)?
        npre = Atrue[:, ch].T                                          # [n, K] prerequisites of the chosen KC
        blocked_true += ((npre & ~L.m).any(1)).astype(float)
        pt = L.p_true(); reg += np.abs(pt[np.arange(N_TEST), ch] - TARGET) - np.abs(pt - TARGET).min(1)
        y = L.step(ch, walk=False); hq[:, T] = ch; hr[:, T] = y
        if first is None: first = ch.copy()
        fails = np.where(y == 0, fails + 1, 0); frus += (fails >= 3); succ += y
        new = L.m.sum(1) - m0; ttm = np.where((ttm > H) & (new >= 5), t + 1, ttm)
    gain = L.m.sum(1) - m0
    return dict(gain=float(gain.mean()), ttm5=float(ttm.mean()), reached5=float((ttm <= H).mean()),
                frustration=float((frus / H).mean()), success=float((succ / H).mean()), regret=float((reg / H).mean()),
                open_share=float((opened / H).mean()), blocked_by_true_state=float((blocked_true / H).mean()),
                per_learner=dict(gain=gain.tolist(), ttm5=ttm.tolist(), regret=(reg / H).tolist(), frus=(frus / H).tolist()),
                first=first.tolist())


def main(world):
    W = S.load_world(world); K = W['K']; G = gate_graphs(world, W); rate = kc_rate(world, K)
    out = {'world': world, 'K': K, 'edges_true': int(W['A'].sum()), 'edges_g100': int(G['g100'].sum()),
           'g100_vs_true': dict(recall=float(((G['g100'] > 0) & (W['A'] > 0)).sum() / max(W['A'].sum(), 1)),
                                precision=float(((G['g100'] > 0) & (W['A'] > 0)).sum() / max(G['g100'].sum(), 1))),
           'H': H, 'warm': WARM, 'n': N_TEST, 'reference': {}, 'kcrate': {}, 'models': {}}
    for p in ('oracle', 'oracle_frontier', 'curriculum', 'random'):
        res, _ = S.run_loop(W, p); res.pop('per_learner', None); out['reference'][p] = res
    for g in GATES:
        out['kcrate'][g] = run(W, 'kcrate', G[g], rate=rate); print(world, 'kcrate', g, round(out['kcrate'][g]['gain'], 3), flush=True)
    for frac in (10, 25):
        for cond in ('empty', 'support_emb', 'true_emb', 'true_dp_emb'):
            tag = f'{world}__f{frac}__s0__{cond}__steps'; f = None
            for d in RUNS:
                if os.path.exists(f'{d}/{tag}.pt'): f = f'{d}/{tag}.pt'
            if f is None: print('missing', tag); continue
            m = torch.load(f, weights_only=False); js = json.load(open(f[:-3] + '.json'))
            rec = dict(cond=cond, frac=frac / 100, test_auc=js['test_auc'], test_auc_first=js['test_auc_first'],
                       score_check_max_abs_diff=check_scores(m, m.K))
            for base in ('target', 'frontier'):
                for g in GATES:
                    rec[f'{base}|{g}'] = run(W, base, G[g], m=m)
            out['models'][tag] = rec
            print(tag, {k: round(v['gain'], 2) for k, v in rec.items() if isinstance(v, dict)}, flush=True)
            json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))
    json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
