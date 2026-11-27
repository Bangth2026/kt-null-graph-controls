"""Release 8.0.0: next-KC decisions with a learner-by-KC simulator, realistic candidate sets and chance-corrected
agreement (reviewer comments M3 and M4). No training: candidate scores, positions and models are those of
r19_decisions.py (25% of the training learners, subsample 0, seeds 0, 100, 200; 5% and 100% decision models).

1. MIRT simulator. P(correct) = sigmoid(theta_u . a_k + c_u - b_k) with two latent dimensions, fitted by penalised
   maximum likelihood (L2, lambda = 1, L-BFGS) to every observed answer of the test learners, like the Rasch learner of
   r19_decisions.py. Unlike the Rasch learner and the global-practice simulator, it ranks KCs differently for different
   learners (learner-by-KC interaction). Both log-fitted simulators see each learner's whole log, including answers
   after the decision positions; they serve as an offline reference, not as a forecast.
2. Regret of every decision model and of the KC-rate rule under MIRT (target and easiest rules).
3. Chance-corrected agreement: kappa = (agreement - chance) / (1 - chance), with chance the mean of 1/|C| over decision
   points, C the candidate set.
4. Realistic candidate sets, applied to every model and rule:
     window  the ten unattempted KCs that come next in curriculum order after the learner's current KC (mean rank at
             which the training learners at 25% first meet each KC); fewer if fewer remain
     unit    unattempted KCs of the current KC's curriculum unit and of the next unit (units ordered by mean rank), on
             the six corpora whose releases name units; points with fewer than two such candidates are dropped
   For each set: agreement (raw and kappa) of control seeds, of graph and control with seeds crossed, and regret change
   by the graph under the three simulators.
Output: results_r22/decisions_r22.json
"""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r19_decisions as D
from r5_controls import load
from r14_controls import first_encounter_rank
from train import seqs
REPO = D.REPO; OUT = f'{REPO}/results_r22'; os.makedirs(OUT, exist_ok=True)
UNITS = ('junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11')
torch.set_num_threads(1)


def mirt(c, n, K, d=2, lam=1.0):
    import glob
    P = np.load(glob.glob(f'{REPO}/runs_r13/{c}__f100__s0__empty__steps_pred.npz')[0])
    u = torch.tensor(P['uid'].astype(np.int64)); k = torch.tensor(P['kc'].astype(np.int64)); y = torch.tensor(P['y'].astype(np.float32))
    N = int(u.max()) + 1; Kx = max(K, int(k.max()) + 1); g = torch.Generator().manual_seed(0)
    th = (0.1 * torch.randn(N, d, generator=g)).requires_grad_(); a = (0.1 * torch.randn(Kx, d, generator=g)).requires_grad_()
    cu = torch.zeros(N, requires_grad=True); b = torch.zeros(Kx, requires_grad=True)
    opt = torch.optim.LBFGS([th, a, cu, b], max_iter=500, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); z = (th[u] * a[k]).sum(1) + cu[u] - b[k]
        L = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='sum') + 0.5 * lam * (th.pow(2).sum() + a.pow(2).sum() + cu.pow(2).sum() + b.pow(2).sum())
        L.backward(); return L
    opt.step(closure)
    with torch.no_grad():
        S = torch.sigmoid(th[:n] @ a.T + cu[:n, None] - b[None, :]).numpy()
        z = (th[u] * a[k]).sum(1) + cu[u] - b[k]; ll = float(-torch.nn.functional.binary_cross_entropy_with_logits(z, y))
    return np.repeat(S[:, None, :], len(D.T), axis=1), dict(mean_loglik=ll, N=N)


def current_kc(c, n):
    df = load(c, 1.0, 0); Kd = int(df.kc.max()) + 1; q, r, _ = seqs(df, Kd, 'test'); q = q.numpy()
    cur = np.full((n, len(D.T)), -1)
    for i in range(min(n, len(q))):
        for li, t in enumerate(D.T):
            if t - 1 < q.shape[1] and q[i, t - 1] < Kd: cur[i, li] = q[i, t - 1]
    return cur


def unit_of(c, K):
    if c not in UNITS: return None
    g = json.load(open(f'{REPO}/data/{c}_groups.json')); df = load(c, 0.25, 0)
    names = df[['kc', 'kc_name']].drop_duplicates('kc').set_index('kc').kc_name.to_dict()
    return np.array([g.get(str(names.get(k)), None) if names.get(k) is not None else None for k in range(K)], object)


def candidate_sets(seen, cur, rank, units):
    n, L, K = seen.shape; win = np.zeros_like(seen); uni = np.zeros_like(seen) if units is not None else None
    if units is not None:
        uk = [u for u in set(units) if u is not None]
        urank = {u: np.mean([rank[k] for k in range(K) if units[k] == u]) for u in uk}
        uorder = sorted(uk, key=lambda u: urank[u]); nxt = {u: (uorder[i + 1] if i + 1 < len(uorder) else None) for i, u in enumerate(uorder)}
    for i in range(n):
        for l in range(L):
            cnd = np.flatnonzero(~seen[i, l]); c0 = cur[i, l]
            if c0 < 0 or len(cnd) < 2: continue
            after = cnd[rank[cnd] >= rank[c0]]; after = after[np.argsort(rank[after], kind='stable')][:10]
            if len(after) < 10:
                rest = np.setdiff1d(cnd, after); rest = rest[np.argsort(np.abs(rank[rest] - rank[c0]), kind='stable')][:10 - len(after)]
                after = np.concatenate([after, rest])
            win[i, l, after] = True
            if units is not None and units[c0] is not None:
                ok = [k for k in cnd if units[k] is not None and (units[k] == units[c0] or units[k] == nxt[units[c0]])]
                if len(ok) >= 2: uni[i, l, ok] = True
    return win, uni


def stats(sc, M, cset, sims, rank):
    """Agreement (raw, kappa) for control seeds and crossed pairs, regret of each model, under candidate mask cset."""
    ok = ~np.isnan(sc[(25, 0, 'support', 0)][..., 0].astype(np.float32))           # positions the learner reached
    pts = [(i, l) for i, l in zip(*np.nonzero((cset.sum(2) >= 2) & ok))]
    if not pts: return None
    chance = float(np.mean([1 / cset[i, l].sum() for i, l in pts]))
    def ch(a, i, l): idx = np.flatnonzero(cset[i, l]); return idx[np.argmin(np.abs(a[i, l, idx].astype(np.float32) - D.TARGET))]
    CC = [(25, 0, 'empty', r) for r in (0, 1, 2)]; GG = [(25, 0, 'support', r) for r in (0, 1, 2)]
    def agree(A, B): return float(np.mean([ch(sc[A], i, l) == ch(sc[B], i, l) for i, l in pts]))
    seeds = np.mean([agree(CC[a], CC[b]) for a, b in ((0, 1), (0, 2), (1, 2))])
    cross = np.mean([agree(GG[a], CC[b]) for a in range(3) for b in range(3) if a != b])
    kap = lambda p: (p - chance) / (1 - chance)
    reg = {}
    for sl, sim in sims.items():
        def rg(m):
            v = []
            for i, l in pts:
                s = sim[i, l].astype(np.float64); idx = np.flatnonzero(cset[i, l])
                if np.isnan(s[idx]).any(): continue
                k = ch(sc[m], i, l); d = np.abs(s - D.TARGET); v.append(d[k] - d[idx].min())
            return float(np.mean(v))
        reg[sl] = dict(control=[rg(m) for m in CC], graph=[rg(m) for m in GG])
    return dict(points=len(pts), mean_candidates=float(np.mean([cset[i, l].sum() for i, l in pts])), chance=chance,
                seeds=float(seeds), crossed=float(cross), kappa_seeds=float(kap(seeds)), kappa_crossed=float(kap(cross)), regret=reg)


def main():
    out = {}
    for c in D.CORPORA:
        M = D.models_of(c); ref = np.load(M[(25, 0, 'support', 0)] + '_cand.npz'); seen = ref['seen']; n, L, K = seen.shape
        sc = {k: np.load(p + '_cand.npz')['score'][..., :K] for k, p in M.items()}
        df25 = load(c, 0.25, 0); rank = first_encounter_rank(df25[df25.split == 'train'], max(K, int(df25.kc.max()) + 1))[:K]
        rank = np.where(np.isfinite(rank), rank, 10.0)
        S_m, info = mirt(c, n, K)
        sims = {'rasch': D.rasch(c, n, K)[..., :K], 'global': D.pfa(c, n, K)[0][..., :K], 'mirt': S_m[..., :K]}
        res = {'mirt_fit': info}
        # regret of every model and of the KC rate under MIRT, all unattempted candidates (as r19/r21)
        from r21_decisions import kc_rate_scores
        dec_m = {}
        for k in M: dec_m['|'.join(map(str, k))] = D.regret(D.decisions(sc[k], seen, rank), sims['mirt'], seen)
        kr = {f: D.regret(D.decisions(kc_rate_scores(c, f, K, n, L), seen, rank), sims['mirt'], seen) for f in (5, 25, 100)}
        res['mirt_regret'] = dict(models=dec_m, kc_rate=kr)
        # rank correlation between learners of MIRT's KC ordering (how much learner-by-KC interaction it has)
        S0 = S_m[:, 0, :]; rk = np.argsort(np.argsort(-S0, 1), 1)
        res['mirt_learner_rank_corr'] = float(np.mean([np.corrcoef(rk[i], rk[j])[0, 1] for i, j in zip(range(0, n - 1, 2), range(1, n, 2))]))
        allc = ~seen; cur = current_kc(c, n); units = unit_of(c, K); win, uni = candidate_sets(seen, cur, rank, units)
        res['sets'] = {'all': stats(sc, M, allc, sims, rank), 'window': stats(sc, M, win, sims, rank)}
        if uni is not None: res['sets']['unit'] = stats(sc, M, uni, sims, rank)
        out[c] = res
        print(c, {s: (round(v['seeds'], 3), round(v['kappa_seeds'], 3), round(v['mean_candidates'], 1)) for s, v in res['sets'].items() if v}, round(res['mirt_learner_rank_corr'], 3), flush=True)
        json.dump(out, open(f'{OUT}/decisions_r22.json', 'w'), indent=1)


if __name__ == '__main__': main()
