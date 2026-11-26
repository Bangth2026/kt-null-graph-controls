"""Release 5.0.0: next-KC decisions for the revision for UMUAI (reviewer comments M5 and M8.1). No new training runs
except the three 5% empty-graph probes per corpus dumped by r19_controls.py (runs_r19, --dump).

Decision models (all on subsample 0 of the candidate files unless stated, same test learners and positions):
  5%   empty-graph probe, seeds 0, 100, 200 (runs_r19)
  25%  empty, support (seeds 0, 100, 200), random code, support in the embedding refinement (s0); empty, random code,
       support, support_emb on subsamples 1 and 2 (runs_r13, runs_r14)
  100% empty, support, random code, seeds 0, 100, 200 (runs_r13, runs_r14)
so first-attempt AUC spans the range from 5% to 100% of the training learners on every corpus.
Simulated learners:
  rasch  P = sigmoid(theta_learner - b_KC), penalised ML on every observed answer of the test learners (as r15_regret.py)
  pfa    a learner who learns: P = sigmoid(b_KC + w1 log(1 + s) + w2 log(1 + f) + w3 (s + 1)/(s + f + 2)), with s and f the
         learner's successes and failures so far on all KCs (performance-factors style, shared weights), fitted by
         logistic regression (L2, C = 1) on every next-attempt answer of the test learners; at a decision point the
         simulator's P for an unattempted KC uses the learner's history up to that point, so it rises as the learner succeeds.
Rules (candidates: KCs the learner has not attempted by that position):
  target        the KC whose predicted P is closest to 0.65
  easiest       the KC with the largest predicted P
  target_tb(e)  stabilised target rule: among the KCs whose |P - 0.65| is within e of the model's best, the one that the
                training learners meet earliest (mean rank of first encounter at 25%, subsample 0): a deterministic
                tie-break by curriculum order (e = 0.01, 0.02)
Outputs (results_r19/decisions.json):
  1. regret of every decision model under both simulators and every rule;
  2. regret against first-attempt AUC: slope with corpus fixed effects over all decision models, its 95% interval, and the
     within-corpus SD and range of AUC;
  3. agreement of next-KC choices between training seeds of the control at 25% and between graph and control with seeds
     crossed, for every rule, and the number of near-ties;
  4. regret of the stabilised rule against the plain target rule.
"""
import os, sys, json, glob, warnings
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import load
from r14_controls import first_encounter_rank
from train import seqs
warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r19'; os.makedirs(OUT, exist_ok=True)
CORPORA = ['assist09', 'assist12', 'assist15', 'assist17', 'junyi', 'algebra05', 'eedi', 'ednet', 'bridge06', 'statics11']
DIRS = [f'{REPO}/runs_r13', f'{REPO}/runs_r14', f'{REPO}/runs_r19']
T = (10, 25, 50, 100, 150); TARGET = 0.65


def find(c, frac, sub, cond, rep):
    tag = f'{c}__f{frac}__s{sub}__{cond}{"__rep%d" % rep if rep else ""}__steps'
    for d in DIRS:
        if os.path.exists(f'{d}/{tag}_cand.npz') and os.path.exists(f'{d}/{tag}.json'):
            return f'{d}/{tag}'
    return None


def models_of(c):
    spec = [(5, 0, 'empty', r) for r in (0, 1, 2)]
    spec += [(25, 0, m, r) for m in ('empty', 'support') for r in (0, 1, 2)] + [(25, 0, 'rvec_emb', 0), (25, 0, 'support_emb', 0)]
    spec += [(25, s, m, 0) for s in (1, 2) for m in ('empty', 'rvec_emb', 'support', 'support_emb')]
    spec += [(100, 0, m, r) for m in ('empty', 'support', 'rvec_emb') for r in (0, 1, 2)]
    out = {}
    for frac, s, m, r in spec:
        p = find(c, frac, s, m, r)
        if p: out[(frac, s, m, r)] = p
    return out


def rasch(c, n, K):
    P = np.load(glob.glob(f'{REPO}/runs_r13/{c}__f100__s0__empty__steps_pred.npz')[0])
    u = P['uid'].astype(int); k = P['kc'].astype(int); y = P['y'].astype(float); N = int(u.max()) + 1; K = max(K, int(k.max()) + 1)
    lam = 1.0
    def f(w):
        th, b = w[:N], w[N:]; z = th[u] - b[k]; p = 1 / (1 + np.exp(-z))
        ll = np.sum(y * z - np.logaddexp(0, z)); g = y - p
        gth = np.bincount(u, g, N) - lam * th; gb = -np.bincount(k, g, K) - lam * b
        return -(ll - 0.5 * lam * (th @ th + b @ b)), -np.concatenate([gth, gb])
    w = minimize(f, np.zeros(N + K), jac=True, method='L-BFGS-B').x
    th, b = w[:N], w[N:]
    S = 1 / (1 + np.exp(-(th[:n, None] - b[None, :])))
    return np.repeat(S[:, None, :], len(T), axis=1)


def pfa(c, n, K):
    df = load(c, 1.0, 0); Kd = int(df.kc.max()) + 1; q, r, _ = seqs(df, Kd, 'test'); q = q.numpy(); r = r.numpy()
    Kx = max(K, Kd); rows = []; ys = []; ks = []
    for i in range(len(q)):
        s = f = 0
        for t in range(q.shape[1]):
            if q[i, t] >= Kd: break
            if t >= 1: rows.append((np.log1p(s), np.log1p(f), (s + 1) / (s + f + 2))); ys.append(r[i, t]); ks.append(q[i, t])
            if r[i, t]: s += 1
            else: f += 1
    X = np.array(rows, np.float32); ks = np.array(ks)
    oh = np.zeros((len(ks), Kx), np.float32); oh[np.arange(len(ks)), ks] = 1
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(np.hstack([oh, X]), np.array(ys))
    b = lr.coef_[0][:Kx] + lr.intercept_[0]; w = lr.coef_[0][Kx:]
    S = np.full((n, len(T), Kx), np.nan, np.float32)
    for i in range(min(n, len(q))):
        s = f = 0
        for t in range(q.shape[1]):
            if q[i, t] >= Kd: break
            if r[i, t]: s += 1
            else: f += 1
            if t in T:
                li = T.index(t); z = b + w @ np.array([np.log1p(s), np.log1p(f), (s + 1) / (s + f + 2)])
                S[i, li] = 1 / (1 + np.exp(-z))
    return S, dict(w=[float(x) for x in w])


def choose(a, cnd, rule, rank):
    idx = np.flatnonzero(cnd); s = a[idx]
    if rule == 'target': return idx[np.argmin(np.abs(s - TARGET))]
    if rule == 'easiest': return idx[np.argmax(s)]
    e = float(rule.split('_tb')[1]); d = np.abs(s - TARGET); near = idx[d <= d.min() + e]
    return near[np.argmin(rank[near])]


RULES = ('target', 'easiest', 'target_tb0.01', 'target_tb0.02')


def decisions(score, seen, rank):
    ok = ~np.isnan(score[..., 0].astype(np.float32)); out = {}
    for i, t in zip(*np.nonzero(ok)):
        cnd = ~seen[i, t]
        if cnd.sum() < 2: continue
        a = score[i, t].astype(np.float32)
        out[(i, t)] = {rl: choose(a, cnd, rl, rank) for rl in RULES}
    return out


def regret(dec, sim, seen):
    res = {rl: [] for rl in RULES}
    for (i, t), ch in dec.items():
        cnd = ~seen[i, t]; s = sim[i, t].astype(np.float64)
        if np.isnan(s[cnd]).any(): continue
        dist = np.abs(s - TARGET)
        for rl, k in ch.items():
            res[rl].append(s[cnd].max() - s[k] if rl == 'easiest' else dist[k] - dist[cnd].min())
    return {rl: float(np.mean(v)) for rl, v in res.items()}


def near_ties(score, seen, e=0.01):
    ok = ~np.isnan(score[..., 0].astype(np.float32)); n = []
    for i, t in zip(*np.nonzero(ok)):
        cnd = ~seen[i, t]
        if cnd.sum() < 2: continue
        d = np.abs(score[i, t][cnd].astype(np.float32) - TARGET); n.append(int((d <= d.min() + e).sum()) - 1)
    return float(np.mean(n)), int(np.median([cnd.sum() for cnd in (~seen[ok]).reshape(-1, seen.shape[-1])]))


def main():
    out = {'per_corpus': {}, 'rows': []}
    for c in CORPORA:
        M = models_of(c); keys = list(M)
        ref = np.load(M[(25, 0, 'support', 0)] + '_cand.npz'); seen = ref['seen']; n, L, K = seen.shape
        sc = {}; auc = {}
        for k, p in M.items():
            z = np.load(p + '_cand.npz'); s = z['score'][..., :K]
            assert (z['seen'][..., :K] == seen).all(), (c, k)
            sc[k] = s; r = json.load(open(p + '.json')); auc[k] = (r['test_auc_first'], r['test_auc'])
        df25 = load(c, 0.25, 0); rank = first_encounter_rank(df25[df25.split == 'train'], max(K, int(df25.kc.max()) + 1))[:K]
        rank = np.where(np.isfinite(rank), rank, 10.0)
        sims = {'rasch': rasch(c, n, K)[..., :K]}
        S, info = pfa(c, n, K); sims['pfa'] = S[..., :K]
        D = {k: decisions(v, seen, rank) for k, v in sc.items()}
        for k in keys:
            for sl, S in sims.items():
                rg = regret(D[k], S, seen)
                out['rows'].append(dict(data=c, frac=k[0], sub=k[1], model=k[2], rep=k[3], sim=sl, auc_first=auc[k][0],
                                        auc_all=auc[k][1], **{f'regret_{rl}': v for rl, v in rg.items()}))
        # agreement at 25%, subsample 0
        def agree(a, b):
            common = set(D[a]) & set(D[b])
            return {rl: float(np.mean([D[a][x][rl] == D[b][x][rl] for x in common])) for rl in RULES}
        CC = [(25, 0, 'empty', r) for r in (0, 1, 2)]; GG = [(25, 0, 'support', r) for r in (0, 1, 2)]
        ag = {'control_seeds': pd.DataFrame([agree(CC[i], CC[j]) for i, j in ((0, 1), (0, 2), (1, 2))]).mean().to_dict(),
              'graph_control_crossed': pd.DataFrame([agree(GG[i], CC[j]) for i in range(3) for j in range(3) if i != j]).mean().to_dict(),
              'graph_control_shared': pd.DataFrame([agree(GG[i], CC[i]) for i in range(3)]).mean().to_dict()}
        nt, med = near_ties(sc[CC[0]], seen)
        out['per_corpus'][c] = dict(agreement=ag, near_ties_0p01=nt, median_candidates=med, pfa=info,
                                    n_models=len(keys), auc_first_range=[float(min(v[0] for v in auc.values())), float(max(v[0] for v in auc.values()))])
        print(c, len(keys), {k: {rl: round(v, 3) for rl, v in d.items()} for k, d in ag.items()}, flush=True)
    R = pd.DataFrame(out['rows'])
    import statsmodels.formula.api as smf
    reg = {}
    for sl in ('rasch', 'pfa'):
        for rl in RULES:
            d = R[R.sim == sl].rename(columns={f'regret_{rl}': 'y'})
            f = smf.ols('y ~ auc_first + C(data)', d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d['data'])[0]})
            lo, hi = f.conf_int().loc['auc_first']
            w = d.groupby('data').auc_first
            reg[f'{sl}|{rl}'] = dict(slope=float(f.params['auc_first']), lo=float(lo), hi=float(hi), p=float(f.pvalues['auc_first']),
                                     n=int(len(d)), auc_sd_within=float((d.auc_first - w.transform('mean')).std()),
                                     auc_range_within_median=float((w.max() - w.min()).median()),
                                     mean_regret=float(d.y.mean()))
    out['regret_vs_auc'] = reg
    # stabilised rule against the plain target rule, 25% control and graph models (seeds 0-2)
    sub = R[(R.frac == 25) & (R['sub'] == 0) & R.model.isin(['empty', 'support'])]
    out['stabilised_vs_plain'] = {sl: {rl: dict(mean=float((g[f'regret_{rl}'] - g['regret_target']).mean()),
                                               lower=int(((g.groupby('data')[f'regret_{rl}'].mean() - g.groupby('data')['regret_target'].mean()) < 0).sum()))
                                       for rl in ('target_tb0.01', 'target_tb0.02')} for sl, g in sub.groupby('sim')}
    out['agreement_mean'] = {k: pd.DataFrame([v['agreement'][k] for v in out['per_corpus'].values()]).mean().to_dict()
                             for k in ('control_seeds', 'graph_control_crossed', 'graph_control_shared')}
    json.dump(out, open(f'{OUT}/decisions.json', 'w'), indent=1, default=float)
    R.to_csv(f'{OUT}/decisions_rows.csv', index=False)
    print(json.dumps(out['regret_vs_auc'], indent=1)); print(json.dumps(out['agreement_mean'], indent=1)); print(json.dumps(out['stabilised_vs_plain'], indent=1))


if __name__ == '__main__': main()
