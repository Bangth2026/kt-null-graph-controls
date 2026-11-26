"""Release 6.0.0: simulated worlds with a known prerequisite graph, and closed-loop evaluation of next-KC policies.

World. A prerequisite-gated BKT learner with forgetting on the KCs of a real corpus:
  - true graph G*: for `sim_dbe`, the instructor relations of DBE-KT22 (undirected in the release) oriented from the
    KC that the training learners of the corpus meet earlier to the one they meet later; for `sim_a09`, the support
    graph of all training learners of ASSISTments 2009 (G100), re-oriented the same way. Both are acyclic.
  - KC k has guess g_k ~ U(0.15, 0.35), slip s_k ~ U(0.05, 0.15), learning rate l_k ~ U(0.10, 0.30) and an initial
    mastery probability pi_k set so that the expected first-attempt success pi_k (1 - s_k) + (1 - pi_k) g_k equals
    the corpus's first-attempt proportion correct on k (clipped to [0.02, 0.95]); a KC with prerequisites starts
    mastered only if its draw says so and at least half of its prerequisites start mastered.
  - Practising an unmastered KC k masters it with probability l_k * a_u * (RHO + (1 - RHO) * f), where f is the share
    of the prerequisites of k that are mastered (f = 1 without prerequisites; RHO = 0.05; learner speed
    a_u = exp(N(0, 0.5^2))). Every step, each
    mastered KC that is not practised is forgotten with probability F = 0.002.
  - Answers: correct with probability 1 - s_k if mastered, g_k otherwise.
Logs. 3,000 learners, 200 steps each, under a curriculum walk: each learner follows the curriculum order perturbed by
  N(0, 3^2) on the rank, stays on a KC until three correct answers in a row or six attempts, and with probability 0.1
  reviews a random KC already passed. Written as data/<world>.parquet (user, kc, kc_name, correct, order), the graph
  as data/<world>_true.json, the parameters as data/<world>_world.npz. The corpora then go through r20_controls.py
  like any other corpus (hash split 70/10/20, subsamples, update budget).
Closed loop. 500 fresh learners (seed 777), 20 warm-up steps of the curriculum walk, then H = 40 steps in which a
  policy chooses the KC. A tracer re-scores every KC after every answer (score_all); policies:
     target      KC whose predicted P(correct) is closest to TARGET = 0.65 (as Section 4.5)
     stable      Algorithm 4 with eps = 0.02 and the curriculum order of the training learners
     oracle      the KC whose TRUE P(correct) is closest to 0.65 (reference)
     frontier    the KC with the highest predicted P(correct) among those predicted below MASTERED = 0.8, i.e. the
                 KC the tracer rates most ready among those it does not rate as mastered (fixed after the target
                 rule was seen to choose mastered KCs in these worlds; see the article, Section 6.8)
     oracle_frontier  the unmastered KC with the largest TRUE probability of being learned (reference)
     curriculum  the curriculum walk continued (reference);   random  uniform over KCs (reference)
  The learners' random numbers are common to all policies (one generator per learner and purpose), so policies are
  compared on the same learners and the same draws until their choices differ.
  Outcomes per learner: gain (KCs mastered after H steps minus before), time to five new masteries (H + 1 if never),
  frustration (share of the H answers that are the third or later failure in a row), regret of the target rule under
  the true model, success rate; and, for two training seeds of one tracer, the share of learners whose first
  adaptive choice agrees and the mean Jaccard overlap of the KCs practised.
Usage: python r20_sim.py world <sim_dbe|sim_a09>;  python r20_sim.py loop <world> [--runs DIR]
"""
import sys, os, json, glob, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); DATA = f'{ROOT}/data'; OUTR = f'{ROOT}/results_r20'
os.makedirs(OUTR, exist_ok=True)
RHO = 0.05; FORGET = 0.002; TARGET = 0.65; MASTERED = 0.8; H = 40; WARM = 20; N_LOG = 3000; T_LOG = 200; N_TEST = 500


def rank_of(df):
    from r14_controls import first_encounter_rank
    from r5_controls import split_of
    d = df.copy(); d['split'] = d.user.map(split_of); K = int(d.kc.max()) + 1
    d = d.groupby('user', sort=False).head(200)
    return first_encounter_rank(d[d.split == 'train'], K), K


def first_rate(df, K):
    d = df.groupby('user', sort=False).head(200)
    f = d.drop_duplicates(['user', 'kc'])
    r = f.groupby('kc').correct.mean().reindex(range(K)).fillna(f.correct.mean()).values
    return r


def build_world(name, seed=20):
    rng = np.random.default_rng(seed)
    if name == 'sim_dbe':
        df = pd.read_parquet(f'{DATA}/dbe_kt22.parquet'); rank, K = rank_of(df)
        names = [None] * K
        for k, n in df[['kc', 'kc_name']].drop_duplicates('kc').itertuples(index=False): names[k] = n
        idx = {n: i for i, n in enumerate(names)}; S = np.zeros((K, K), bool)
        for a, b in json.load(open(f'{DATA}/dbe_kt22_expert.json'))['pairs']:
            if a in idx and b in idx and a != b: S[idx[a], idx[b]] = S[idx[b], idx[a]] = True
    elif name == 'sim_a09':
        df = pd.read_parquet(f'{DATA}/assist09.parquet'); rank, K = rank_of(df)
        names = [None] * K
        for k, n in df[['kc', 'kc_name']].drop_duplicates('kc').itertuples(index=False): names[k] = n
        G = np.load(f'{ROOT}/graphs_r19/assist09__g100.npz')['A'][:K, :K] > 0; S = G | G.T
    else:
        raise SystemExit(name)
    rank = np.where(np.isfinite(rank), rank, 2.0)
    A = np.zeros((K, K), np.float32)
    for i, j in zip(*np.nonzero(np.triu(S, 1))):
        if (rank[i], i) <= (rank[j], j): A[i, j] = 1
        else: A[j, i] = 1
    order = np.lexsort((np.arange(K), rank))                        # curriculum order (topological for A)
    fr = first_rate(df, K)
    g = rng.uniform(0.15, 0.35, K); s = rng.uniform(0.05, 0.15, K); l = rng.uniform(0.10, 0.30, K)
    pi = np.clip((fr - g) / np.maximum(1 - s - g, 1e-3), 0.02, 0.95)
    np.savez_compressed(f'{DATA}/{name}_world.npz', A=A, g=g, s=s, l=l, pi=pi, order=order, rank=rank)
    names = [f'k{k}' if n is None else str(n) for k, n in enumerate(names)]
    json.dump({'note': f'true prerequisite graph of the simulated world {name} (r20_sim.py)',
               'pairs': [(names[i], names[j]) for i, j in zip(*np.nonzero(A))]}, open(f'{DATA}/{name}_true.json', 'w'))
    W = dict(A=A, g=g, s=s, l=l, pi=pi, order=order, rank=rank, names=names, K=K)
    logs(name, W); return W


def load_world(name):
    z = np.load(f'{DATA}/{name}_world.npz'); W = {k: z[k] for k in z.files}; W['K'] = len(W['g'])
    W['names'] = [p for p in pd.read_parquet(f'{DATA}/{name}.parquet').drop_duplicates('kc').sort_values('kc').kc_name]
    return W


class Learners:
    """Vectorised state of n simulated learners with per-learner generators (common random numbers)."""
    def __init__(s, W, n, seed):
        s.W = W; K = W['K']; s.n = n; ss = np.random.SeedSequence(seed)
        kids = ss.spawn(n)
        s.rng_init = [np.random.default_rng(k.spawn(1)[0]) for k in kids]
        s.r_ans = [np.random.default_rng(k.spawn(2)[1]) for k in kids]
        s.r_learn = [np.random.default_rng(k.spawn(3)[2]) for k in kids]
        s.r_forget = [np.random.default_rng(k.spawn(4)[3]) for k in kids]
        s.r_walk = [np.random.default_rng(k.spawn(5)[4]) for k in kids]
        A = W['A'] > 0; s.pre = [np.flatnonzero(A[:, k]) for k in range(K)]
        m = np.zeros((n, K), bool); s.speed = np.zeros(n)
        for u in range(n):
            r = s.rng_init[u]; s.speed[u] = np.exp(r.normal(0, 0.5)); draw = r.random(K) < W['pi']
            for k in W['order']:
                m[u, k] = draw[k] and (len(s.pre[k]) == 0 or m[u, s.pre[k]].mean() >= 0.5)
        s.m = m
        # curriculum walk state
        s.lorder = []
        for u in range(n):
            noisy = W['rank'] * (K - 1) + s.r_walk[u].normal(0, 3, K)
            s.lorder.append(np.argsort(noisy, kind='stable'))
        s.ptr = np.zeros(n, int); s.att = np.zeros(n, int); s.streak = np.zeros(n, int)

    def p_true(s):
        W = s.W; return np.where(s.m, 1 - W['s'], W['g'])

    def walk_choice(s):
        ch = np.zeros(s.n, int)
        for u in range(s.n):
            r = s.r_walk[u]; p = min(s.ptr[u], s.W['K'] - 1)
            if p > 0 and r.random() < 0.1: ch[u] = s.lorder[u][r.integers(0, p)]
            else: ch[u] = s.lorder[u][p]
        return ch

    def step(s, ch, walk=False):
        W = s.W; y = np.zeros(s.n, int)
        for u in range(s.n):
            k = ch[u]; mu = s.m[u]
            y[u] = int(s.r_ans[u].random() < ((1 - W['s'][k]) if mu[k] else W['g'][k]))
            fr = s.r_forget[u].random(W['K']) < FORGET; fr[k] = False; mu[fr] = False
            if not mu[k]:
                f = mu[s.pre[k]].mean() if len(s.pre[k]) else 1.0; pl = W['l'][k] * s.speed[u] * (RHO + (1 - RHO) * f)
                if s.r_learn[u].random() < min(pl, 1.0): mu[k] = True
            if walk:
                cur = s.lorder[u][min(s.ptr[u], W['K'] - 1)]
                if k == cur:
                    s.att[u] += 1; s.streak[u] = s.streak[u] + 1 if y[u] else 0
                    if s.streak[u] >= 3 or s.att[u] >= 6: s.ptr[u] += 1; s.att[u] = 0; s.streak[u] = 0
        return y


def logs(name, W):
    L = Learners(W, N_LOG, seed=4242); rows = []
    Q = np.zeros((N_LOG, T_LOG), int); Y = np.zeros((N_LOG, T_LOG), int)
    for t in range(T_LOG):
        ch = L.walk_choice(); Q[:, t] = ch; Y[:, t] = L.step(ch, walk=True)
    names = W['names']
    df = pd.DataFrame({'user': np.repeat([f'u{u:05d}' for u in range(N_LOG)], T_LOG), 'kc': Q.ravel(),
                       'kc_name': [names[k] for k in Q.ravel()], 'correct': Y.ravel(), 'order': np.tile(np.arange(T_LOG), N_LOG)})
    # KC index must cover every KC: keep the world's index (names in KC order)
    df.to_parquet(f'{DATA}/{name}.parquet')
    print(name, 'K', W['K'], 'edges', int(W['A'].sum()), 'correct', round(df.correct.mean(), 3), 'KCs seen', df.kc.nunique(),
          'mastered at end', round(float(L.m.mean()), 3))


# ---------------------------------------------------------------------------------------------- scoring all KCs
def score_all(m, q, r):
    """P(correct) at every KC as the next item, after the histories q, r [B, t] (all positions valid)."""
    from r5_controls import FKGKT3
    from r20_models import NCAKT, GAMMA, TAU
    m.eval()
    with torch.no_grad():
        q = torch.as_tensor(q).clamp(max=m.K); r = torch.as_tensor(r); B, T = q.shape   # KCs absent from the logs -> padding
        if isinstance(m, NCAKT):
            K = m.K; E = m.E.weight[:K]; s0 = m.shift(); D = m.static_delta(E)
            a_in = torch.sigmoid(m.gin(torch.stack([m.lognk, D.norm(dim=1)], 1))).squeeze(1) if m.use_gate else torch.ones(K)
            Zin = m.ln(E + s0 + a_in.unsqueeze(1) * D)
            x = Zin[q] + m.R(r); h, _ = m.rnn(x); hl = h[:, -1]                      # [B, d]
            P = torch.sigmoid(m.mast(hl))                                            # [B, K]
            if m.dynamic:
                g = torch.sigmoid(GAMMA * (P - TAU)); outp = torch.tanh(m.Wout(m.Aout @ E)); msgs = []
                for j in range(m.Ain.size(0)):
                    Wj = m.Ain[j].unsqueeze(0) * g.unsqueeze(1)                       # [B, K(target), K(prereq)]
                    msgs.append(torch.tanh(m.Win(Wj @ E)) + outp[j].unsqueeze(0))
                Dt = msgs[0] - torch.stack(msgs[1:]).mean(0) if m.calibrate else msgs[0] - s0
            else:
                Dt = D.unsqueeze(0).expand(B, -1, -1)
            if m.use_gate:
                pt = P.clamp(1e-4, 1 - 1e-4); Hn = -(pt * pt.log() + (1 - pt) * (1 - pt).log())
                feat = torch.stack([m.lognk.unsqueeze(0).expand(B, -1), Hn, Dt.norm(dim=2)], 2)
                al = torch.sigmoid(m.gt(feat)).squeeze(2)
            else:
                al = torch.ones(B, K)
            zn = m.ln(E.unsqueeze(0) + s0 + al.unsqueeze(2) * Dt)                     # [B, K, d]
            lg = torch.einsum('bkd,btd->bkt', m.Wq(zn), m.Wk(x)) / m.d ** 0.5
            lg = lg + m.bsame * (torch.arange(K).view(1, K, 1) == q.unsqueeze(1)).float()
            a = torch.softmax(lg, -1); c = torch.einsum('bkt,btd->bkd', a, m.Wv(x))
            hh = hl.unsqueeze(1).expand(-1, K, -1)
            return torch.sigmoid(m.out(torch.cat([hh, c, zn, hh * zn], 2)).squeeze(2)).numpy()
        if isinstance(m, FKGKT3):
            b = m.base; K = m.K; Z = b.kc_repr(); x = Z[q] + b.R(r); h, _ = b.rnn(x); hl = h[:, -1]
            zn = Z[:K]; lg = torch.einsum('kd,btd->bkt', m.Wq(zn), m.Wk(x)) / m.d ** 0.5
            kk = torch.arange(K).view(1, K, 1)
            lg = lg + m.bsame * (kk == q.unsqueeze(1)).float()
            lg = lg + m.bpre * m.S[q.unsqueeze(1).expand(-1, K, -1), kk.expand(B, -1, T)] + m.bsuc * m.S[kk.expand(B, -1, T), q.unsqueeze(1).expand(-1, K, -1)]
            a = torch.softmax(lg, -1); c = torch.einsum('bkt,btd->bkd', a, m.Wv(x))
            hh = hl.unsqueeze(1).expand(-1, K, -1); z = zn.unsqueeze(0).expand(B, -1, -1)
            return torch.sigmoid(m.out(torch.cat([hh, c, z, hh * z], 2)).squeeze(2)).numpy()
    raise TypeError(type(m))


def check_scores(m, K, seed=0):
    """score_all must equal the model's own forward pass with that KC as the next item."""
    g = np.random.default_rng(seed); q = torch.tensor(g.integers(0, K, (3, 12))); r = torch.tensor(g.integers(0, 2, (3, 12)))
    S = score_all(m, q, r); worst = 0.0
    for k in [0, K // 2, K - 1]:
        qn = torch.cat([q[:, 1:], torch.full((3, 1), k)], 1)
        with torch.no_grad(): p = torch.sigmoid(m(q, r, qn))[:, -1].numpy()
        worst = max(worst, float(np.abs(p - S[:, k]).max()))
    return worst


def choose(policy, S, L, W, rank_tr, rng):
    K = W['K']
    if policy == 'random': return np.array([r.integers(0, K) for r in rng])
    if policy == 'curriculum': return L.walk_choice()
    if policy == 'oracle': return np.abs(L.p_true() - TARGET).argmin(1)
    if policy == 'oracle_frontier':
        # the unmastered KC with the largest true probability of being learned when practised
        A = W['A'] > 0; npre = A.sum(0); f = np.where(npre > 0, (L.m.astype(float) @ A) / np.maximum(npre, 1), 1.0)
        pl = W['l'][None, :] * (RHO + (1 - RHO) * f); pl = np.where(L.m, -1.0, pl); return pl.argmax(1)
    if policy == 'frontier':
        # the KC most likely to be answered correctly among those the tracer does not yet rate as mastered
        X = np.where(S < MASTERED, S, -np.inf); ch = X.argmax(1); none = ~np.isfinite(X).any(1)
        ch[none] = S[none].argmin(1); return ch
    d = np.abs(S - TARGET)
    if policy == 'target': return d.argmin(1)
    if policy == 'stable':
        best = d.min(1, keepdims=True); cand = d <= best + 0.02
        key = np.where(cand, rank_tr[None, :S.shape[1]], np.inf); return key.argmin(1)
    raise ValueError(policy)


def run_loop(W, policy, m=None, rank_tr=None, seed=777):
    L = Learners(W, N_TEST, seed=seed); K = W['K']
    hq = np.zeros((N_TEST, WARM + H), int); hr = np.zeros((N_TEST, WARM + H), int)
    for t in range(WARM):
        ch = L.walk_choice(); hq[:, t] = ch; hr[:, t] = L.step(ch, walk=True)
    m0 = L.m.sum(1).copy(); rng = [np.random.default_rng(seed * 1000 + u) for u in range(N_TEST)]
    ttm = np.full(N_TEST, H + 1); fails = np.zeros(N_TEST, int); frus = np.zeros(N_TEST); succ = np.zeros(N_TEST)
    reg = np.zeros(N_TEST); chosen = np.zeros((N_TEST, H), int)
    for t in range(H):
        S = (np.concatenate([score_all(m, hq[i:i + 100, :WARM + t], hr[i:i + 100, :WARM + t]) for i in range(0, N_TEST, 100)])
             if m is not None else None)
        ch = choose(policy, S, L, W, rank_tr, rng); pt = L.p_true()
        reg += np.abs(pt[np.arange(N_TEST), ch] - TARGET) - np.abs(pt - TARGET).min(1)
        y = L.step(ch, walk=(policy == 'curriculum')); hq[:, WARM + t] = ch; hr[:, WARM + t] = y; chosen[:, t] = ch
        fails = np.where(y == 0, fails + 1, 0); frus += (fails >= 3); succ += y
        new = L.m.sum(1) - m0; ttm = np.where((ttm > H) & (new >= 5), t + 1, ttm)
    gain = L.m.sum(1) - m0
    return dict(gain=float(gain.mean()), ttm5=float(ttm.mean()), reached5=float((ttm <= H).mean()), frustration=float((frus / H).mean()),
                success=float((succ / H).mean()), regret=float((reg / H).mean()),
                per_learner=dict(gain=gain.tolist(), ttm5=ttm.tolist(), frus=(frus / H).tolist(), regret=(reg / H).tolist())), chosen


def loop(world, runs):
    W = load_world(world); K = W['K']
    from r5_controls import load
    df = load(world, 1.0, 0); rank_tr = np.load(f'{DATA}/{world}_world.npz')['rank']
    out = {'world': world, 'K': K, 'edges': int(W['A'].sum()), 'H': H, 'warm': WARM, 'n': N_TEST, 'reference': {}, 'models': {}}
    for p in ('oracle', 'oracle_frontier', 'curriculum', 'random'):
        res, _ = run_loop(W, p); out['reference'][p] = res; print(world, p, {k: round(v, 4) for k, v in res.items() if k != 'per_learner'}, flush=True)
    for f in sorted(glob.glob(f'{runs}/{world}__*.pt')):
        tag = os.path.basename(f)[:-3]; m = torch.load(f, weights_only=False)
        if type(m).__name__ == 'FKGKT3State': continue          # state-channel models: prediction only
        err = check_scores(m, m.K); js = json.load(open(f[:-3] + '.json'))
        rec = {'cond': js['cond'], 'frac': js['frac'], 'rep': js['rep'], 'test_auc': js['test_auc'], 'test_auc_first': js['test_auc_first'],
               'score_check_max_abs_diff': err}
        for p in ('target', 'stable', 'frontier'):
            res, ch = run_loop(W, p, m, rank_tr); rec[p] = res; rec[p + '_chosen'] = ch[:, :10].tolist()
        out['models'][tag] = rec
        print(tag, round(err, 7), {p: (round(rec[p]['gain'], 3), round(rec[p]['regret'], 4)) for p in ('target', 'stable', 'frontier')}, flush=True)
    json.dump(out, open(f'{OUTR}/closed_loop_{world}.json', 'w'))


if __name__ == '__main__':
    if sys.argv[1] == 'world': build_world(sys.argv[2])
    elif sys.argv[1] == 'loop':
        runs = sys.argv[sys.argv.index('--runs') + 1] if '--runs' in sys.argv else f'{ROOT}/runs_r20_sim'
        loop(sys.argv[2], runs)
