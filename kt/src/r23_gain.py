"""Release 9.0.0: a learning-gain base rule for the closed loop (reviewer comment R4 of the second round).

gain   the KC with the largest predicted increase in its own success after one practice step, computed by a one-step
       rollout of the tracer:  Delta_k = P_k * P'_k(1) + (1 - P_k) * P'_k(0) - P_k,  where P_k is the tracer's current
       prediction for KC k and P'_k(y) its prediction for k after the history is extended by an attempt at k with outcome y.
       The rollout is exact for the probe (FKGKT3): one GRU step from the last hidden state and one extra attention
       token; check_gain() compares it with score_all on the explicitly extended history.

Everything else is as r22_simdec.py (worlds, 24 tracers, gates none / true / dp / g100, learners, warm-up, horizon).
The results are merged into results_r23/decisions_sim_<world>.json together with the r22 policies, and the gate G100
is flagged as degenerate when no KC is ever open under it (the rule then falls back to all KCs at every step).
Usage: python r23_gain.py <world>
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_sim as S, r21_sim as X, r22_simdec as D
from r20_sim import Learners, score_all, TARGET, H, WARM, N_TEST
GAIN_GATES = ('none', 'true', 'dp')                     # G100 is reported for the other rules; it is degenerate in sim_dbe_mix
ROOT = S.ROOT; OUT = f'{ROOT}/results_r23'; os.makedirs(OUT, exist_ok=True)


def gain_scores(m, q, r):
    """[B, K] expected one-step increase of the predicted success on each KC if that KC is practised now."""
    m.eval()
    with torch.no_grad():
        q = torch.as_tensor(q).clamp(max=m.K); r = torch.as_tensor(r); B, T = q.shape; b = m.base; K = m.K
        Z = b.kc_repr(); x = Z[q] + b.R(r); h, hn = b.rnn(x)                         # hn [1, B, d]
        zn = Z[:K]; d = zn.size(1)
        kk = torch.arange(K).view(1, K, 1)
        lg = torch.einsum('kd,btd->bkt', m.Wq(zn), m.Wk(x)) / m.d ** 0.5
        lg = lg + m.bsame * (kk == q.unsqueeze(1)).float()
        lg = lg + m.bpre * m.S[q.unsqueeze(1).expand(-1, K, -1), kk.expand(B, -1, T)] + m.bsuc * m.S[kk.expand(B, -1, T), q.unsqueeze(1).expand(-1, K, -1)]
        V = m.Wv(x)                                                                  # [B, T, d]
        hl = h[:, -1]; hh = hl.unsqueeze(1).expand(-1, K, -1); z = zn.unsqueeze(0).expand(B, -1, -1)
        a0 = torch.softmax(lg, -1); P = torch.sigmoid(m.out(torch.cat([hh, torch.einsum('bkt,btd->bkd', a0, V), z, hh * z], 2)).squeeze(2))
        Sd = torch.diagonal(m.S[:K, :K])
        out = []
        for y in (0, 1):
            xn = zn + b.R(torch.full((K,), y, dtype=torch.long))                         # [K, d] new token for each candidate
            h1, _ = b.rnn(xn.unsqueeze(0).expand(B, -1, -1).reshape(B * K, 1, d), hn.repeat_interleave(K, dim=1))
            h1 = h1.reshape(B, K, d)
            ln = (m.Wq(zn) * m.Wk(xn)).sum(1) / m.d ** 0.5 + m.bsame + (m.bpre + m.bsuc) * Sd   # [K]
            L2 = torch.cat([lg, ln.view(1, K, 1).expand(B, -1, -1)], 2); a = torch.softmax(L2, -1)
            c = torch.einsum('bkt,btd->bkd', a[:, :, :T], V) + a[:, :, T:] * m.Wv(xn).unsqueeze(0)
            out.append(torch.sigmoid(m.out(torch.cat([h1, c, z, h1 * z], 2)).squeeze(2)))
        return (P * out[1] + (1 - P) * out[0] - P).numpy(), P.numpy()


def check_gain(m, K, seed=0):
    """The rollout must equal score_all on the explicitly extended history."""
    g = np.random.default_rng(seed); q = g.integers(0, K, (3, 10)); r = g.integers(0, 2, (3, 10)); worst = 0.0
    with torch.no_grad():
        dl, P = gain_scores(m, q, r)
        for k in (0, K // 2, K - 1):
            p1 = score_all(m, np.concatenate([q, np.full((3, 1), k)], 1), np.concatenate([r, np.ones((3, 1), int)], 1))[:, k]
            p0 = score_all(m, np.concatenate([q, np.full((3, 1), k)], 1), np.concatenate([r, np.zeros((3, 1), int)], 1))[:, k]
            worst = max(worst, float(np.abs(P[:, k] * p1 + (1 - P[:, k]) * p0 - P[:, k] - dl[:, k]).max()))
    return worst


def run_gain(W, gate, m, seed=777):
    """As r21_sim.run with base rule 'gain'."""
    L = Learners(W, N_TEST, seed=seed); K = W['K']
    hq = np.zeros((N_TEST, WARM + H), int); hr = np.zeros((N_TEST, WARM + H), int)
    for t in range(WARM):
        ch = L.walk_choice(); hq[:, t] = ch; hr[:, t] = L.step(ch, walk=True)
    m0 = L.m.sum(1).copy(); fails = np.zeros(N_TEST, int); frus = np.zeros(N_TEST); reg = np.zeros(N_TEST)
    opened = np.zeros(N_TEST); fallback = np.zeros(N_TEST); blocked_true = np.zeros(N_TEST); Atrue = W['A'] > 0
    for t in range(H):
        T = WARM + t
        Ok = X.open_set(gate, X.passed(hq, hr, T, K)); none = ~Ok.any(1); fallback += none; Ok[none] = True; opened += Ok.mean(1)
        Dl = np.concatenate([gain_scores(m, hq[i:i + 100, :T], hr[i:i + 100, :T])[0] for i in range(0, N_TEST, 100)])[:, :K]
        if Dl.shape[1] < K:
            Dl = np.concatenate([Dl, np.full((N_TEST, K - Dl.shape[1]), np.nan)], 1)
            Ok = Ok & ~np.isnan(Dl); Ok[~Ok.any(1)] = ~np.isnan(Dl[~Ok.any(1)])
        ch = np.where(Ok, np.nan_to_num(Dl, nan=-np.inf), -np.inf).argmax(1)
        blocked_true += ((Atrue[:, ch].T & ~L.m).any(1)).astype(float)
        pt = L.p_true(); reg += np.abs(pt[np.arange(N_TEST), ch] - TARGET) - np.abs(pt - TARGET).min(1)
        y = L.step(ch, walk=False); hq[:, T] = ch; hr[:, T] = y
        fails = np.where(y == 0, fails + 1, 0); frus += (fails >= 3)
    gain = L.m.sum(1) - m0
    return dict(gain=float(gain.mean()), frustration=float((frus / H).mean()), regret=float((reg / H).mean()),
                open_share=float((opened / H).mean()), fallback_share=float((fallback / H).mean()),
                blocked_by_true_state=float((blocked_true / H).mean()), per_learner=dict(gain=gain.tolist(), regret=(reg / H).tolist(), frus=(frus / H).tolist()))


def fallback_share(W, gate, seed=777):
    """Share of steps at which no KC is open under the gate, with the learners of the curriculum walk (a property of the
    gate graph, independent of the base rule)."""
    L = Learners(W, N_TEST, seed=seed); K = W['K']; hq = np.zeros((N_TEST, WARM + H), int); hr = np.zeros((N_TEST, WARM + H), int); n = 0
    for t in range(WARM + H):
        if t >= WARM: n += (~X.open_set(gate, X.passed(hq, hr, t, K)).any(1)).mean()
        ch = L.walk_choice(); hq[:, t] = ch; hr[:, t] = L.step(ch, walk=True)
    return float(n / H)


def main(world):
    W = S.load_world(world); K = W['K']; G = X.gate_graphs(world, W)
    out = json.load(open(os.environ.get('GAIN_SRC', f'{ROOT}/results_r22') + f'/decisions_sim_{world}.json'))
    g = G['g100'] > 0
    out['g100_structure'] = dict(edges=int(g.sum()), kcs_without_prerequisite=int((g.sum(0) == 0).sum()),
                                 fallback_share_curriculum=fallback_share(W, G['g100']),
                                 fallback_share_true_gate=fallback_share(W, G['true']))
    out['gain_check_max_abs_diff'] = 0.0
    for tag, rec in out['models'].items():
        f = None
        for d in D.RUNS:
            if os.path.exists(f'{d}/{tag}.pt'): f = f'{d}/{tag}.pt'
        m = torch.load(f, weights_only=False)
        out['gain_check_max_abs_diff'] = max(out['gain_check_max_abs_diff'], check_gain(m, m.K))
        for gname in GAIN_GATES:
            rec[f'gain|{gname}'] = run_gain(W, G[gname], m)
        print(world, tag, {gname: round(rec[f'gain|{gname}']['gain'], 2) for gname in GAIN_GATES}, flush=True)
        json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))
    json.dump(out, open(f'{OUT}/decisions_sim_{world}.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
