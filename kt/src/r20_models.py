"""Release 6.0.0: models added for the second UMUAI version.

NCAKT   Null-calibrated adaptive graph knowledge tracing. The probe of Eq. (3)-(4) (FKGKT3, graph in the
        embedding refinement only, attention bias off as in `support_emb`) with three changes, each of which can
        be switched off for an ablation:
        (a) null calibration: the message a KC receives from its graph neighbours is taken relative to the mean
            message it would receive over M degree-preserving rewirings of the same graph,
                Delta_k = m_k(A) - mean_m m_k(A_null^(m));
        (b) mastery-conditioned prerequisite edges at the target KC: the prerequisite part of the message to the
            target KC c_{t+1} weights each prerequisite a by g_{t,a} = sigmoid(gamma (p_{t,a} - tau)), where p_t is
            a per-KC mastery estimate read from the knowledge state h_t by a linear head (trained, like DKT's
            output, to predict the next answer); the same weights are applied to the null graphs;
        (c) a learned gate alpha in [0, 1] on Delta, from log(1 + n_k) (training interactions on KC k), the
            entropy of p_{t,k} and the norm of Delta, so that
                z_{t,k} = LN(E_k + s0 + alpha_{t,k} Delta_{t,k}),
            where s0 = tanh(b_in) + tanh(b_out) is the shift that the empty-graph control adds to every KC.
            With alpha = 0 the model is the empty-graph control exactly.
        Training adds  lambda_1 * relu(margin - (l_null - l_G)) on the first attempts of the batch, where l_G is
        the cross-entropy of the model and l_null (no gradient) that of the same network with alpha = 0 at the
        target, and lambda_2 * mean(alpha) at the target (an L1 penalty, alpha >= 0), plus the cross-entropy of
        the mastery head with weight lambda_p.
        Constants fixed before any run and not tuned: M = 3, gamma = 10, tau = 0.6, margin = 0.005,
        lambda_1 = 0.5, lambda_2 = 0.01, lambda_p = 0.5.
QProbe  The probe at the item (question) level. Every interaction and target is an item i with KC kc(i). The
        representation of an item is
                z_i = LN(s0 + [Q_i] + [E_kc(i)] + [m_kc(i)(A)])
        where the bracketed terms are switched on by the condition: Q an item table, E the KC table read through the
        item-to-KC map (the Q-matrix; one KC per item), m the embedding refinement of Eq. (3) over a KC graph A.
        An optional item difficulty b_i is added to the output logit.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from models import rownorm

M_NULL = 3; GAMMA = 10.0; TAU = 0.6; MARGIN = 0.005; LAM1 = 0.5; LAM2 = 0.01; LAMP = 0.5


def _pad(A, K):
    A = torch.tensor(np.asarray(A), dtype=torch.float32).clone(); A.fill_diagonal_(0)
    return A


class NCAKT(nn.Module):
    def __init__(s, K, A, nulls, n_k, d=64, drop=0.2, dynamic=True, calibrate=True, gate=True, margin=True):
        super().__init__(); s.K = K; s.d = d
        s.dynamic, s.calibrate, s.use_gate, s.use_margin = dynamic, calibrate, gate, margin
        s.E = nn.Embedding(K + 1, d, padding_idx=K); s.R = nn.Embedding(2, d)
        s.Win = nn.Linear(d, d); s.Wout = nn.Linear(d, d); s.ln = nn.LayerNorm(d)
        s.rnn = nn.GRU(d, d, batch_first=True); s.dp = nn.Dropout(drop)
        s.Wq = nn.Linear(d, d); s.Wk = nn.Linear(d, d); s.Wv = nn.Linear(d, d); s.bsame = nn.Parameter(torch.tensor(1.0))
        s.out = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, 1))
        s.mast = nn.Linear(d, K)                                   # mastery head p_t = sigmoid(mast(h_t))
        s.gin = nn.Linear(2, 1); s.gt = nn.Linear(3, 1)            # gates (input side static, target side dynamic)
        for g in (s.gin, s.gt): nn.init.zeros_(g.weight); nn.init.zeros_(g.bias)
        G = [_pad(A, K)] + [_pad(x, K) for x in nulls]
        s.register_buffer('Ain', torch.stack([rownorm(x.T) for x in G]))     # [1+M, K, K]
        s.register_buffer('Aout', torch.stack([rownorm(x) for x in G]))
        s.register_buffer('lognk', torch.log1p(torch.tensor(np.asarray(n_k), dtype=torch.float32)) / 10.0)
        s.last_alpha = None

    def shift(s):
        return torch.tanh(s.Win.bias) + torch.tanh(s.Wout.bias)

    def messages(s, E):
        """Static messages for every graph: [1+M, K, d]."""
        return torch.tanh(s.Win(s.Ain @ E)) + torch.tanh(s.Wout(s.Aout @ E))

    def static_delta(s, E):
        Mg = s.messages(E)
        if s.calibrate: return Mg[0] - Mg[1:].mean(0)
        return Mg[0] - s.shift()

    def forward(s, q, r, qn, return_null=False):
        K = s.K; E = s.E.weight[:K]; s0 = s.shift()
        D = s.static_delta(E)                                                   # [K, d]
        a_in = torch.sigmoid(s.gin(torch.stack([s.lognk, D.norm(dim=1)], 1))).squeeze(1) if s.use_gate else torch.ones(K)
        Zin = s.ln(E + s0 + a_in.unsqueeze(1) * D)
        Zin = torch.cat([Zin, torch.zeros(1, s.d)], 0)
        qc = q.clamp(max=K); qnc = qn.clamp(max=K)
        x = s.dp(Zin[qc] + s.R(r)); h, _ = s.rnn(x)
        P = torch.sigmoid(s.mast(h))                                            # [B, T, K] mastery estimate
        # target-side delta
        if s.dynamic:
            g = torch.sigmoid(GAMMA * (P - TAU))                                # [B, T, K]
            tk = qn.clamp(max=K - 1)
            outp = torch.tanh(s.Wout(s.Aout @ E))                               # [1+M, K, d] successor part (static)
            msgs = []
            for j in range(s.Ain.size(0)):
                W = s.Ain[j][tk] * g                                            # [B, T, K] prerequisites of the target, gated
                msgs.append(torch.tanh(s.Win(W @ E)) + outp[j][tk])
            mg = msgs[0]
            Dt = mg - torch.stack(msgs[1:]).mean(0) if s.calibrate else mg - s0
        else:
            Dt = D[qn.clamp(max=K - 1)]
        Et = E[qn.clamp(max=K - 1)]
        if s.use_gate:
            pt = P.gather(2, qn.clamp(max=K - 1).unsqueeze(2)).squeeze(2).clamp(1e-4, 1 - 1e-4)
            H = -(pt * pt.log() + (1 - pt) * (1 - pt).log())
            feat = torch.stack([s.lognk[qn.clamp(max=K - 1)], H, Dt.norm(dim=2)], 2)
            alpha = torch.sigmoid(s.gt(feat)).squeeze(2)
        else:
            alpha = torch.ones(qn.shape)
        s.last_alpha = alpha
        lo = s.readout(q, qc, qnc, x, h, s.ln(Et + s0 + alpha.unsqueeze(2) * Dt))
        s.last_mast = P
        if return_null:
            return lo, s.readout(q, qc, qnc, x, h, s.ln(Et + s0))
        return lo

    def readout(s, q, qc, qnc, x, h, zn):
        T = q.size(1)
        logit = (s.Wq(zn) @ s.Wk(x).transpose(1, 2)) / s.d ** 0.5
        same = (qnc.unsqueeze(2) == qc.unsqueeze(1)).float()
        logit = logit + s.bsame * same
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) | (q >= s.K).unsqueeze(1)
        a = torch.softmax(logit.masked_fill(mask, -1e9), -1)
        c = s.dp(a @ s.Wv(x))
        return s.out(torch.cat([h, c, zn, h * zn], 2)).squeeze(2)

    def extra_loss(s, q, r, qn, y, lo, lo_null):
        K = s.K; mk = qn < K; L = torch.zeros(())
        P = s.last_mast
        pm = P.gather(2, qn.clamp(max=K - 1).unsqueeze(2)).squeeze(2).clamp(1e-5, 1 - 1e-5)
        L = L + LAMP * F.binary_cross_entropy(pm[mk], y[mk].float())
        if s.use_gate:
            L = L + LAM2 * s.last_alpha[mk].mean()
        if s.use_margin and lo_null is not None:
            T = q.size(1)
            seen = ((qn.unsqueeze(2) == q.unsqueeze(1)) & torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)).any(2)
            fm = mk & ~seen
            if fm.any():
                lg = F.binary_cross_entropy_with_logits(lo[fm], y[fm].float())
                ln = F.binary_cross_entropy_with_logits(lo_null[fm].detach(), y[fm].float())
                L = L + LAM1 * F.relu(MARGIN - (ln - lg))
        return L


class QProbe(nn.Module):
    """Item-level probe. q, qn are item sequences (padding = n_items); kc_of maps items to KCs."""
    def __init__(s, n_items, K, kc_of, A=None, use_item=True, use_kc=True, use_diff=False, d=64, drop=0.2):
        super().__init__(); s.N = n_items; s.K = K; s.d = d
        s.use_item, s.use_kc, s.use_diff = use_item, use_kc, use_diff
        s.register_buffer('kc_of', torch.cat([torch.tensor(np.asarray(kc_of), dtype=torch.long), torch.tensor([K])]))
        s.Q = nn.Embedding(n_items + 1, d, padding_idx=n_items); s.E = nn.Embedding(K + 1, d, padding_idx=K)
        s.R = nn.Embedding(2, d); s.Win = nn.Linear(d, d); s.Wout = nn.Linear(d, d); s.ln = nn.LayerNorm(d)
        A = np.zeros((K, K), np.float32) if A is None else A
        A = _pad(A, K); s.register_buffer('Ain', rownorm(A.T)); s.register_buffer('Aout', rownorm(A))
        s.rnn = nn.GRU(d, d, batch_first=True); s.dp = nn.Dropout(drop)
        s.Wq = nn.Linear(d, d); s.Wk = nn.Linear(d, d); s.Wv = nn.Linear(d, d); s.bsame = nn.Parameter(torch.tensor(1.0))
        s.out = nn.Sequential(nn.Linear(4 * d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, 1))
        s.b = nn.Embedding(n_items + 1, 1, padding_idx=n_items); nn.init.zeros_(s.b.weight)

    def item_repr(s):
        Ek = s.E.weight[:s.K]
        mk = torch.tanh(s.Win(s.Ain @ Ek)) + torch.tanh(s.Wout(s.Aout @ Ek))      # [K, d]; empty graph -> shift
        kc = s.kc_of[:s.N]
        Z = mk[kc]
        if s.use_kc: Z = Z + Ek[kc]
        if s.use_item: Z = Z + s.Q.weight[:s.N]
        Z = s.ln(Z)
        return torch.cat([Z, torch.zeros(1, s.d)], 0)

    def forward(s, q, r, qn):
        Z = s.item_repr(); qc = q.clamp(max=s.N); qnc = qn.clamp(max=s.N)
        x = s.dp(Z[qc] + s.R(r)); h, _ = s.rnn(x); zn = Z[qnc]; T = q.size(1)
        logit = (s.Wq(zn) @ s.Wk(x).transpose(1, 2)) / s.d ** 0.5
        same = (s.kc_of[qnc].unsqueeze(2) == s.kc_of[qc].unsqueeze(1)).float()
        logit = logit + s.bsame * same
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1) | (q >= s.N).unsqueeze(1)
        a = torch.softmax(logit.masked_fill(mask, -1e9), -1)
        c = s.dp(a @ s.Wv(x))
        lo = s.out(torch.cat([h, c, zn, h * zn], 2)).squeeze(2)
        if s.use_diff: lo = lo + s.b(qnc).squeeze(2)
        return lo
