"""Release 5.0.0: runs added in the revision for User Modeling and User-Adapted Interaction (UMUAI).

Same data pipeline, subsamples, update budget (at most 2,000 updates, validation every 25, patience 8),
training loop and evaluation as r14_controls.py (CPU, one thread, NT=1). Every condition of r14_controls.py
(and so of r13_controls.py) is passed through to it unchanged and reproduces its archived run; this is how
the extra training seeds (--rep 1, 2 of `empty` and `support`) and the 500-interaction cap (KT_MAXLEN=500)
are run. This script adds two families of conditions.

1. Transfer of structure (graph built from more learners than the tracer sees). The support graph of Eq. (2)
   is built once from ALL training learners of the corpus (the 100% condition), on the same KC index, and
   given to a tracer trained on the subsample (5% or 10%). Nulls of that graph: DP100, a degree-preserving
   rewiring (double edge swaps, ten per edge, seed 1000 + s, as DP(G)), and ER100, an Erdos-Renyi graph with
   as many directed edges (seed 2000 + s, as ER(G)). Conditions:
     g100        both entry points (as `support`)
     g100_emb    embedding refinement only (as `support_emb`);  dp100_emb, er100_emb its nulls
     st_g100     state-routing channel only (as `st_support`);  st_dp100 its null
   The graph of the full training set is cached in graphs_r19/<data>__g100.npz.

2. A published graph-based tracer with a fourth entry point, the training loss. PDKT (Chen et al. 2018) turns
   prerequisite edges into a ranking constraint on DKT: for an edge a -> b, the predicted probability of a
   correct answer on b should not exceed that on a. We use the KC-level form
       L = BCE + lambda * mean_{t, (a,b) in E'} relu(sigma(y_t,b) - sigma(y_t,a)),
   with y_t the K output logits of DKT (models.DKT, d = 64) at step t, E' a uniform draw of at most 512 edges of
   the graph per update (all edges if fewer), and lambda = 1.0, fixed before any run (no tuning). Conditions:
     pdkt_empty  (plain DKT: no edge, no penalty), pdkt_er, pdkt_dp, pdkt_support, pdkt_uniform
   so the ladder of Eq. (1) is run inside PDKT with the graph entering only through the loss.
Output: runs_r19/<tag>.json, <tag>_pred.npz, as r14_controls.py.
"""
import sys, os, json, time, argparse, numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r14_controls as R14
from r5_controls import FKGKT3, load, transition_support, rewire, ece
from r6_controls import erdos_renyi
from r9_controls import neighbour_sets
from models import DKT
import r13_controls as R13
from train import seqs, batches
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt')
GR5 = f'{ROOT}/graphs_r5'; GR = f'{ROOT}/graphs_r19'; os.makedirs(GR, exist_ok=True)
NEW = ('g100', 'g100_emb', 'dp100_emb', 'er100_emb', 'st_g100', 'st_dp100',
       'pdkt_empty', 'pdkt_er', 'pdkt_dp', 'pdkt_support', 'pdkt_uniform')
LAMBDA = 1.0; MAX_EDGES = 512


def g100_of(data, K):
    p = f'{GR}/{data}__g100.npz'
    if os.path.exists(p):
        g = np.load(p); return g['A'][:K, :K], int(g['n_learners'])
    full = load(data, 1.0, 0); tr = full[full.split == 'train']; Kf = int(full.kc.max()) + 1
    A = (transition_support(tr, max(K, Kf)) > 0).astype(np.float32)
    np.savez_compressed(p, A=A, n_learners=tr.user.nunique()); return A[:K, :K], int(tr.user.nunique())


def support_of(df, a, K):
    tr = df[df.split == 'train']
    gpath = f'{GR5}/{a.data}__f{int(round(a.frac*100))}__s{a.sub}.npz'
    if os.path.exists(gpath):
        g = np.load(gpath); return g['support'], g['rewired']
    mu = transition_support(tr, K); sup = (mu > 0).astype(np.float32)
    rw, _ = rewire(sup, seed=1000 + a.sub); rw = rw.astype(np.float32)
    os.makedirs(GR5, exist_ok=True); np.savez_compressed(gpath, support=sup, rewired=rw); return sup, rw


class PDKT(DKT):
    """DKT with the prerequisite ranking penalty of Chen et al. (2018) over the edges of A."""
    def __init__(s, K, A, d=64, drop=0.2):
        super().__init__(K, d, drop); a, b = np.nonzero(A); s.ea = torch.tensor(a); s.eb = torch.tensor(b)

    def penalty(s, q, r, rng):
        if len(s.ea) == 0: return torch.zeros(())
        x = s.emb(torch.where(q < s.K, q + s.K * r, torch.full_like(q, 2 * s.K)))
        h, _ = s.rnn(x); P = torch.sigmoid(s.out(s.dp(h)))             # [B, T, K]
        idx = np.arange(len(s.ea)) if len(s.ea) <= MAX_EDGES else rng.choice(len(s.ea), MAX_EDGES, replace=False)
        ea, eb = s.ea[idx], s.eb[idx]; v = F.relu(P[..., eb] - P[..., ea])   # [B, T, E']
        mk = (q < s.K).float().unsqueeze(2)
        return (v * mk).sum() / (mk.sum() * len(idx)).clamp(min=1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--out', default=f'{ROOT}/runs_r19')
    a, rest = ap.parse_known_args(argv)
    if a.cond not in NEW:
        return R14.main(argv if '--out' in argv else argv + ['--out', a.out])
    seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '') + "__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    df = load(a.data, a.frac, a.sub); K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    sup, rw = support_of(df, a, K); er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    extra = {}
    if 'g100' in a.cond or 'dp100' in a.cond or 'er100' in a.cond:
        G100, n100 = g100_of(a.data, K)
        S = (sup + sup.T) > 0; B = (G100 + G100.T) > 0
        extra.update(edges_g100=int(G100.sum()), n_learners_g100=n100,
                     jaccard_G_G100_undirected=float((B & S).sum() / max(1, (B | S).sum())),
                     recall_G100_in_G_directed=float((G100 * sup).sum() / max(1, G100.sum())),
                     precision_G_in_G100_directed=float((G100 * sup).sum() / max(1, sup.sum())))
        if 'dp100' in a.cond:
            A, _ = rewire(G100, seed=1000 + a.sub); A = A.astype(np.float32)
        elif 'er100' in a.cond:
            A = erdos_renyi(K, int(G100.sum()), seed=2000 + a.sub)
        else:
            A = G100
    else:
        U = np.ones((K, K), np.float32); np.fill_diagonal(U, 0)
        A = {'pdkt_empty': np.zeros((K, K), np.float32), 'pdkt_er': er, 'pdkt_dp': rw, 'pdkt_support': sup,
             'pdkt_uniform': U}[a.cond]
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    pen = False
    if a.cond.startswith('pdkt_'):
        m = PDKT(K, A); family = 'pdkt'; pen = a.cond != 'pdkt_empty'
    elif a.cond.startswith('st_'):
        m = R13.FKGKT3State(K, A); family = 'probe_state'
    elif a.cond == 'g100':
        m = FKGKT3(K, A, recency=False); family = 'probe'
    else:
        m = FKGKT3(K, A, recency=False); family = 'probe'
        with torch.no_grad(): m.S.zero_()
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    bs = 128; ebs = 128; max_steps, eval_every, patience = 2000, 25, 8
    prng = np.random.default_rng(9000 + seed)
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, bs, True, rng):
            m.train(); lo = m(qa, ra, qn); mk = qn < K
            loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
            if pen: loss = loss + LAMBDA * m.penalty(qa, ra, prng)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); steps += 1
            if steps % eval_every == 0:
                o = R13.evaluate(m, vq, vr, K, ebs, {}); va = roc_auc_score(o['y'], o['p']); hist.append(va)
                if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
                else: bad += 1
                if bad >= patience: stop = True; break
            if steps >= max_steps: stop = True; break
    if state is not None: m.load_state_dict(state)
    o = R13.evaluate(m, sq, sr, K, ebs, {'sup': neighbour_sets(sup), 'er': neighbour_sets(er)})
    P, Y, NW = o['p'], o['y'], o['new']
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=a.rep, seed=seed, cond=a.cond, family=family, budget='steps',
               K=K, device='cpu', torch=torch.__version__, batch_size=bs, lr=a.lr, d=64, maxlen=200,
               n_params=int(sum(p.numel() for p in m.parameters())), n_train_learners=int(tr.user.nunique()),
               n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(np.asarray(A).sum()),
               pdkt_lambda=LAMBDA if family == 'pdkt' else None,
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NW], P[NW])),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NW.sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(steps >= max_steps and bad < patience), n_evals=len(hist), valid_hist=hist, **extra)
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NW,
                        kc=o['kc'].astype(np.int32), pos=o['pos'].astype(np.int16), uid=o['uid'].astype(np.int32))
    res['secs_total'] = time.time() - t0
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'], 4), 'steps', steps,
          's', round(res['secs_total']), flush=True)


if __name__ == '__main__': main()
