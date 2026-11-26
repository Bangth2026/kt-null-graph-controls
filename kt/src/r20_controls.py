"""Release 6.0.0: training runs of the second UMUAI version.

Same data, subsamples, update budget (at most 2,000 updates, validation every 25, patience 8), batch size, seed
handling and evaluation as r19_controls.py (CPU, one thread). Conditions of r19_controls.py (and so of r14 and r13)
are passed through unchanged. Added:

1. NCA-KT (r20_models.NCAKT), graph in the embedding refinement:
     nca           support graph G, all three components (null calibration, mastery-conditioned edges, gate)
     nca_dp        the same model given DP(G) in place of G (its nulls are rewirings of DP(G)): the edge term of NCA-KT
     nca_static    no mastery conditioning (g = 1)
     nca_nonull    no null calibration (Delta = m(G) - m(empty))
     nca_nogate    alpha = 1, no L1 penalty
     nca_nomargin  lambda_1 = 0
     nca_expert, nca_llm   (DBE-KT22) the instructor-authored graph, the generated graph;  nca_map (Junyi) its map
   Null graphs: M = 3 degree-preserving rewirings of the input graph, seeds 3000 + 10 s + m (m = 0, 1, 2), so none
   equals DP(G) (seed 1000 + s).
2. Authored and generated graphs (instructor relations oriented by curriculum order, see authored()) in the probe's embedding refinement (as `map_emb` of r14_controls.py):
     expert_emb, expert_dp_emb   DBE-KT22 instructor graph (data/dbe_kt22_expert.json) and its rewiring
     llm_emb, llm_dp_emb         DBE-KT22 graph generated from KC names and descriptions (data/dbe_kt22_llm.json)
     map_dp_emb                  Junyi: degree-preserving rewiring of the published map
3. The item-level probe (r20_models.QProbe), on corpora prepared by prep_r20.py (data/<name>_q.parquet):
     q_kc      KC table through the Q-matrix only (no item table): the KC-level probe written over items
     q_id      item table only (no Q-matrix)
     q_qm      item table + KC table through the Q-matrix
     q_rqm     item table + KC table through a size-preserving random Q-matrix (items permuted among KCs, so every
               KC keeps its number of items; seed 4000 + s)
     q_qm_g    q_qm + embedding refinement of the KC table over the support graph G
     q_qm_dp   q_qm + the same over DP(G)
     q_diff    KC table through the Q-matrix + an item difficulty b_i on the logit (no item table)
   First attempts are first attempts at a KC (the KC sequence of the same learner), as everywhere in the study;
   first attempts at an item are stored as well (`test_auc_first_item`).
4. empty_fa: the empty-graph probe trained with LAM1 (0.5) times the cross-entropy of the first attempts of each
   batch added to the loss, a graph-free control for the margin term of NCA-KT (which, when active, adds the same
   weight to the first-attempt loss).
Output: runs_r20/<tag>.json, <tag>_pred.npz.
"""
import sys, os, json, time, argparse, numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r19_controls as R19
import r14_controls as R14
from r5_controls import FKGKT3, load, rewire, ece, split_of, MAXLEN
from r6_controls import erdos_renyi
from r9_controls import neighbour_sets
import r13_controls as R13
from train import seqs, batches
from r20_models import NCAKT, QProbe, LAM1
torch.set_num_threads(int(os.environ.get('NT', '1')))
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); DATA = f'{ROOT}/data'
NCA = ('nca', 'nca_dp', 'nca_static', 'nca_nonull', 'nca_nogate', 'nca_nomargin', 'nca_expert', 'nca_llm', 'nca_map', 'nca_true')
GRAPHS = ('expert_emb', 'expert_dp_emb', 'llm_emb', 'llm_dp_emb', 'map_dp_emb', 'true_emb', 'true_dp_emb', 'st_true', 'st_true_dp')
SIMPROBE = ('empty', 'support_emb', 'er_emb', 'dp_emb')
FA = ('empty_fa',)   # graph-free control for the margin term: the empty-graph probe with LAM1 x cross-entropy on first attempts   # handled here (and the model saved) on simulated corpora
QC = ('q_kc', 'q_id', 'q_qm', 'q_rqm', 'q_qm_g', 'q_qm_dp', 'q_diff')


def named_graph(path, names):
    idx = {n: i for i, n in enumerate(names)}; K = len(names); A = np.zeros((K, K), np.float32)
    for a, b in json.load(open(path))['pairs']:
        if a in idx and b in idx and a != b: A[idx[a], idx[b]] = 1
    return A


def authored(a, df, K):
    names = R14.kc_names(df, K)
    if a.cond in ('expert_emb', 'expert_dp_emb', 'nca_expert'):
        # DBE-KT22 lists every instructor relation in both directions (101 of 101 pairs reciprocal), so the relation
        # is undirected; each pair is oriented from the KC the training learners meet earlier to the one they meet
        # later (r14_controls.first_encounter_rank), which gives a graph without reciprocal pairs, as G.
        S = named_graph(f'{DATA}/{a.data}_expert.json', names); S = np.maximum(S, S.T)
        rank = R14.first_encounter_rank(df[df.split == 'train'], K); A = np.zeros_like(S)
        for i, j in zip(*np.nonzero(np.triu(S, 1))):
            if rank[i] <= rank[j]: A[i, j] = 1
            else: A[j, i] = 1
        return A
    if a.cond in ('llm_emb', 'llm_dp_emb', 'nca_llm'): return named_graph(f'{DATA}/{a.data}_llm.json', names)
    if a.cond in ('map_dp_emb', 'nca_map'): return R14.junyi_map(names)
    if a.cond in ('true_emb', 'true_dp_emb', 'nca_true', 'st_true', 'st_true_dp'): return named_graph(f'{DATA}/{a.data}_true.json', names)
    raise ValueError(a.cond)


def load_q(name, frac, sub):
    """As r5_controls.load on data/<name>_q.parquet (same rows, same learners, same subsample)."""
    df = pd.read_parquet(f'{DATA}/{name}_q.parquet')
    df = df.groupby('user', sort=False).head(MAXLEN).reset_index(drop=True)
    df['split'] = df.user.map(split_of)
    if frac < 1.0:
        tr = np.array(sorted(df.loc[df.split == 'train', 'user'].unique()))
        keep = set(np.random.default_rng(12345 + sub).choice(tr, max(2, int(round(frac * len(tr)))), replace=False))
        df = df[(df.split != 'train') | df.user.isin(keep)].reset_index(drop=True)
    return df


def item_seqs(df, N, split, L=200):
    d = df[df.split == split]; Q = []
    for u, g in d.groupby('user', sort=False):
        q = g.q.values[:L]
        if len(q) < 2: continue
        Q.append(q)
    T = max(len(x) for x in Q); out = np.full((len(Q), T), N)
    for i, a in enumerate(Q): out[i, :len(a)] = a
    return torch.tensor(out)


def evaluate_q(m, iq, kq, r, N, K, bs=128):
    """Predictions of an item-level model; first attempts by KC (kq) and by item (iq)."""
    m.eval(); P = []; Y = []; NEWK = []; NEWI = []; KC = []; UID = []
    with torch.no_grad():
        for i0 in range(0, len(iq), bs):
            j = np.arange(i0, min(len(iq), i0 + bs))
            ia, inx = iq[j][:, :-1], iq[j][:, 1:]; ka, knx = kq[j][:, :-1], kq[j][:, 1:]; ra, y = r[j][:, :-1], r[j][:, 1:]
            p = torch.sigmoid(m(ia, ra, inx)); mask = (inx < N).numpy()
            P.append(p.numpy()[mask]); Y.append(y.numpy()[mask]); KC.append(knx.numpy()[mask]); UID.append(np.repeat(j, mask.sum(1)))
            T = inx.shape[1]
            tri = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)
            sk = ((knx.unsqueeze(2) == ka.unsqueeze(1)) & tri).any(2).numpy()
            si = ((inx.unsqueeze(2) == ia.unsqueeze(1)) & tri).any(2).numpy()
            NEWK.append(~sk[mask]); NEWI.append(~si[mask])
    return dict(p=np.concatenate(P), y=np.concatenate(Y), new=np.concatenate(NEWK), new_item=np.concatenate(NEWI),
                kc=np.concatenate(KC), uid=np.concatenate(UID))


def nulls_of(A, s, M=3):
    return [rewire(A, seed=3000 + 10 * s + j)[0].astype(np.float32) for j in range(M)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument('--data'); ap.add_argument('--frac', type=float, default=0.1); ap.add_argument('--sub', type=int, default=0)
    ap.add_argument('--cond'); ap.add_argument('--rep', type=int, default=0); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--out', default=f'{ROOT}/runs_r20')
    a, rest = ap.parse_known_args(argv)
    sim = str(a.data).startswith('sim_')
    if a.cond not in NCA + GRAPHS + QC + FA and not (sim and a.cond in SIMPROBE):
        return R19.main(argv if '--out' in argv else argv + ['--out', a.out])
    seed = a.sub + 100 * a.rep; os.makedirs(a.out, exist_ok=True)
    tag = f"{a.data}__f{int(round(a.frac*100))}__s{a.sub}__{a.cond}" + (f"__rep{a.rep}" if a.rep else '') + "__steps"
    if os.path.exists(f'{a.out}/{tag}.json'): print('skip', tag); return
    item = a.cond in QC
    df = load_q(a.data, a.frac, a.sub) if item else load(a.data, a.frac, a.sub)
    K = int(df.kc.max()) + 1; tr = df[df.split == 'train']
    sup, rw = R19.support_of(df, a, K); er = erdos_renyi(K, int(sup.sum()), seed=2000 + a.sub)
    extra = {}
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if a.cond in NCA:
        if a.cond in ('nca_expert', 'nca_llm', 'nca_map', 'nca_true'): A = authored(a, df, K)
        elif a.cond == 'nca_dp': A = rw
        else: A = sup
        nk = np.bincount(tr.kc.values, minlength=K)
        kw = dict(dynamic=a.cond != 'nca_static', calibrate=a.cond != 'nca_nonull', gate=a.cond != 'nca_nogate',
                  margin=a.cond != 'nca_nomargin')
        m = NCAKT(K, A, nulls_of(A, a.sub), nk, **kw); family = 'nca'; extra.update(nca=kw)
    elif a.cond in FA:
        A = np.zeros((K, K), np.float32); m = FKGKT3(K, A, recency=False); family = 'probe_fa'
    elif a.cond in SIMPROBE:
        A = {'empty': np.zeros((K, K), np.float32), 'support_emb': sup, 'er_emb': er, 'dp_emb': rw}[a.cond]
        m = FKGKT3(K, A, recency=False)
        with torch.no_grad():
            if a.cond != 'empty': m.S.zero_()
        family = 'probe'
    elif a.cond in GRAPHS:
        A = authored(a, df, K)
        if a.cond.endswith('_dp_emb') or a.cond == 'st_true_dp': A = rewire(A, seed=1000 + a.sub)[0].astype(np.float32)
        if a.cond.startswith('st_'):
            m = R13.FKGKT3State(K, A); family = 'probe_state'
        else:
            m = FKGKT3(K, A, recency=False)
            with torch.no_grad(): m.S.zero_()
            family = 'probe'
    else:
        N = int(df.q.max()) + 1
        qk = df.drop_duplicates('q').set_index('q').kc.reindex(range(N)).fillna(0).astype(int).values
        if a.cond == 'q_rqm':
            perm = np.random.default_rng(4000 + a.sub).permutation(N); qk = qk[perm]
        A = {'q_qm_g': sup, 'q_qm_dp': rw}.get(a.cond)
        m = QProbe(N, K, qk, A=A, use_item=a.cond in ('q_id', 'q_qm', 'q_rqm', 'q_qm_g', 'q_qm_dp'),
                   use_kc=a.cond != 'q_id', use_diff=a.cond == 'q_diff')
        family = 'qprobe'; A = sup if A is None else A
        extra.update(n_items=N, items_seen_train=int(tr.q.nunique()),
                     test_items_unseen_share=float((~df[df.split == 'test'].q.isin(set(tr.q))).mean()))
    tq, trr, _ = seqs(df, K, 'train'); vq, vr, _ = seqs(df, K, 'valid'); sq, sr, _ = seqs(df, K, 'test')
    if item:
        N = m.N; ti = item_seqs(df, N, 'train'); vi = item_seqs(df, N, 'valid'); si = item_seqs(df, N, 'test')
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)
    best = -1; bad = 0; t0 = time.time(); hist = []; state = None; best_at = 0; steps = 0; stop = False
    bs = 128; ebs = 128; max_steps, eval_every, patience = 2000, 25, 8

    def valid_auc():
        if item:
            o = evaluate_q(m, vi, vq, vr, N, K, ebs)
        else:
            o = R13.evaluate(m, vq, vr, K, ebs, {})
        return roc_auc_score(o['y'], o['p'])
    while not stop:
        for j, qa, ra, qn, y in batches(tq, trr, bs, True, rng):
            m.train()
            if item:
                ia, inx = ti[j][:, :-1], ti[j][:, 1:]
                lo = m(ia, ra, inx); mk = inx < N
                loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
            else:
                mk = qn < K
                if family == 'nca':
                    lo, lo0 = m(qa, ra, qn, return_null=m.use_margin)  if m.use_margin else (m(qa, ra, qn), None)
                    loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float()) + m.extra_loss(qa, ra, qn, y, lo, lo0)
                else:
                    lo = m(qa, ra, qn); loss = F.binary_cross_entropy_with_logits(lo[mk], y[mk].float())
                    if family == 'probe_fa':
                        T = qa.size(1)
                        seen = ((qn.unsqueeze(2) == qa.unsqueeze(1)) & torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0)).any(2)
                        fm = mk & ~seen
                        if fm.any(): loss = loss + LAM1 * F.binary_cross_entropy_with_logits(lo[fm], y[fm].float())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); steps += 1
            if steps % eval_every == 0:
                va = valid_auc(); hist.append(va)
                if va > best + 1e-5: best = va; bad = 0; state = {k: v.detach().clone() for k, v in m.state_dict().items()}; best_at = steps
                else: bad += 1
                if bad >= patience: stop = True; break
            if steps >= max_steps: stop = True; break
    if state is not None: m.load_state_dict(state)
    if item:
        o = evaluate_q(m, si, sq, sr, N, K, ebs)
    else:
        o = R13.evaluate(m, sq, sr, K, ebs, {'sup': neighbour_sets(sup), 'er': neighbour_sets(er)})
    P, Y, NW = o['p'], o['y'], o['new']
    if family == 'nca':
        with torch.no_grad():
            al = []; m.eval()
            for jj, qa, ra, qn, yy in batches(sq, sr, ebs, False, None):
                m(qa, ra, qn); mk = (qn < K)
                seen = ((qn.unsqueeze(2) == qa.unsqueeze(1)) & torch.tril(torch.ones(qa.size(1), qa.size(1), dtype=torch.bool)).unsqueeze(0)).any(2)
                al.append(np.stack([m.last_alpha[mk].numpy(), (~seen)[mk].numpy()], 1))
            al = np.concatenate(al)
        extra.update(alpha_mean=float(al[:, 0].mean()), alpha_first=float(al[al[:, 1] > 0, 0].mean()),
                     alpha_repeat=float(al[al[:, 1] == 0, 0].mean()))
    res = dict(tag=tag, data=a.data, frac=a.frac, sub=a.sub, rep=a.rep, seed=seed, cond=a.cond, family=family, budget='steps',
               K=K, device='cpu', torch=torch.__version__, batch_size=bs, lr=a.lr, d=64, maxlen=200,
               n_params=int(sum(p.numel() for p in m.parameters())), n_train_learners=int(tr.user.nunique()),
               n_train_inter=int(len(tr)), edges=int(sup.sum()), edges_cond=int(np.asarray(A).sum()),
               test_auc=float(roc_auc_score(Y, P)), test_auc_first=float(roc_auc_score(Y[NW], P[NW])),
               test_acc=float(((P > 0.5) == Y).mean()), test_rmse=float(np.sqrt(((P - Y) ** 2).mean())), test_ece=ece(P, Y),
               n_pred=int(len(Y)), n_first=int(NW.sum()), valid_auc=float(best), steps=steps, best_step=best_at,
               hit_max=bool(steps >= max_steps and bad < patience), n_evals=len(hist), valid_hist=hist, **extra)
    if item:
        NI = o['new_item']; res['test_auc_first_item'] = float(roc_auc_score(Y[NI], P[NI])); res['n_first_item'] = int(NI.sum())
    np.savez_compressed(f'{a.out}/{tag}_pred.npz', p=P.astype(np.float32), y=Y.astype(np.int8), new=NW,
                        kc=o['kc'].astype(np.int32), uid=o['uid'].astype(np.int32), **({'new_item': o['new_item']} if item else {}))
    if sim: torch.save(m, f'{a.out}/{tag}.pt')
    res['secs_total'] = time.time() - t0
    json.dump(res, open(f'{a.out}/{tag}.json', 'w'))
    print(tag, 'auc', round(res['test_auc'], 4), 'first', round(res['test_auc_first'], 4), 'steps', steps,
          's', round(res['secs_total']), flush=True)


if __name__ == '__main__': main()
