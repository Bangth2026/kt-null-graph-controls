"""Unit tests of the graphs used by the knowledge-tracing experiments (kt/src): the order-preserving support graph of
Eq. (2), the degree-preserving rewiring, the Erdos-Renyi null, the unit-preserving null and the two constructions of
release 3.13.0. Run with: PYTHONPATH=. pytest tests/test_kt_graphs.py (needs torch for the import of kt/src)."""
import os, sys
import numpy as np, pandas as pd
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kt', 'src'))
torch = pytest.importorskip('torch')
from r5_controls import transition_support, rewire          # noqa: E402
from r6_controls import erdos_renyi                          # noqa: E402
import r14_controls as R14                                   # noqa: E402
import r16_controls as R16                                   # noqa: E402


def logs(seed=0, n=60, K=12, L=40):
    rng = np.random.default_rng(seed); rows = []
    for u in range(n):
        q = np.sort(rng.integers(0, K, L)) if u % 2 else rng.integers(0, K, L)   # half the learners follow the order 0..K-1
        for t, k in enumerate(q): rows.append(dict(user=f'u{u}', kc=int(k), correct=int(rng.random() < .6 + .02 * k), order=t))
    return pd.DataFrame(rows), K


def test_support_is_antisymmetric_and_zero_diagonal():
    df, K = logs(); mu = transition_support(df, K)
    assert np.all(np.diag(mu) == 0)
    assert not np.any((mu > 0) & (mu.T > 0))           # only the dominant direction of a pair can be an edge
    assert np.all((mu >= 0) & (mu <= 1))


def test_support_threshold():
    df, K = logs(); T = np.zeros((K, K))
    for _, g in df.groupby('user', sort=False):
        q = g.kc.values; r = q[np.r_[True, q[1:] != q[:-1]]]; np.add.at(T, (r[:-1], r[1:]), 1)
    mu = transition_support(df, K, s=10)
    assert np.all(mu[(T + T.T) < 10] == 0)


def test_rewire_keeps_degrees():
    df, K = logs(); A = (transition_support(df, K) > 0).astype(np.float32)
    R, done = rewire(A, seed=1)
    assert R.sum() == A.sum() and done > 0
    assert np.array_equal(R.sum(0), A.sum(0)) and np.array_equal(R.sum(1), A.sum(1))
    assert np.all(np.diag(R) == 0) and not np.any((R > 0) & (R.T > 0))


def test_erdos_renyi_size():
    A = erdos_renyi(20, 37, seed=3)
    assert A.sum() == 37 and np.all(np.diag(A) == 0) and not np.any((A > 0) & (A.T > 0))


def test_unit_random_keeps_within_unit_count():
    df, K = logs(); sup = (transition_support(df, K) > 0).astype(np.float32)
    grp = np.array([str(k // 4) for k in range(K)]); same = grp[:, None] == grp[None, :]
    U = R14.unit_random(sup, grp, R14.first_encounter_rank(df, K), seed=7)
    assert U.sum() == sup.sum() and (U * same).sum() == (sup * same).sum()


def test_top_pairs_and_corr_graph():
    W = np.random.default_rng(0).random((9, 9)); np.fill_diagonal(W, 0)
    A = R16.top_pairs(W, 11)
    assert A.sum() == 11 and not np.any((A > 0) & (A.T > 0)) and np.all(np.diag(A) == 0)
    df, K = logs(); C = R16.corr_graph(df, K, 15, min_common=5)
    assert C.sum() == 15 and not np.any((C > 0) & (C.T > 0))


# release 5.0.0 ---------------------------------------------------------------------------------------------------------
import r19_controls as R19                                   # noqa: E402
import r19_decisions as R19D                                 # noqa: E402


def test_pdkt_penalty_zero_without_edges_and_positive_when_violated():
    K = 6; q = torch.randint(0, K, (4, 10)); r = torch.randint(0, 2, (4, 10)); rng = np.random.default_rng(0)
    m0 = R19.PDKT(K, np.zeros((K, K), np.float32)); assert float(m0.penalty(q, r, rng)) == 0.0
    A = np.zeros((K, K), np.float32); A[0, 1] = 1; m = R19.PDKT(K, A)
    with torch.no_grad():
        m.out.weight.zero_(); m.out.bias.zero_(); m.out.bias[1] = 3.0      # P(1) > P(0): the constraint 0 -> 1 is violated
    m.eval(); assert float(m.penalty(q, r, rng)) > 0.4


def test_stabilised_rule_breaks_near_ties_by_curriculum_order():
    a = np.array([0.651, 0.655, 0.70, 0.30]); cnd = np.ones(4, bool); rank = np.array([0.9, 0.1, 0.0, 0.5])
    assert R19D.choose(a, cnd, 'target', rank) == 0
    assert R19D.choose(a, cnd, 'target_tb0.01', rank) == 1          # 0.655 is within 0.01 of the best and met earlier
    assert R19D.choose(a, cnd, 'easiest', rank) == 2


def test_dp_of_a_larger_graph_keeps_degrees_and_has_no_reciprocal_pair():
    df, K = logs(seed=3, n=200); A = (transition_support(df, K, s=2) > 0).astype(np.float32)
    D, _ = rewire(A, seed=1)
    assert (D.sum(0) == A.sum(0)).all() and (D.sum(1) == A.sum(1)).all()
    assert not np.any((D > 0) & (D.T > 0)) and np.all(np.diag(D) == 0)


# release 6.0.0 ---------------------------------------------------------------------------------------------------------
import r20_models as R20M                                    # noqa: E402
import r20_controls as R20                                   # noqa: E402
import r20_sim as R20S                                       # noqa: E402


def _dag(K=10, seed=0, p=0.25):
    A = np.triu((np.random.default_rng(seed).random((K, K)) < p).astype(np.float32), 1); return A


def test_nca_with_closed_gate_is_the_empty_graph_readout():
    K = 10; A = _dag(K); m = R20M.NCAKT(K, A, R20.nulls_of(A, 0), np.arange(K) * 5); m.eval()
    with torch.no_grad():
        m.gt.bias.fill_(-50.0)                                   # alpha -> 0 at the target
    q = torch.randint(0, K, (3, 8)); r = torch.randint(0, 2, (3, 8)); qn = torch.randint(0, K, (3, 8))
    with torch.no_grad():
        lo, lo0 = m(q, r, qn, return_null=True)
    assert torch.allclose(lo, lo0, atol=1e-6)


def test_nca_nulls_keep_degrees_and_differ_from_dp():
    A = _dag(14, seed=3, p=0.3); N = R20.nulls_of(A, 0); D, _ = rewire(A, seed=1000)
    for X in N:
        assert (X.sum(0) == A.sum(0)).all() and (X.sum(1) == A.sum(1)).all()
    assert not any(np.array_equal(X, D) for X in N)


def test_score_all_matches_forward_for_probe_and_nca():
    from r5_controls import FKGKT3
    K = 12; A = _dag(K, seed=1); torch.manual_seed(0)
    m1 = FKGKT3(K, A)
    with torch.no_grad(): m1.S.zero_()
    m2 = R20M.NCAKT(K, A, R20.nulls_of(A, 0), np.arange(K))
    assert R20S.check_scores(m1, K) < 1e-5 and R20S.check_scores(m2, K) < 1e-5


def test_random_q_matrix_keeps_items_per_kc():
    qk = np.array([0, 0, 1, 2, 2, 2, 3]); perm = np.random.default_rng(4000).permutation(len(qk)); rq = qk[perm]
    assert (np.bincount(rq, minlength=4) == np.bincount(qk, minlength=4)).all()


def test_simulated_learners_share_random_numbers_across_policies():
    K = 6; A = _dag(K, seed=2, p=0.4)
    W = dict(A=A, g=np.full(K, .2), s=np.full(K, .1), l=np.full(K, .3), pi=np.full(K, .3), order=np.arange(K), rank=np.linspace(0, 1, K), K=K)
    a = R20S.Learners(W, 5, seed=1); b = R20S.Learners(W, 5, seed=1)
    assert (a.m == b.m).all()
    ch = np.zeros(5, int); assert (a.step(ch) == b.step(ch)).all()


# ------------------------------------------------------------------ release 7.0.0
def test_gate_opens_kc_only_when_all_prerequisites_passed():
    import r21_sim as R21S
    G = np.zeros((4, 4), np.float32); G[0, 2] = G[1, 2] = 1; G[2, 3] = 1            # 0,1 -> 2 -> 3
    P = np.array([[True, False, False, False], [True, True, False, False], [True, True, True, False]])
    O = R21S.open_set(G, P)
    assert O[:, 0].all() and O[:, 1].all()                                            # no prerequisites
    assert list(O[:, 2]) == [False, True, True] and list(O[:, 3]) == [False, False, True]


def test_passed_needs_two_correct_in_a_row():
    import r21_sim as R21S
    hq = np.array([[0, 0, 1, 1, 1]]); hr = np.array([[1, 1, 1, 0, 1]])
    P = R21S.passed(hq, hr, 5, 3)
    assert P[0, 0] and not P[0, 1] and not P[0, 2]


def test_support_is_dominant_direction_with_min_count():
    import pandas as pd
    from r5_controls import transition_support
    # 12 moves 0->1, 3 moves 1->0 (support 15 >= 10): edge 0->1 only; 4 moves 1->2: below s, no edge
    seqs = [[0, 1]] * 12 + [[1, 0]] * 3 + [[1, 2]] * 4
    df = pd.DataFrame([dict(user=f'u{i}', kc=k) for i, s in enumerate(seqs) for k in s])
    A = transition_support(df, 3, s=10) > 0
    assert A[0, 1] and not A[1, 0] and not A[1, 2] and A.sum() == 1


def test_route_graph_chains_siblings_without_reciprocal_pairs():
    import json, tempfile, argparse, pandas as pd
    import r21_controls as R21, r20_controls as R20
    d = tempfile.mkdtemp(); json.dump({'groups': [['a', 'b', 'c'], ['d', 'e']]}, open(f'{d}/xes3g5m_route.json', 'w'))
    old = R20.DATA; R20.DATA = d
    try:
        rows = [dict(user='u1', kc=k, kc_name=n, split='train') for k, n in [(2, 'c'), (0, 'a'), (1, 'b'), (4, 'e'), (3, 'd')]]
        df = pd.DataFrame(rows); A = R21.route_graph(df, 5, ['a', 'b', 'c', 'd', 'e'])
    finally:
        R20.DATA = old
    assert A.sum() == 3 and (A * A.T).sum() == 0
    assert A[2, 0] == 1 and A[0, 1] == 1 and A[4, 3] == 1                             # order in which the learner met them


# ---------------------------------------------------------------- release 8.0.0
def test_partial_q_perturbation_keeps_items_per_kc_and_moves_only_the_share():
    import numpy as np, r20_controls as R20
    qk = np.repeat(np.arange(20), 10); out = R20.perturb_q(qk, 0.2, 5000)
    assert (np.bincount(out, minlength=20) == np.bincount(qk, minlength=20)).all()
    assert (out != qk).sum() <= 40 and (out != qk).sum() > 0
    assert (R20.perturb_q(qk, 0.2, 5000) == out).all()                                # reproducible


def test_mix_learners_get_a_perturbed_order_and_keep_the_base_order_otherwise():
    import numpy as np, r20_sim as S, r22_sim as S22
    W = S.load_world('sim_dbe'); base = S.Learners(W, 30, 7); mix = S22.MixLearners(W, 30, 7, W['K'] / 3)
    for o in mix.lorder: assert sorted(o) == list(range(W['K']))
    assert np.mean([not np.array_equal(a, b) for a, b in zip(mix.lorder, base.lorder)]) > 0.9
    assert np.allclose(mix.speed, base.speed)                                        # same learners otherwise


def test_window_candidates_are_unseen_and_at_most_ten():
    import numpy as np, r22_decisions as D
    rng = np.random.default_rng(0); n, L, K = 4, 6, 40
    seen = rng.random((n, L, K)) < 0.3; cur = rng.integers(0, K, (n, L)); rank = rng.permutation(K) / K
    win, uni = D.candidate_sets(seen, cur, rank, None)
    assert uni is None and (win & seen).sum() == 0 and win.sum(-1).max() <= 10



# ---------------------------------------------------------------- release 9.0.0
def test_gain_rollout_equals_scoring_the_extended_history():
    import glob, torch, r23_gain as G
    f = sorted(glob.glob('/home/claude/kt/runs_r2*_sim/sim_dbe__f10__s0__empty__steps.pt'))
    if not f: return
    m = torch.load(f[0], weights_only=False)
    assert G.check_gain(m, m.K) < 1e-5


def test_order_stats_counts_unattempted_and_not_passed_prerequisites():
    import numpy as np, pandas as pd, r23_order as O
    A = np.zeros((3, 3)); A[0, 1] = 1                                    # 0 is a prerequisite of 1
    df = pd.DataFrame(dict(user=['a'] * 3 + ['b'] * 4, kc=[1, 0, 2, 0, 0, 1, 2], correct=[0, 1, 1, 1, 1, 1, 0], order=[0, 1, 2, 0, 1, 2, 3]))
    r = O.order_stats(df, A)
    assert r['with_prerequisites'] == 2 and r['share_unattempted'] == 0.5 and r['share_not_passed'] == 0.5
    assert r['correct_if_unattempted'] == 0.0 and r['correct_if_passed'] == 1.0
