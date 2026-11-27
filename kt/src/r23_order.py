"""Release 9.0.0: how often learners meet a KC before its prerequisites, in real corpora with an authored map and in the
simulated worlds (reviewer comment R1 of the second round).

For each learner's first attempt at a KC that has at least one prerequisite in the map (among KCs that occur in the
corpus), two states of the prerequisites are recorded from the learner's own earlier answers:
  unattempted  at least one prerequisite has not been attempted before
  not passed   at least one prerequisite is not passed, a prerequisite being passed when the learner's last two answers
               on it were correct (the criterion of the gate, r21_sim.passed)
and the proportion correct of first attempts in each state. The same observable criteria are applied to the logs of the
simulated worlds, where the true state (mastery) is also known (r22_sim.share), so that real and simulated shares are
measured the same way.
Maps: Junyi, the published prerequisite map (directed; r14_controls.junyi_map); DBE-KT22, the instructor relations,
which are undirected and are oriented by the order in which training learners first meet the KCs (r20_controls.authored),
so an out-of-order share there is bounded by construction; XES3G5M, the KC routes (r21_controls.route_graph), chained in
the order learners meet sibling KCs, likewise oriented by encounter order. sim-DBE, sim-DBE-mix, sim-A09: the true graph.
Corpora as used in the study (r5_controls.load at 100%: at most 6,000 learners and 200 interactions each), all learners.
Output: results_r23/prereq_order.json
"""
import os, sys, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r5_controls import load
import r14_controls as R14, r20_controls as R20, r21_controls as R21
ROOT = os.environ.get('KT_ROOT', '/home/claude/kt'); OUT = f'{ROOT}/results_r23'; os.makedirs(OUT, exist_ok=True)


def graph(name, df, K):
    names = R14.kc_names(df, K)
    if name == 'junyi': return R14.junyi_map(names), 'published prerequisite map (directed)'
    if name == 'dbe_kt22':
        return R20.authored(argparse.Namespace(data='dbe_kt22', cond='expert_emb'), df, K), 'instructor relations, oriented by first-encounter order'
    if name == 'xes3g5m': return R21.route_graph(df, K, names), 'KC routes, chained in encounter order'
    return R20.named_graph(f'{ROOT}/data/{name}_true.json', names), 'true graph'


def order_stats(df, A):
    """A[p, k] = 1: p is a prerequisite of k."""
    K = A.shape[0]; pre = [np.flatnonzero(A[:, k]) for k in range(K)]
    n = un = np_ = 0; c_un = c_at = c_np = c_p = 0; n_at = n_p = 0
    for _, g in df.sort_values(['user', 'order']).groupby('user', sort=False):
        q = g.kc.values; y = g.correct.values
        seen = np.zeros(K, bool); last = np.full(K, -1); prev = np.full(K, -1)
        for k, r in zip(q, y):
            if not seen[k] and len(pre[k]):
                p = pre[k]; n += 1
                if (~seen[p]).any(): un += 1; c_un += r
                else: n_at += 1; c_at += r
                if ~((last[p] == 1) & (prev[p] == 1)).all(): np_ += 1; c_np += r
                else: n_p += 1; c_p += r
            seen[k] = True; prev[k] = last[k]; last[k] = r
    nfa = int(df.drop_duplicates(['user', 'kc']).shape[0])
    f = lambda a, b: float(a / b) if b else None
    return dict(first_attempts=nfa, with_prerequisites=int(n), kcs_with_prerequisites=int(sum(len(p) > 0 for p in pre)), edges=int(A.sum()),
                share_unattempted=f(un, n), share_not_passed=f(np_, n), share_not_passed_of_all=f(np_, nfa),
                correct_if_unattempted=f(c_un, un), correct_if_attempted=f(c_at, n_at), correct_if_not_passed=f(c_np, np_), correct_if_passed=f(c_p, n_p))


def main():
    out = {}
    for name in ('junyi', 'dbe_kt22', 'xes3g5m', 'sim_dbe', 'sim_dbe_mix', 'sim_a09'):
        df = load(name, 1.0, 0); K = int(df.kc.max()) + 1
        A, what = graph(name, df, K)
        out[name] = dict(map=what, learners=int(df.user.nunique()), **order_stats(df, A))
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)
    json.dump(out, open(f'{OUT}/prereq_order.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
