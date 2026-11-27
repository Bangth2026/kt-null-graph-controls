"""Release 7.0.0: training runs of the third UMUAI version (reviewer comments 4.1 and 4.6).

Same data, subsamples, update budget, seeds and evaluation as r20_controls.py, whose conditions pass through unchanged.
Added conditions (graph in the embedding refinement only, attention bias off, as the true_emb runs of release 6.0.0):

  Simulated worlds (sim_a09, sim_dbe), for the positive control with the full ladder on the known graph G*:
    uni_emb        U, the complete graph on the KCs without self-loops
    true_er_emb    ER(G*), a directed Erdos-Renyi graph with as many edges as G* (seed 2000 + s, as er_emb)
  (true_emb = G*, true_dp_emb = DP(G*), empty, er_emb = ER(G), dp_emb = DP(G), support_emb = G are those of r20.)

  XES3G5M (held-out corpus), an authored structure:
    route_emb      the KC routes published with XES3G5M: leaf KCs that share their parent node in the route tree are
                   chained in the order in which the training learners of the subsample meet them (an edge from each
                   sibling to the next), data/xes3g5m_route.json lists the sibling groups
    route_dp_emb   its degree-preserving rewiring (seed 1000 + s)
Usage: as r20_controls.py, e.g. python r21_controls.py --data sim_a09 --frac 0.05 --sub 1 --cond uni_emb --out DIR
"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_controls as R20
import r14_controls as R14
from r6_controls import erdos_renyi

NEW = ('uni_emb', 'true_er_emb', 'route_emb', 'route_dp_emb')
_authored = R20.authored


def route_graph(df, K, names):
    groups = json.load(open(f'{R20.DATA}/xes3g5m_route.json'))['groups']
    idx = {n: i for i, n in enumerate(names)}
    rank = R14.first_encounter_rank(df[df.split == 'train'], K); A = np.zeros((K, K), np.float32)
    for g in groups:
        ks = [idx[n] for n in g if n in idx and np.isfinite(rank[idx[n]])]
        ks.sort(key=lambda k: (rank[k], k))
        for a, b in zip(ks[:-1], ks[1:]): A[a, b] = 1
    return A


def authored(a, df, K):
    if a.cond == 'uni_emb':
        A = np.ones((K, K), np.float32); np.fill_diagonal(A, 0); return A
    if a.cond == 'true_er_emb':
        G = R20.named_graph(f'{R20.DATA}/{a.data}_true.json', R14.kc_names(df, K))
        return erdos_renyi(K, int(G.sum()), seed=2000 + a.sub)
    if a.cond in ('route_emb', 'route_dp_emb'):
        return route_graph(df, K, R14.kc_names(df, K))
    return _authored(a, df, K)


R20.authored = authored
R20.GRAPHS = R20.GRAPHS + NEW

if __name__ == '__main__':
    R20.main()
