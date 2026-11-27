"""Release 8.0.0: a third simulated world whose logs partly violate prerequisite order (reviewer comment M2), and the
share of first attempts made before the true prerequisites are mastered in every world (reviewer question 5).

sim_dbe_mix: the parameters, true graph G* and learning rule of sim_dbe (r20_sim.py) unchanged; only the walk that
generates the logs differs. Each learner's curriculum order is the world's order perturbed by Gaussian noise with
standard deviation K/3 ranks (20 of 59 KCs) instead of 3, so learners often meet a KC before its prerequisites.
The same per-learner random streams are used. Logs: 3,000 learners x 200 steps, as for the other worlds.

Usage: python r22_sim.py world        # writes data/sim_dbe_mix.parquet, _world.npz, _true.json
       python r22_sim.py share        # results_r22/prereq_share.json
"""
import os, sys, json, shutil, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_sim as S
DATA = S.DATA; OUT = f'{S.ROOT}/results_r22'; os.makedirs(OUT, exist_ok=True)
MIX = {'sim_dbe_mix': ('sim_dbe', 1 / 3)}


class MixLearners(S.Learners):
    def __init__(s, W, n, seed, sd):
        super().__init__(W, n, seed)
        K = W['K']; s.lorder = []
        ss = np.random.SeedSequence(seed + 1)                                # separate stream for the order noise
        for u, k in enumerate(ss.spawn(n)):
            noisy = W['rank'] * (K - 1) + np.random.default_rng(k).normal(0, sd, K)
            s.lorder.append(np.argsort(noisy, kind='stable'))


def learners(name, W, n, seed):
    if name in MIX: return MixLearners(W, n, seed, MIX[name][1] * W['K'])
    return S.Learners(W, n, seed)


def make_world(name):
    base, _ = MIX[name]
    shutil.copy(f'{DATA}/{base}_world.npz', f'{DATA}/{name}_world.npz')
    t = json.load(open(f'{DATA}/{base}_true.json')); t['note'] = f'true prerequisite graph of {name} (= {base}; r22_sim.py)'
    json.dump(t, open(f'{DATA}/{name}_true.json', 'w'))
    W = S.load_world(base); L = learners(name, W, S.N_LOG, 4242)
    Q = np.zeros((S.N_LOG, S.T_LOG), int); Y = np.zeros((S.N_LOG, S.T_LOG), int)
    for t_ in range(S.T_LOG):
        ch = L.walk_choice(); Q[:, t_] = ch; Y[:, t_] = L.step(ch, walk=True)
    names = W['names']
    df = pd.DataFrame({'user': np.repeat([f'u{u:05d}' for u in range(S.N_LOG)], S.T_LOG), 'kc': Q.ravel(),
                       'kc_name': [names[k] for k in Q.ravel()], 'correct': Y.ravel(), 'order': np.tile(np.arange(S.T_LOG), S.N_LOG)})
    df.to_parquet(f'{DATA}/{name}.parquet')
    print(name, 'correct', round(df.correct.mean(), 3), 'KCs seen', df.kc.nunique())


def share(name):
    """Replays the log generation with the same seed and records, for every first attempt at a KC, whether all of the
    KC's true prerequisites were mastered just before it."""
    W = S.load_world(MIX[name][0] if name in MIX else name); L = learners(name, W, S.N_LOG, 4242)
    seen = np.zeros((S.N_LOG, W['K']), bool); A = W['A'] > 0; npre = A.sum(0)
    tot = 0; tot_pre = 0; before = 0; first_correct = {True: [], False: []}
    for t_ in range(S.T_LOG):
        ch = L.walk_choice(); idx = np.arange(S.N_LOG); fa = ~seen[idx, ch]
        unmet = np.array([(A[:, k] & ~L.m[u]).any() for u, k in zip(idx, ch)])
        y = L.step(ch, walk=True)
        tot += fa.sum(); tot_pre += (fa & (npre[ch] > 0)).sum(); before += (fa & unmet).sum()
        for f, u_, yy in zip(fa & (npre[ch] > 0), unmet, y):
            if f: first_correct[bool(u_)].append(yy)
        seen[idx, ch] = True
    return dict(first_attempts=int(tot), with_prerequisites=int(tot_pre), before_prerequisites_mastered=int(before),
                share_all=float(before / tot), share_of_kcs_with_prerequisites=float(before / max(tot_pre, 1)),
                correct_if_unmet=float(np.mean(first_correct[True])) if first_correct[True] else None,
                correct_if_met=float(np.mean(first_correct[False])) if first_correct[False] else None)


if __name__ == '__main__':
    if sys.argv[1] == 'world':
        for n in MIX: make_world(n)
    else:
        out = {w: share(w) for w in ('sim_dbe', 'sim_a09', 'sim_dbe_mix')}
        json.dump(out, open(f'{OUT}/prereq_share.json', 'w'), indent=1); print(json.dumps(out, indent=1))
