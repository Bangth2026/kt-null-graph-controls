"""Release 9.0.0: closed loop of sim_dbe_mix with its own learners (reviewer comment R5b of the second round).

In release 8.0.0 the closed loop of sim_dbe_mix (r22_simdec.py) generated its test learners with the learners of
sim_dbe (r20_sim.Learners), whose curriculum walk is not perturbed; only the tracers and G100, trained on the sim_dbe_mix
logs, differed from sim_dbe. Here every learner of the closed loop of sim_dbe_mix (warm-up and adaptive steps, all
references, the forgetting calibration) is a MixLearners learner of r22_sim.py (curriculum position perturbed with
SD = K/3), as in the logs. The policies, tracers and gates are those of r22_simdec.py and r23_gain.py.
Output: results_r23/decisions_sim_sim_dbe_mix.json (replaces the release 8.0.0 file for this world).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r20_sim as S, r21_sim as X, r22_simdec as D, r22_sim as M, r23_gain as G


def mix_learners(W, n, seed=777):
    return M.MixLearners(W, n, seed, M.MIX['sim_dbe_mix'][1] * W['K'])


S.Learners = X.Learners = G.Learners = mix_learners
D.OUT = f'{S.ROOT}/results_r23_mix'; os.makedirs(D.OUT, exist_ok=True)
D.main('sim_dbe_mix')
os.environ['GAIN_SRC'] = D.OUT
G.main('sim_dbe_mix')
