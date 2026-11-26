"""Release 5.0.0: Fig. 5 of the UMUAI revision (next-KC decisions). Reads results_r19/decisions.json and decisions_rows.csv;
writes results_r19/fig5_decisions.png. (a) Agreement of next-KC choices by rule for three pairs of models at 25%;
(b) regret of the target rule against first-attempt AUC, both centred within corpus, for the two simulated learners.
Shapes and hatching distinguish the series, so the figure reads in grayscale."""
import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = f'{REPO}/results_r19'
C1, C2, C3 = '#257de0', '#f47230', '#00b894'; INK = '#333333'; GRID = '#dddddd'
D = json.load(open(f'{RES}/decisions.json')); R = pd.read_csv(f'{RES}/decisions_rows.csv')
plt.rcParams.update({'font.size': 8.5, 'axes.edgecolor': '#888888', 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
A = D['agreement_mean']; rules = [('target', 'Target'), ('target_tb0.01', 'Stabilized\nε = 0.01'), ('target_tb0.02', 'Stabilized\nε = 0.02'), ('easiest', 'Easiest')]
pairs = [('control_seeds', 'Control, two seeds', C1, ''), ('graph_control_crossed', 'Graph vs control, seeds crossed', C2, '///'),
         ('graph_control_shared', 'Graph vs control, shared initial weights', C3, '...')]
x = np.arange(len(rules)); w = 0.26
for i, (k, lab, col, h) in enumerate(pairs):
    v = [A[k][r] for r, _ in rules]
    ax[0].bar(x + (i - 1) * w, v, w - 0.03, color=col, hatch=h, edgecolor='white', linewidth=0.6, label=lab)
ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in rules]); ax[0].set_ylabel('Share of decisions with the same KC')
ax[0].yaxis.grid(True, color=GRID, linewidth=0.6); ax[0].set_axisbelow(True); ax[0].set_ylim(0, max(A['graph_control_shared'].values()) * 1.18)
ax[0].legend(frameon=False, fontsize=7, loc='upper left'); ax[0].set_title('(a) Agreement of next-KC choices, 25%', fontsize=9, loc='left')
for sim, lab, col, mk in (('rasch', 'Rasch learner', C1, 'o'), ('pfa', 'Learning simulator (PFA-style)', C2, 's')):
    d = R[R.sim == sim].copy()
    d['x'] = d.auc_first - d.groupby('data').auc_first.transform('mean'); d['y'] = d.regret_target - d.groupby('data').regret_target.transform('mean')
    ax[1].scatter(d.x, d.y, s=12, marker=mk, facecolor='none' if mk == 's' else col, edgecolor=col, linewidth=0.8, label=lab, alpha=0.85)
    s = D['regret_vs_auc'][f'{sim}|target']['slope']; xs = np.linspace(d.x.min(), d.x.max(), 2)
    ax[1].plot(xs, s * xs, color=col, linewidth=1.6, linestyle='-' if mk == 'o' else '--')
ax[1].axhline(0, color='#aaaaaa', linewidth=0.6); lo, hi = ax[1].get_ylim(); ax[1].set_ylim(lo, hi + 0.55 * (hi - lo)); ax[1].set_xlabel('First-attempt AUC (centred within corpus)')
ax[1].set_ylabel('Regret, target rule (centred)'); ax[1].legend(frameon=False, fontsize=7, loc='upper left', ncol=1)
ax[1].set_title('(b) Regret against first-attempt AUC', fontsize=9, loc='left'); ax[1].yaxis.grid(True, color=GRID, linewidth=0.6); ax[1].set_axisbelow(True)
fig.tight_layout(); fig.savefig(f'{RES}/fig5_decisions.png', dpi=300); print('saved')
