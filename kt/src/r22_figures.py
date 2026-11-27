"""Release 8.0.0: Fig. 3 of the fourth UMUAI version (positive control by world, three worlds)."""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r22'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})
P = json.load(open(f'{OUT}/positive_control.json'))
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.2), sharey=True)
for ax, w, title in zip(axs, ('sim_dbe', 'sim_dbe_mix', 'sim_a09'), ('(a) sim-DBE', '(b) sim-DBE-mix', '(c) sim-A09')):
    for key, lab, col, mk in (('true_emb-true_dp_emb', 'true graph: G* − DP(G*)', '#1f5fa8', 'o'), ('support_emb-dp_emb', 'log-built graph: G − DP(G)', '#c4581b', 's')):
        xs, ms, lo, hi = [], [], [], []
        for f in ('5', '10', '25', '100'):
            v = P['by_world'][w].get(f, {}).get(key)
            if v is None: continue
            xs.append(float(f)); ms.append(v['mean'])
            c = v['ci'] or [v['mean'], v['mean']]; lo.append(v['mean'] - c[0]); hi.append(c[1] - v['mean'])
        ax.errorbar(xs, ms, yerr=[lo, hi], marker=mk, ms=3.5, lw=1, capsize=2, color=col, label=lab)
    ax.axhline(0, color='0.5', lw=0.6); ax.set_xscale('log'); ax.set_xticks([5, 10, 25, 100]); ax.set_xticklabels(['5%', '10%', '25%', '100%']); ax.set_xlim(4.2, 120); ax.minorticks_off()
    ax.set_title(title, loc='left', fontsize=7.5); ax.set_xlabel('training learners')
axs[0].set_ylabel('edge term, first-attempt AUC')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=2)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f'{OUT}/fig3_positive_control.png', dpi=300); plt.close(fig)
