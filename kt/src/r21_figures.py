"""Release 7.0.0: Figs. 2 and 3 of the third UMUAI version (ladder diagram; positive control)."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = f'{REPO}/results_r21'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7})


def fig_ladder():
    fig, ax = plt.subplots(figsize=(6.3, 1.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    rungs = [('∅', 'empty graph', 'no edges;\nevery other\nparameter kept'), ('U', 'complete graph', 'maps of the graph\nswitched on;\nno KC-specific input'),
             ('ER(G)', 'Erdős–Rényi', 'number of edges\nof G'), ('DP(G)', 'degree-preserving\nrewiring', 'in- and out-degree\nof every KC'),
             ('G', 'support graph\n(or G* in the worlds)', 'the edges\nthemselves')]
    xs = [0.9, 2.95, 5.0, 7.05, 9.1]
    for x, (sym, name, keeps) in zip(xs, rungs):
        ax.add_patch(FancyBboxPatch((x - 0.75, 1.75), 1.5, 0.9, boxstyle='round,pad=0.02', fc='#eef3fa', ec='#1f5fa8', lw=0.8))
        ax.text(x, 2.38, sym, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, 1.98, name, ha='center', va='center', fontsize=6.3)
        ax.text(x, 1.05, keeps, ha='center', va='center', fontsize=6.2, color='0.25')
    terms = ['Δ(U)\nmaps', 'Δ(ER) − Δ(U)\nKC-specific input', 'Δ(DP) − Δ(ER)\ndegrees', 'Δ(G) − Δ(DP)\nedges']
    for a, b, t in zip(xs[:-1], xs[1:], terms):
        ax.annotate('', xy=(b - 0.78, 2.2), xytext=(a + 0.78, 2.2), arrowprops=dict(arrowstyle='->', color='#c4581b', lw=1))
        ax.text((a + b) / 2, 2.95, t, ha='center', va='center', fontsize=6.2, color='#c4581b')
    ax.text(0.15, 0.3, 'Each rung keeps everything kept by the rung to its left. Δ(G) = sum of the four terms (Eq. 1).', fontsize=6.3, color='0.3')
    fig.tight_layout(pad=0.2); fig.savefig(f'{OUT}/fig2_ladder.png', dpi=300); plt.close(fig)


def fig_pc():
    P = json.load(open(f'{OUT}/positive_control.json'))
    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.2), sharey=True)
    for ax, w, title in zip(axs, ('sim_dbe', 'sim_a09'), ('(a) sim-DBE', '(b) sim-A09')):
        for key, lab, col, mk in (('true_emb-true_dp_emb', 'true graph G*: G* − DP(G*)', '#1f5fa8', 'o'), ('support_emb-dp_emb', 'log-built G: G − DP(G)', '#c4581b', 's')):
            xs, ms, se = [], [], []
            for f in ('5', '10', '25', '100'):
                v = P['by_world'][w].get(f, {}).get(key)
                if v is None: continue
                xs.append(float(f)); ms.append(v['mean']); se.append(v['se'] if v['se'] is not None else 0)
            ax.errorbar(xs, ms, yerr=se, marker=mk, ms=3.5, lw=1, capsize=2, color=col, label=lab)
        ax.axhline(0, color='0.5', lw=0.6); ax.set_xscale('log'); ax.set_xticks([5, 10, 25, 100]); ax.set_xticklabels(['5%', '10%', '25%', '100%'])
        ax.set_title(title, loc='left', fontsize=7.5); ax.set_xlabel('training learners')
    axs[0].set_ylabel('edge term, first-attempt AUC')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(f'{OUT}/fig3_positive_control.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    fig_ladder()
    if os.path.exists(f'{OUT}/positive_control.json'): fig_pc()
