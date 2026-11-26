"""Release 5.0.0: builds Online Resource 4 of the UMUAI revision (Sections S14 to S16) as Markdown from the result files of
r19_baselines.py, r19_stats.py and r19_decisions.py, and the run files of runs_r19. Writes results_r19/online_resource_4.md.
"""
import os, json, glob
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = f'{REPO}/results_r19'
NAMES = {'assist09': 'ASSISTments 2009', 'assist12': 'ASSISTments 2012', 'assist15': 'ASSISTments 2015', 'assist17': 'ASSISTments 2017',
         'junyi': 'Junyi Academy', 'algebra05': 'Algebra 2005', 'eedi': 'Eedi', 'ednet': 'EdNet KT1', 'bridge06': 'Bridge to Alg. 2006',
         'statics11': 'Statics 2011', 'xes3g5m': 'XES3G5M', 'xes3g5m1': 'XES3G5M, single-KC questions'}
ORDER = list(NAMES)[:10]


def md_table(df, floatfmt='{:+.4f}'):
    cols = list(df.columns); out = ['| ' + ' | '.join(str(c) for c in cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('' if pd.isna(v) else (floatfmt.format(v) if abs(v) < 1 else f'{v:,.0f}'))
            else: cells.append(str(v))
        out.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(out)


def iv(o, key='mixed'):
    if not o or key not in o: return ''
    m = o[key]; return f"{m['lo']:+.4f}, {m['hi']:+.4f}"


def term_row(lab, o):
    f, a = o['first'], o['all']
    if not f: return dict(Term=lab)
    return {'Term': lab, 'Units': f['units'], 'First att.': f"{f['mean']:+.4f} ({f['pos']}/{f['corpora']})", 'Mixed 95%': iv(f),
            'Fit': f.get('mixed', {}).get('kind', ''), 'Fixed t 95%': iv(f, 'fixed_t'), 'Cluster bootstrap 95%': iv(f, 'cluster_boot'),
            'All pred.': f"{a['mean']:+.4f} ({a['pos']}/{a['corpora']})" if a else ''}


def main():
    S = json.load(open(f'{RES}/stats.json')); B = json.load(open(f'{RES}/baselines.json'))
    GQ = pd.DataFrame(json.load(open(f'{RES}/graph_quality.json')))
    D = json.load(open(f'{RES}/decisions.json')) if os.path.exists(f'{RES}/decisions.json') else None
    L = ['---', 'title: "Online Resource 4: analyses added in the revision (Sections S14 to S16)"', 'subtitle: "When does a concept graph help a knowledge tracer? Entry-point null-graph controls and decision stability for KC-level learner models"',
         'author: "Hai Bang Truong, Saigon International University (truonghaibang@siu.edu.vn)"', '---', '',
         'Every value below is computed by the scripts of release 5.0.0 of the accompanying code (kt/src/r19_*.py) from the run files of kt/runs_r19 and of the earlier releases; the result files are kt/results_r19/*.json. Section numbers continue those of Online Resources 1 and 3.', '']
    # S14 -----------------------------------------------------------------------------------------------------------
    L += ['# S14 Data, reproduction and graphs', '', '## S14.1 Reproduction of the rebuilt corpora', '',
          'Every corpus was rebuilt from the public copies listed in the README of release 5.0.0 with kt/src/prep.py (Junyi with prep_junyi_lowmem.py, XES3G5M with prep_r14.py) and matches Table 1 of the article in learners and capped interactions. The empty-graph control on subsample 0 at 10% was rerun on each rebuilt corpus (kt/runs_r19/repro) and compared with the archived run of the same condition, seed and code.', '']
    rows = []
    for f in sorted(glob.glob(f'{REPO}/runs_r19/repro/*.json')):
        r = json.load(open(f)); c = r['data']; arch = None
        for d in ('runs_r9', 'runs_r13', 'runs_r14', 'runs_r6'):
            p = f"{REPO}/{d}/{c}__f10__s0__empty__steps.json"
            if os.path.exists(p):
                a = json.load(open(p))
                if c != 'ednet' or d == 'runs_r13': arch = (d, a); break
        if arch:
            rows.append({'Corpus': NAMES.get(c, c), 'Archived run': arch[0], 'Archived test AUC': f"{arch[1]['test_auc']:.10f}", 'Rerun test AUC': f"{r['test_auc']:.10f}",
                         'Identical': 'yes' if abs(arch[1]['test_auc'] - r['test_auc']) < 1e-12 else 'no'})
    L += [md_table(pd.DataFrame(rows)), '', 'Statics 2011 was also rerun before the new runs (0.7471066538). EdNet is compared with the rebuilt copy used since release 3.7.0 (runs_r13); against the original release-2.0.0 copy its test AUC differs by 6.5e-6 (kt/README.md, release 3.7.0).', '']
    L += ['## S14.2 Structure ablations of the eight graph-based tracers examined', '',
          '**Table S14.1** What the experimental sections of the eight tracers named in Section 2.1 compare. A tick in the last column would mean a comparison of the graph with a random graph of the same size inside an otherwise unchanged network; none of the eight makes one.', '',
          '| Tracer | Structure ablation reported | Random graph of the same size, network unchanged |', '|---|---|---|',
          '| GKT (Nakagawa et al. 2019) | dense graph against statistics-based and learned graphs, found close | no |',
          '| GIKT (Yang et al. 2021) | number of GCN layers over the question–skill graph varied from 0 to 3 (0: embeddings indexed directly); related skills and related history removed from the interaction modules | no |',
          '| SKT (Tong et al. 2020) | propagation over the prerequisite and similarity relations removed by module (author to verify against the ICDM paper) | no |',
          '| RKT (Pandey and Srivastava 2020) | exercise-relation modeling removed, alone and with position encoding or forgetting; alternative computations of the relation matrix compared | no |',
          '| DyGKT (Cheng et al. 2024) | multiset indicator removed; time encoders simplified or removed; question-difficulty module replaced by question- or concept-ID embeddings; no ablation of graph structure | no |',
          '| PSI-KT (Zhou et al. 2024) | graph inference removed (each KC independent); inferred graph compared with expert and crowd annotations on Junyi | no |',
          '| SINKT (Fu et al. 2024) | generated concept graph replaced by a transition graph built from the logs | no |',
          '| HGKT (Tong et al. 2022) | support relations dropped together with the modules that read them | no |', '']
    L += ['## S14.3 Graph quality by training size', '',
          '**Table S14.3** Support graph of every subsample against the graph from all training learners (*G*~100~). Edges: directed edges; recall: share of the edges of *G*~100~ in the subsample graph; precision: share of the subsample graph\'s edges in *G*~100~; Jaccard: of undirected pairs; reciprocal: reciprocal pairs in *G* and in DP(*G*).', '']
    g = GQ.copy(); g['Corpus'] = g.data.map(NAMES); g['Size'] = (g.frac * 100).round().astype(int).astype(str) + '%'
    g = g[['Corpus', 'Size', 'sub', 'n_train', 'edges', 'edges_100', 'recall', 'precision', 'jaccard', 'reciprocal_G', 'reciprocal_DP']]
    g.columns = ['Corpus', 'Size', 'Subsample', 'Training learners', 'Edges', 'Edges of G100', 'Recall', 'Precision', 'Jaccard', 'Reciprocal G', 'Reciprocal DP']
    L += [md_table(g, '{:.3f}'), '']
    # S15 -----------------------------------------------------------------------------------------------------------
    L += ['# S15 Prediction', '', '## S15.1 First-attempt references and averaging schemes', '',
          '**Table S15.1** First-attempt AUC at 10% of the training learners, mean over the subsamples of the 32-unit design, by corpus and averaging scheme (Section 5.6 of the article). EdNet: no archived prediction file of the support graph on the rebuilt copy.', '']
    rows = []
    for c in ORDER:
        u = [r for r in B if r['data'] == c]; o = {'Corpus': NAMES[c], 'Units': len(u)}
        for m, lab in (('kc_rate', 'KC rate'), ('pfa', 'PFA-style'), ('empty', 'Probe ∅'), ('support_emb', 'Probe G (E)')):
            for sch, sl in (('pooled', 'pooled'), ('per_learner', 'per learner'), ('within_kc', 'within KC')):
                v = [r[m][sch] for r in u if m in r and r[m].get(sch) is not None]
                o[f'{lab}, {sl}'] = float(np.mean(v)) if v else np.nan
        rows.append(o)
    L += [md_table(pd.DataFrame(rows), '{:.4f}'), '']
    L += ['## S15.2 Primary comparison: interval methods, balanced design and seeds', '',
          'Mixed: linear mixed model with a random intercept per corpus, REML (statsmodels mixedlm), Wald interval; Fit: "mixed", or "ols" where the random intercept was not estimable (one unit per corpus, or a singular fit), in which case ordinary least squares over the units is used. Fixed t: mean of corpus means with a t interval over corpora. Cluster bootstrap: corpora resampled, then subsamples within corpus, 10,000 draws. In parentheses: corpora with a positive difference.', '']
    rows = [term_row('5%, update budget (32 units)', S['primary_5pct']), term_row('10%, 32 units (primary)', S['primary_32']),
            term_row('10%, 50 units (balanced)', S['primary_50'])]
    ts = S['three_seeds']
    rows.append(term_row(f"10%, three seeds per unit ({ts['units']} units)", {'first': ts['first'], 'all': ts['all']}))
    rows.append(term_row('same units, seed 0 only', {'first': ts['seed0_same_units'], 'all': None}))
    L += [md_table(pd.DataFrame(rows)), '', f"Smallest effect of interest from the control alone: δ~seed~ = {ts['control_seed_sd_rms']:.4f} (root mean square over units of the within-unit SD of the empty-graph control's first-attempt AUC across seeds 0, 100 and 200; median {ts['control_seed_sd_median']:.4f}).", '']
    pc = ts['first']['per_corpus'] if ts['first'] else {}
    if pc: L += ['Per corpus, three-seed gain on first attempts: ' + '; '.join(f"{NAMES[k]} {v:+.4f}" for k, v in pc.items()) + '.', '']
    L += ['## S15.3 Exploratory terms with false-discovery-rate adjustment', '',
          f"**Table S15.2** Every exploratory first-attempt term of Tables 3 and 4 of the earlier version and of Section 6.4 (results_r14, results_r16): mixed-model estimate and SE, two-sided *p* from the Wald statistic, Benjamini–Hochberg *q* over the {S['fdr']['n_terms']} terms, and the equivalence class from the 90% interval at δ = δ~seed~ (B+/B−: beyond δ; N: nonzero, not beyond; S: nonzero, inside ±δ; E: equivalent to zero; U: unresolved). {S['fdr']['n_p05']} terms have *p* < 0.05 against {S['fdr']['expected_false_at_p05']:.1f} expected by chance; {S['fdr']['n_q05']} have *q* < 0.05.", '']
    T = pd.DataFrame(S['fdr']['terms'])
    T['Term'] = T.term.str.replace('r14/equivalence/', '').str.replace('r14/', '').str.replace('r16/other_constructions/', 'other constructions/')
    T = T.sort_values('p')[['Term', 'mean', 'est', 'se', 'p', 'q_bh'] + (['class_dseed'] if 'class_dseed' in T else [])]
    T.columns = ['Term', 'Mean over corpora', 'Mixed estimate', 'SE', 'p', 'q (BH)'] + (['Class at δ_seed'] if 'class_dseed' in T.columns else [])
    L += [md_table(T, '{:.4f}'), '']
    L += ['## S15.4 Transfer of structure', '', '**Table S15.4** Graph built from all training learners (*G*~100~) given to the probe trained on 10% of them; first attempts unless stated.', '']
    rows = [term_row(k.replace(' | U20', ''), v) for k, v in S['transfer'].items() if k != 'overlap' and k.endswith('U20')]
    L += [md_table(pd.DataFrame(rows)), '']
    ov = S['transfer'].get('overlap', {})
    if ov:
        o = pd.DataFrame(ov).T.reset_index().rename(columns={'index': 'data'}); o['Corpus'] = o.data.map(NAMES)
        o = o[['Corpus', 'e', 'e100', 'recall', 'precision', 'jaccard']]; o.columns = ['Corpus', 'Edges of G', 'Edges of G100', 'Recall', 'Precision', 'Jaccard']
        L += ['Overlap of the subsample graph with *G*~100~ on the units of Table S15.4:', '', md_table(o, '{:.3f}'), '']
    L += ['## S15.5 PDKT: the graph in the training loss', '', '**Table S15.5** Null-graph ladder inside PDKT (Eq. 5, λ = 1, at most 512 edges per update), 10%, update budget. pdkt_empty is DKT; the last row compares DKT with the empty-graph probe.', '']
    rows = [term_row(k, v) for k, v in S['pdkt'].items()]
    L += [md_table(pd.DataFrame(rows)), '']
    L += ['## S15.6 Longer histories', '']
    for k, v in S['cap500'].items():
        rows = [term_row('500 interactions per learner', {'first': v['len500'], 'all': None}), term_row('200 interactions, same units', {'first': v['len200'], 'all': None})]
        L += [f"**Table S15.6** {k}, embedding refinement, 10%; units: " + ', '.join(f"{NAMES.get(c, c)} s{s}" for c, s in v['units']) + '.', '', md_table(pd.DataFrame(rows)), '']
    if 'xes_single_kc' in S:
        L += ['## S15.7 XES3G5M without multi-KC questions', '', S['xes_single_kc_text'], '']
    # S16 -----------------------------------------------------------------------------------------------------------
    if D:
        L += ['# S16 Next-KC decisions', '', '## S16.1 Agreement by corpus and rule', '',
              '**Table S16.1** Share of decisions in which two models choose the same next KC, 25% of the training learners, subsample 0; mean over the pairs of seeds 0, 100, 200. Near-ties: mean number of other candidates within 0.01 of the best value of the control (seed 0) under the target rule.', '']
        rows = []
        for c, v in D['per_corpus'].items():
            a = v['agreement']; rows.append({'Corpus': NAMES[c], 'Candidates (median)': v['median_candidates'], 'Near-ties': v['near_ties_0p01'],
                                             **{f'{p} | {rl}': a[pk][rl] for pk, p in (('control_seeds', 'C seeds'), ('graph_control_crossed', 'G-C crossed'), ('graph_control_shared', 'G-C shared'))
                                                for rl in ('target', 'target_tb0.01', 'target_tb0.02', 'easiest')}})
        L += [md_table(pd.DataFrame(rows), '{:.3f}'), '']
        L += ['## S16.2 Regret against first-attempt AUC', '',
              '**Table S16.2** Regression of regret on first-attempt AUC over every decision model (5%, 25%, 100%; corpus fixed effects; standard errors clustered by corpus).', '']
        rows = [{'Simulator | rule': k, **{kk: vv for kk, vv in v.items()}} for k, v in D['regret_vs_auc'].items()]
        L += [md_table(pd.DataFrame(rows), '{:.4f}'), '']
        R = pd.read_csv(f'{RES}/decisions_rows.csv')
        w = R.groupby(['data', 'sim', 'frac']).agg(models=('model', 'size'), auc_first=('auc_first', 'mean'), regret_target=('regret_target', 'mean'),
                                                     regret_tb=('regret_target_tb0.01', 'mean'), regret_easiest=('regret_easiest', 'mean')).reset_index()
        w['data'] = w.data.map(NAMES); w.columns = ['Corpus', 'Simulator', 'Training %', 'Models', 'First-att. AUC', 'Regret, target', 'Regret, stabilized 0.01', 'Regret, easiest']
        L += ['**Table S16.3** Mean regret by corpus, simulator and training size of the decision models.', '', md_table(w, '{:.4f}'), '']
        L += ['Weights of the learning simulator (log(1 + successes), log(1 + failures), smoothed proportion correct) by corpus: ' + '; '.join(f"{NAMES[c]} " + ', '.join(f'{x:+.3f}' for x in v['pfa']['w']) for c, v in D['per_corpus'].items()) + '.', '']
    open(f'{RES}/online_resource_4.md', 'w').write('\n'.join(L))
    print('written', len(L))


if __name__ == '__main__': main()
