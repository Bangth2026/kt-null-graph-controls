"""Release 6.0.0: Online Resource 5 of the second UMUAI revision (markdown), from results_r20/analysis.json,
results_r20/closed_loop_<world>.json and the run files. Every value is read from those files."""
import os, json, glob, csv
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
A = json.load(open(f'{REPO}/results_r20/analysis.json'))
NAMES = {'assist09': 'ASSISTments 2009', 'assist12': 'ASSISTments 2012', 'assist15': 'ASSISTments 2015', 'assist17': 'ASSISTments 2017',
         'junyi': 'Junyi Academy', 'algebra05': 'Algebra 2005', 'eedi': 'Eedi', 'ednet': 'EdNet KT1', 'bridge06': 'Bridge to Alg. 2006',
         'statics11': 'Statics 2011', 'dbe_kt22': 'DBE-KT22'}


def f(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)): return '--'
    return ('+' if x >= 0 else '−') + f'{abs(x):.{nd}f}'


def md(df):
    cols = list(df.columns); out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def summary_row(name, o):
    if not o: return None
    m = o.get('mixed') or {}
    return {'Term': name, 'Units': o['units'], 'Mean': f(o['mean']), '>0': f"{o['pos']}/{o['n']}",
            '95% interval': f"{f(m.get('lo'))}, {f(m.get('hi'))}" if m else '--', 'Fit': m.get('kind', '--'),
            'Bootstrap 95%': f"{f(o['boot'][0])}, {f(o['boot'][1])}" if o.get('boot') else '--',
            'Wilcoxon p': f"{o['p_wilcoxon']:.3f}" if 'p_wilcoxon' in o else '--'}


def per_corpus(block, keys, col='test_auc_first'):
    rows = []
    corp = sorted({c for k in keys for c in ((A[block][k] or {}).get(col) or {}).get('per_corpus', {})})
    for c in corp:
        r = {'Corpus': NAMES.get(c, c)}
        for k, lab in keys.items():
            v = ((A[block][k] or {}).get(col) or {}).get('per_corpus', {}).get(c)
            r[lab] = f(v) if v is not None else '--'
        rows.append(r)
    return pd.DataFrame(rows)


L = ['---', 'title: "Online Resource 5: NCA-KT, the item-level ladder, DBE-KT22 and the simulated worlds"',
     'subtitle: "When does a concept graph help a knowledge tracer? Null-graph controls, null-calibrated gating and closed-loop tests for graph-based learner models"',
     'author: "Hai Bang Truong, Saigon International University (truonghaibang@siu.edu.vn)"', '---', '',
     'Every value below is computed by the scripts of release 6.0.0 of the accompanying code (`kt/src/r20_*.py`) from the run files of `kt/runs_r20` and `kt/runs_r20_sim` and of the earlier releases; the result files are `kt/results_r20/*.json`. Section numbers continue those of Online Resources 1, 3 and 4. Differences are test AUC of the first condition minus the second, on the same subsample and seed. Intervals: mixed model with a random intercept per corpus (mixed) or ordinary least squares over units where a corpus contributes one unit (ols); corpus-clustered bootstrap; Wilcoxon signed-rank test over corpus means. All terms are exploratory.', '']
# S17 ------------------------------------------------------------------------------------------------
L += ['# S17 NCA-KT', '', '## S17.1 Summary of every term, first attempts and all predictions', '']
for col, lab in (('test_auc_first', 'first attempts'), ('test_auc', 'all predictions')):
    rows = [summary_row(k, (v or {}).get(col)) for k, v in A['nca'].items() if isinstance(v, dict) and 'test_auc_first' in v]
    rows = [r for r in rows if r]
    L += [f'**Table S17.{1 if col == "test_auc_first" else 2}** NCA-KT terms, {lab}', '', md(pd.DataFrame(rows)), '']
L += ['## S17.2 By corpus (first attempts)', '', '**Table S17.3** NCA-KT terms by corpus, mean over the units of each corpus', '',
      md(per_corpus('nca', {'nca-empty': 'NCA − ∅', 'nca-support_emb': 'NCA − G (emb.)', 'nca-nca_dp (edges)': 'NCA(G) − NCA(DP)',
                            'nca-nca_static': '− mastery cond.', 'nca-nca_nonull': '− calibration', 'nca-nca_nogate': '− gate',
                            'nca-nca_nomargin': '− margin', 'empty_fa-empty': 'Reweighting − ∅', 'nca-empty_fa': 'NCA − reweighting'})), '']
L += ['**Table S17.4** With all training learners, by corpus (first attempts)', '',
      md(per_corpus('nca', {'100: nca-empty': 'NCA − ∅', '100: nca-support': 'NCA − G', '100: support-empty': 'G − ∅',
                            '100: empty_fa-empty': 'Reweighting − ∅', '100: nca-empty_fa': 'NCA − reweighting'})), '']
if 'alpha' in A['nca']:
    d = pd.DataFrame(A['nca']['alpha'])
    d = d.assign(**{c: d[c].map(lambda x: f'{x:.3f}') for c in ('alpha', 'first', 'repeat')})
    L += ['**Table S17.5** Mean gate value alpha at the target, by condition and training size (test predictions; first and repeated attempts)', '', md(d), '']
# S18 ------------------------------------------------------------------------------------------------
L += ['# S18 The item-level ladder', '']
for i, (col, lab) in enumerate((('test_auc_first', 'first attempts at a KC'), ('test_auc_first_item', 'first attempts at an item'), ('test_auc', 'all predictions'))):
    rows = [summary_row(k, (v or {}).get(col)) for k, v in A['item'].items() if k not in ('levels',) and isinstance(v, dict)]
    rows = [r for r in rows if r]
    L += [f'**Table S18.{i + 1}** Item-level terms, {lab}. Conditions: q_kc KC table only; q_id item table only; q_qm item table and Q-matrix; q_rqm item table and size-preserving random Q-matrix; q_qm_g, q_qm_dp Q-matrix with the support graph or its rewiring; q_diff KC table and item difficulty.', '', md(pd.DataFrame(rows)), '']
lv = pd.DataFrame(A['item']['levels'])
for col, lab in (('first', 'first attempts at a KC'), ('first_item', 'first attempts at an item'), ('all', 'all predictions')):
    t = lv.pivot(index='data', columns='cond', values=col).reset_index(); t['data'] = t['data'].map(lambda c: NAMES.get(c, c))
    for c in t.columns[1:]: t[c] = t[c].map(lambda x: f'{x:.4f}' if pd.notna(x) else '--')
    L += [f'**Table S18.{4 + ["first", "first_item", "all"].index(col)}** Test AUC by corpus and condition, {lab} (mean over the subsamples run)', '', md(t.rename(columns={'data': 'Corpus'})), '']
u = lv.groupby('data').unseen.mean().reset_index(); u['data'] = u['data'].map(lambda c: NAMES.get(c, c)); u['unseen'] = u['unseen'].map(lambda x: f'{100 * x:.1f}%')
L += ['**Table S18.7** Share of test answers on items that no training learner of the 10% subsample answered', '', md(u.rename(columns={'data': 'Corpus', 'unseen': 'Unseen items'})), '']
# S19 ------------------------------------------------------------------------------------------------
L += ['# S19 DBE-KT22', '', 'Five subsamples at 10% of the training learners, one seed each; *t* intervals over the five subsamples. The instructor relations of the release (KC_Relationships.csv) list every pair in both directions and are oriented here from the KC the training learners meet earlier to the one they meet later. The generated graph (data/dbe_kt22_llm.json) has 87 directed prerequisite pairs among the 59 KCs of the prepared corpus.', '']
rows = []
for k, v in A['dbe'].items():
    for col, lab in (('test_auc_first', 'first'), ('test_auc', 'all')):
        o = (v or {}).get(col)
        if o and 'units' in o and o.get('t'):
            rows.append({'Term': k, 'Outcome': lab, 'Units': o['units'], 'Mean': f(o['mean']), '>0': f"{o['pos']}/{o['units']}", '95% t interval': f"{f(o['t'][0])}, {f(o['t'][1])}"})
L += ['**Table S19.1** DBE-KT22 terms', '', md(pd.DataFrame(rows)), '']
runs = []
for fp in sorted(glob.glob(f'{REPO}/runs_r20/dbe_kt22__f10__*.json')):
    r = json.load(open(fp)); runs.append({'Subsample': r['sub'], 'Condition': r['cond'], 'Test AUC': f"{r['test_auc']:.4f}", 'First att.': f"{r['test_auc_first']:.4f}", 'Updates': r['steps']})
L += ['**Table S19.2** Every DBE-KT22 run at 10%', '', md(pd.DataFrame(runs).sort_values(['Condition', 'Subsample'])), '']
# S20 ------------------------------------------------------------------------------------------------
L += ['# S20 Simulated worlds and the closed loop', '', 'World parameters are in data/<world>_world.npz and the true graphs in data/<world>_true.json (kt/src/r20_sim.py). Training: 3,000 simulated learners, the hash split of the study (2,100 training learners), subsample 0 at 10% and all training learners. Closed loop: 500 new learners (seed 777), 20 warm-up steps, 40 adaptive steps. The score check is the largest absolute difference between the all-KC scores used by the policies and the model\'s own forward pass on three random histories (it is at most 1e-7 for every model).', '']
for w in ('sim_dbe', 'sim_a09'):
    S = A['sim'][w]; lab = 'DBE-KT22 world' if w == 'sim_dbe' else 'ASSISTments 2009 world'
    t = pd.DataFrame([{'Size and condition': k, 'Seeds': v['n'], 'First att.': f"{v['first']:.4f}", 'All pred.': f"{v['all']:.4f}"} for k, v in S['auc'].items()])
    L += [f'**Table S20.{1 if w == "sim_dbe" else 2}** {lab}: test AUC by condition (mean over seeds)', '', md(t), '']
    t = pd.DataFrame([{'Term': k, 'First att.': f(v['first']), 'All pred.': f(v['all']), 'Per seed (first)': ', '.join(f(x) for x in v['reps'])} for k, v in S['terms'].items()])
    L += [f'**Table S20.{3 if w == "sim_dbe" else 4}** {lab}: prediction terms', '', md(t), '']
    cl = S.get('closed_loop')
    if cl:
        ref = pd.DataFrame([{'Policy': k, **{m: f'{v[m]:.3f}' for m in ('gain', 'ttm5', 'reached5', 'frustration', 'success', 'regret')}} for k, v in cl['reference'].items()])
        L += [f'**Table S20.{5 if w == "sim_dbe" else 6}** {lab}: reference policies (gain in KCs; ttm5 steps to five new masteries, 41 if never; reached5 share of learners reaching five; frustration; success rate; regret of the target rule under the true model)', '', md(ref), '']
        rows = []
        for k, v in cl['policies'].items():
            if not isinstance(v, dict): continue
            rows.append({'Size:tracer:rule': k, **{m: (f'{v[m]:.3f}' if m in v else '--') for m in ('gain', 'ttm5', 'frustration', 'success', 'regret', 'first_choice_agreement', 'jaccard10')}, 'Seeds': v['n_seeds']})
        L += [f'**Table S20.{7 if w == "sim_dbe" else 8}** {lab}: tracer-driven policies (first_choice_agreement and jaccard10 between the two seeds of a tracer: same first adaptive choice; overlap of the first ten KCs practised)', '', md(pd.DataFrame(rows)), '']
        rows = [{'Pair:rule:outcome': k, 'Mean': f(v['mean'], 3), '95% interval': f"{f(v['lo'], 3)}, {f(v['hi'], 3)}"} for k, v in cl['pairs'].items()]
        L += [f'**Table S20.{9 if w == "sim_dbe" else 10}** {lab}: paired differences per learner (seed-averaged; interval over learners)', '', md(pd.DataFrame(rows)), '']
# S21 ------------------------------------------------------------------------------------------------
rows = []
for d in ('runs_r20', 'runs_r20_sim'):
    for fp in sorted(glob.glob(f'{REPO}/{d}/*.json')):
        r = json.load(open(fp)); rows.append({'set': d, 'tag': r['tag'], 'data': r['data'], 'frac': r['frac'], 'sub': r['sub'], 'seed': r['seed'], 'cond': r['cond'],
                                              'test_auc': r['test_auc'], 'test_auc_first': r['test_auc_first'], 'steps': r['steps'], 'secs': round(r.get('secs_total', 0))})
pd.DataFrame(rows).to_csv(f'{REPO}/results_r20/run_list_r20.csv', index=False)
cnt = pd.DataFrame(rows).groupby(['set', 'cond', 'frac']).size().reset_index(name='runs')
L += ['# S21 Runs of release 6.0.0', '', f'{len(rows)} training runs (kt/results_r20/run_list_r20.csv lists each with its tag, subsample, seed, test AUC and run time). By condition:', '', md(cnt), '']
open(f'{REPO}/results_r20/online_resource_5.md', 'w').write('\n'.join(L))
print('written', len(rows), 'runs')
