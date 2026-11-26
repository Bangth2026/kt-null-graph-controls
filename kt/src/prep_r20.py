"""Release 6.0.0: the corpora of prep.py with the question (item) of every interaction kept, plus DBE-KT22.

For every corpus of prep.py whose release names an item below the KC, the loader below is prep.py's loader
with one more column, `q`, the item identifier. prep.finish is replaced by finish_q, which applies exactly the
same filtering, learner draw (default_rng(2026), one process per corpus), sorting and KC index as prep.finish
and writes two files:
    data/<name>.parquet     identical to the file prep.py writes (user, kc, kc_name, correct, order)
    data/<name>_q.parquet   the same rows in the same order with `q` (integer item index) and `q_name`
The KC-level file is checked against a file written earlier by prep.py when one exists (--check).

Items: ASSISTments 2009 and 2012 problem_id; ASSISTments 2017 problemId; Eedi QuestionId; EdNet question_id;
Algebra 2005, Bridge to Algebra 2006 and Statics 2011 the pair (Problem Name, Step Name). ASSISTments 2015 and
Junyi name no item below the KC and are not prepared here.

DBE-KT22 (Abdelrahman et al. 2022; public copy Unggi/dbe-kt22_raw_data on Hugging Face) is new in this release.
Transaction.csv gives student, question, answer_state (correct/incorrect) and start_time; Question_KC_Relationships
maps each question to one or more KCs; KC_Relationships is the instructor-authored relation between KCs. As for
XES3G5M (prep_r14.py), each answered question keeps its first KC (lowest KC id among its tags); the share of
multi-KC questions is reported by this script. The expert graph is written to data/dbe_kt22_expert.json as a
list of directed pairs (from_kc_name, to_kc_name) as given in the release.
Usage: python prep_r20.py <a09|a12|a17|eedi|ednet|alg05|bridge06|statics|dbe> [--check]
"""
import sys, os, json, zipfile, ast, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep
R, O = prep.R, prep.O


def finish_q(df, name, meta=None):
    df = df.dropna(subset=['user', 'kc', 'correct', 'order'])
    df['correct'] = df['correct'].astype(int).clip(0, 1)
    df = df.sort_values(['user', 'order'], kind='stable')
    n = df.groupby('user').size(); keep = n[n >= prep.MINI].index
    if len(keep) > prep.MAXU: keep = prep.RNG.choice(np.array(keep), prep.MAXU, replace=False)
    df = df[df.user.isin(set(keep))].copy()
    kcs = sorted(df.kc.astype(str).unique()); kmap = {k: i for i, k in enumerate(kcs)}
    df['kc_name'] = df.kc.astype(str); df['kc'] = df.kc_name.map(kmap)
    df['user'] = df.user.astype(str)
    df['q_name'] = df['q'].astype(str); qs = sorted(df.q_name.unique()); qmap = {k: i for i, k in enumerate(qs)}
    df['q'] = df.q_name.map(qmap)
    df = df[['user', 'kc', 'kc_name', 'correct', 'order', 'q', 'q_name']].reset_index(drop=True)
    kc_only = df[['user', 'kc', 'kc_name', 'correct', 'order']]
    path = f'{O}/{name}.parquet'
    if '--check' in sys.argv and os.path.exists(path):
        old = pd.read_parquet(path)
        same = old.reset_index(drop=True).equals(kc_only.reset_index(drop=True))
        print(name, 'identical to the existing KC-level file:', same)
        if not same: raise SystemExit('KC-level file differs')
    else:
        kc_only.to_parquet(path)
    df.to_parquet(f'{O}/{name}_q.parquet')
    if meta is not None:
        json.dump({k: meta.get(k) for k in kcs}, open(f'{O}/{name}_groups.json', 'w'))
    m = {'name': name, 'interactions': len(df), 'learners': df.user.nunique(), 'kcs': len(kcs), 'items': len(qs),
         'correct_rate': float(df.correct.mean())}
    print(m); return m


prep.finish = finish_q


def a09():
    d = pd.read_csv(f'{R}/a09/skill_builder_data_corrected.csv', encoding='latin-1', low_memory=False,
                    usecols=['order_id', 'user_id', 'skill_id', 'correct', 'original', 'problem_id'])
    d = d[d.original == 1].dropna(subset=['skill_id']).drop_duplicates(['user_id', 'order_id'])
    return finish_q(pd.DataFrame({'user': d.user_id, 'kc': d.skill_id.astype(int), 'correct': d.correct, 'order': d.order_id,
                                  'q': d.problem_id}), 'assist09')


def a12():
    parts = []
    for c in pd.read_csv(f'{R}/a12/2012-2013-data-with-predictions-4-final.csv',
                         usecols=['problem_log_id', 'user_id', 'skill_id', 'correct', 'original', 'problem_id'], chunksize=2_000_000):
        c = c[(c.original == 1)].dropna(subset=['skill_id']); c = c[c.correct.isin([0, 1])]; parts.append(c)
    d = pd.concat(parts)
    return finish_q(pd.DataFrame({'user': d.user_id, 'kc': d.skill_id.astype(int), 'correct': d.correct, 'order': d.problem_log_id,
                                  'q': d.problem_id}), 'assist12')


def a17():
    d = pd.read_csv(f'{R}/a17/anonymized_full_release_competition_dataset.csv',
                    usecols=['studentId', 'skill', 'correct', 'original', 'startTime', 'action_num', 'problemId'], low_memory=False)
    d = d[d.original == 1].dropna(subset=['skill'])
    return finish_q(pd.DataFrame({'user': d.studentId, 'kc': d.skill, 'correct': d.correct, 'order': d.startTime * 1e4 + d.action_num % 10000,
                                  'q': d.problemId}), 'assist17')


def _pslc(path, kccol, name, unit_level, corr):
    cols = ['Row', 'Anon Student Id', kccol, corr, 'Problem Hierarchy', 'Problem Name', 'Step Name']
    kw = dict(sep='\t') if path.endswith('.txt') else dict(encoding='utf-8-sig')
    d = pd.read_csv(path, usecols=cols, **kw)
    d = d.dropna(subset=[kccol])
    if corr == 'First Attempt':
        d = d[d['First Attempt'].isin(['correct', 'incorrect', 'hint'])]; c = (d['First Attempt'] == 'correct').astype(int)
    else:
        c = d[corr]
    d['kc'] = d[kccol].str.split('~~').str[0]
    unit = d.groupby('kc')['Problem Hierarchy'].agg(lambda s: s.astype(str).str.split(',').str[unit_level].mode().iloc[0]).to_dict()
    q = d['Problem Name'].astype(str) + '||' + d['Step Name'].astype(str)
    return finish_q(pd.DataFrame({'user': d['Anon Student Id'], 'kc': d.kc, 'correct': c, 'order': d.Row, 'q': q}), name, meta=unit)


def alg05():
    return _pslc(f'{R}/alg05/algebra_2005_2006/algebra_2005_2006_train.txt', 'KC(Default)', 'algebra05', 0, 'Correct First Attempt')


def bridge06():
    return _pslc(f'{R}/bridge/bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_train.txt', 'KC(SubSkills)', 'bridge06', 0, 'Correct First Attempt')


def statics():
    return _pslc(f'{R}/statics/AllData_student_step_2011F.csv', 'KC (F2011)', 'statics11', 1, 'First Attempt')


def eedi():
    t = pd.read_csv(f'{R}/eedi/train_data/train_task_3_4.csv'); a = pd.read_csv(f'{R}/eedi/metadata/answer_metadata_task_3_4.csv', usecols=['AnswerId', 'DateAnswered'])
    q = pd.read_csv(f'{R}/eedi/metadata/question_metadata_task_3_4.csv'); s = pd.read_csv(f'{R}/eedi/metadata/subject_metadata.csv'); s.columns = [c.strip('﻿') for c in s.columns]
    lvl = dict(zip(s.SubjectId, s.Level)); par = dict(zip(s.SubjectId, s.ParentId))
    def leaf(lst):
        l = ast.literal_eval(lst); l3 = [x for x in l if lvl.get(x) == 3]; return l3[0] if l3 else None
    q['kc'] = q.SubjectId.map(leaf); qk = dict(zip(q.QuestionId, q.kc))
    t = t.merge(a, on='AnswerId', how='left'); t['kc'] = t.QuestionId.map(qk); t = t.dropna(subset=['kc', 'DateAnswered'])
    t['order'] = pd.to_datetime(t.DateAnswered).astype('int64') * 1e-9 * 1000 + t.AnswerId % 1000
    grp = {str(int(k)): str(par.get(int(k))) for k in t.kc.unique()}
    t['kc'] = t.kc.astype(int).astype(str)
    return finish_q(pd.DataFrame({'user': t.UserId, 'kc': t.kc, 'correct': t.IsCorrect, 'order': t.order, 'q': t.QuestionId}), 'eedi', meta=grp)


def ednet():
    q = pd.read_csv(f'{R}/ednet/contents/questions.csv'); ans = dict(zip(q.question_id, q.correct_answer))
    tag = dict(zip(q.question_id, q.tags.astype(str).str.split(';').str[0])); part = dict(zip(q.question_id, q.part))
    z = zipfile.ZipFile(f'{R}/EdNet-KT1.zip'); names = [n for n in z.namelist() if n.endswith('.csv')]
    pick = prep.RNG.choice(len(names), 20000, replace=False); rows = []
    for i in pick:
        d = pd.read_csv(z.open(names[i]))
        if len(d) < prep.MINI: continue
        d['user'] = names[i][4:-4]; rows.append(d)
    d = pd.concat(rows)
    d['correct'] = (d.user_answer == d.question_id.map(ans)).astype(int); d['kc'] = d.question_id.map(tag)
    tagpart = {}
    for qid, t in tag.items(): tagpart.setdefault(t, part[qid])
    return finish_q(pd.DataFrame({'user': d.user, 'kc': d.kc, 'correct': d.correct, 'order': d.timestamp * 1000 + d.solving_id % 1000,
                                  'q': d.question_id}), 'ednet', meta={k: str(v) for k, v in tagpart.items()})


def dbe():
    P = f'{R}/dbe'
    t = pd.read_csv(f'{P}/Transaction.csv', low_memory=False)
    qk = pd.read_csv(f'{P}/Question_KC_Relationships.csv'); kcs = pd.read_csv(f'{P}/KCs.csv'); rel = pd.read_csv(f'{P}/KC_Relationships.csv')
    print('columns', list(t.columns)[:20], list(qk.columns), list(kcs.columns), list(rel.columns))
    qcol = 'question_id'; kcol = 'knowledgecomponent_id'
    first = qk.groupby(qcol)[kcol].min(); nk = qk.groupby(qcol)[kcol].nunique()
    print('questions', len(nk), 'multi-KC questions', int((nk > 1).sum()), round(float((nk > 1).mean()), 3))
    name = dict(zip(kcs['id'], kcs['name']))
    t = t[t[qcol].isin(first.index)].copy()
    st = t['answer_state'].astype(str).str.lower()
    t = t[st.isin(['true', 'false', '1', '0', 'correct', 'incorrect'])]; st = t['answer_state'].astype(str).str.lower()
    c = st.isin(['true', '1', 'correct']).astype(int)
    order = pd.to_datetime(t['start_time'], errors='coerce', utc=True).astype('int64') // 10**6 * 1000 + (t['id'] % 1000 if 'id' in t else 0)
    kc = t[qcol].map(first).map(lambda k: str(name.get(k, k)))
    multi = t[qcol].map(nk) > 1
    print('answers on multi-KC questions', round(float(multi.mean()), 3))
    pairs = [(str(name.get(a, a)), str(name.get(b, b))) for a, b in zip(rel.iloc[:, 1], rel.iloc[:, 2])]
    json.dump({'columns': list(rel.columns), 'pairs': pairs}, open(f'{O}/dbe_kt22_expert.json', 'w'))
    return finish_q(pd.DataFrame({'user': t['student_id'], 'kc': kc, 'correct': c, 'order': order, 'q': t[qcol]}), 'dbe_kt22')


if __name__ == '__main__':
    out = {}
    for f in [x for x in sys.argv[1:] if not x.startswith('--')]: out[f] = globals()[f]()
    json.dump(out, open(f'{O}/stats_r20_{"_".join(x for x in sys.argv[1:] if not x.startswith("--"))}.json', 'w'), indent=1)
