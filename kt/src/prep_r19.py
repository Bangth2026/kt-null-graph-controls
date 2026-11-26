"""Release 5.0.0: XES3G5M without multi-KC questions (reviewer minor comment 10).

As prep_r14.xes3g5m, except that every answer to a question tagged with more than one KC is dropped instead of being
kept with its first KC (1,023 of 7,652 questions, 12.5% of the answer rows of the release). Same learner rules; the learners
are those of data/xes3g5m.parquet (prep_r14.py), so training, validation and test learners match the replication corpus. Output: data/xes3g5m1.parquet.
Usage: python prep_r19.py
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import prep
R = prep.R


def xes3g5m1():
    d = pd.concat([pd.read_parquet(f'{R}/xes/train.parquet'), pd.read_parquet(f'{R}/xes/test.parquet')])
    rows = []
    for uid, qs, cs, rs, ts, sm, rp in zip(d.uid, d.questions, d.concepts, d.responses, d.timestamps, d.selectmasks, d.is_repeat):
        qs = np.asarray(qs); cs = np.asarray(cs); rs = np.asarray(rs); ts = np.asarray(ts); sm = np.asarray(sm); rp = np.asarray(rp)
        nxt = np.r_[rp[1:], 0]
        k = (sm == 1) & (rp == 0) & (cs >= 0) & (rs >= 0) & (nxt == 0)      # first KC row not followed by a further KC row
        rows.append(pd.DataFrame({'user': uid, 'kc': cs[k], 'correct': rs[k], 't': ts[k], 'i': np.arange(len(cs))[k]}))
    keep = set(pd.read_parquet(f'{prep.O}/xes3g5m.parquet').user.astype(str).unique())
    d = pd.concat(rows); d = d[d.user.astype(str).isin(keep)].drop_duplicates(['user', 't', 'kc', 'correct'])
    d = d.sort_values(['user', 't', 'i'], kind='stable'); d['order'] = np.arange(len(d))
    return prep.finish(d[['user', 'kc', 'correct', 'order']], 'xes3g5m1')


if __name__ == '__main__': xes3g5m1()
