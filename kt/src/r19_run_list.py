"""Release 5.0.0: list of the training runs added for the UMUAI revision (runs_r19), for Online Resource 2.
Writes results_r19/run_list_r19.csv (one row per run file: tag, corpus, size, subsample, seed, condition, family, cap,
device, test AUC, first-attempt AUC, updates, seconds) and prints counts by family."""
import os, json, glob, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for f in sorted(glob.glob(f'{REPO}/runs_r19/*.json') + glob.glob(f'{REPO}/runs_r19/repro/*.json')):
    r = json.load(open(f))
    rows.append(dict(file=os.path.relpath(f, REPO), tag=r['tag'], data=r['data'], frac=r['frac'], sub=r['sub'], seed=r.get('seed'), cond=r['cond'],
                     family=r.get('family'), maxlen=r.get('maxlen', 200), repro='/repro/' in f, device=r.get('device'),
                     test_auc=r['test_auc'], test_auc_first=r['test_auc_first'], steps=r.get('steps'), secs=round(r.get('secs_total', 0), 1)))
df = pd.DataFrame(rows); df.to_csv(f'{REPO}/results_r19/run_list_r19.csv', index=False)
def kind(r):
    if r.repro: return 'reproduction of archived runs'
    if r['cond'].startswith('pdkt'): return 'PDKT ladder'
    if '100' in r['cond']: return 'transfer of structure (G100)'
    if r.maxlen == 500: return '500-interaction cap'
    if r['data'] == 'xes3g5m1': return 'XES3G5M without multi-KC questions'
    if r.frac == 0.05: return '5% probes with candidate dumps (decisions)'
    return 'extra training seeds'
df['kind'] = df.apply(kind, axis=1); print(df.kind.value_counts().to_string()); print('total', len(df))
