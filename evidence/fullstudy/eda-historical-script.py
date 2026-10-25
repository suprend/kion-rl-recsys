import csv
import json
from collections import Counter
from pathlib import Path

root = Path('D:/tlab-recsys-2026/data/raw/kion/full')
users, items, dates, pct = Counter(), Counter(), Counter(), Counter()
missing, invalid, total, duration_min = 0, 0, 0, None
for row in csv.DictReader((root / 'interactions.csv').open(encoding='utf-8', newline='')):
    total += 1
    users[int(row['user_id'])] += 1
    items[int(row['item_id'])] += 1
    dates[row['last_watch_dt']] += 1
    value = row['watched_pct']
    if not value:
        missing += 1
    else:
        p = float(value)
        pct[p] += 1
        invalid += int(not 0 <= p <= 100)
    d = int(row['total_dur'])
    duration_min = d if duration_min is None else min(duration_min, d)

def quantile(hist, q):
    n = sum(hist.values())
    pos = (n - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, n - 1)
    a = b = None
    count = 0
    for value, size in sorted(hist.items()):
        if a is None and count + size > lo:
            a = value
        if count + size > hi:
            b = value
            break
        count += size
    return a + (b - a) * (pos - lo)

summary = {'interactions': total, 'users_in_interactions': len(users), 'items_in_interactions': len(items), 'date_min': min(dates), 'date_max': max(dates), 'watched_pct_missing': missing, 'watched_pct_invalid': invalid, 'watched_pct_mean': sum(v*n for v,n in pct.items()) / sum(pct.values()), 'watched_pct_quantiles': {str(q):quantile(pct,q) for q in [0,.1,.25,.5,.75,.9,.99,1]}, 'watched_pct_zero_share': pct[0]/total, 'watched_pct_100_share': pct[100]/total, 'history_length_quantiles': {str(q):quantile(Counter(users.values()),q) for q in [.25,.5,.75,.9,.99]}, 'users_ge_5_events': sum(n >= 5 for n in users.values()), 'users_ge_2_events': sum(n >= 2 for n in users.values()), 'total_dur_min': duration_min}
for name,key,observed in [('users','user_id',users),('items','item_id',items)]:
    ids = []
    with (root / (name+'.csv')).open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            ids.append(int(row[key]))
    summary[name+'_table_rows'] = len(ids)
    summary[name+'_table_duplicate_ids'] = len(ids)-len(set(ids))
    summary[name+'_orphan_ids'] = len(set(observed)-set(ids))
summary['daily_counts'] = dict(sorted(dates.items()))
out = Path('D:/tlab-recsys-2026/artifacts/task2-kion')
out.mkdir(exist_ok=True,parents=True)
(out/'eda.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='daily_counts'},indent=2),flush=True)
