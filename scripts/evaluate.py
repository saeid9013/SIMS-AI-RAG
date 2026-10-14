"""Run local retrieval evaluation against an already-ingested private corpus.
Input JSONL: {"question":"...","expected_sources":[{"name":"...","location":"صفحه 2"}]}
Multiple expected sources measure fraction of relevant source locations retrieved.
"""
import argparse
import json
import statistics
import time
from app.db import init_db
from app.retrieval import search

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('--k',type=int,default=5)
    args=p.parse_args(); init_db()
    recalls=[];times=[];details=[]
    with open(args.dataset,encoding='utf-8') as f:
        for line in f:
            item=json.loads(line)
            expected={(s['name'],s['location']) for s in item['expected_sources']}
            if not expected:
                raise ValueError('Each case needs at least one expected source; evaluate refusal separately.')
            started=time.perf_counter();rows=search(item['question'],args.k);times.append(time.perf_counter()-started)
            got={(r['name'],r['location']) for r in rows}
            recall=len(got & expected)/len(expected);recalls.append(recall)
            details.append({'question':item['question'],'recall':recall,'retrieved':list(got)})
    if not recalls:
        raise ValueError('Empty evaluation dataset')
    output={'cases':len(recalls),'k':args.k,'mean_recall':statistics.mean(recalls),
            'mean_seconds':statistics.mean(times),'max_seconds':max(times),'details':details}
    print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
