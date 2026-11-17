"""Collect read-only request billing records, distinguishing actual charges from estimates."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import getpass
import json
import os
from typing import Any
import pandas as pd
from openai import APIError
from openrouter_client import make_client
from synthetic_pilot import OUT, key_usage, save_json, stamp


def collect():
    manifest=pd.read_csv(OUT/'manifest.csv',keep_default_na=False)
    def one(row):
        result={'sample_id':row['sample_id'],'generation_id':row.get('generation_id'),'cost_status':'unavailable'}
        try:
            with make_client(timeout=30) as client:
                response=client.get('/generation',cast_to=dict[str,Any],options={'params':{'id':row['generation_id']}})
            data=response.get('data',{})
            if data.get('total_cost') is not None:
                result.update(cost_status='reported',total_cost_usd=float(data['total_cost']),billing_model=data.get('model'),provider=data.get('provider_name'))
        except APIError as error: result['http_status']=getattr(error,'status_code',None)
        except Exception as error: result['error_type']=type(error).__name__
        return result
    # Probe compatibility before querying the whole speech ledger.
    probe=one(manifest.iloc[0].to_dict())
    costs=[probe]
    if probe['cost_status']=='reported':
        with ThreadPoolExecutor(max_workers=3) as pool: costs.extend(pool.map(one,manifest.iloc[1:].to_dict('records')))
    frame=pd.DataFrame(costs); frame.to_csv(OUT/'generation_costs.csv',index=False,encoding='utf-8-sig')
    reviews=json.loads((OUT/'automated_review.json').read_text(encoding='utf-8')) if (OUT/'automated_review.json').exists() else []
    review_costs=[r.get('review_usage',{}).get('cost') for r in reviews if isinstance(r.get('review_usage'),dict)]
    with make_client(timeout=30) as client: usage=key_usage(client)
    ledger=json.loads((OUT/'budget.json').read_text(encoding='utf-8'))
    summary={'checked_utc':stamp(),'speech_requests_with_cost':int(frame.cost_status.eq('reported').sum()),
             'speech_requests_total':len(manifest),'reported_speech_cost_usd':float(frame.get('total_cost_usd',pd.Series(dtype=float)).sum()),
             'audio_reviews_with_cost':sum(v is not None for v in review_costs),'reported_review_cost_usd':sum(float(v) for v in review_costs if v is not None),
             'latest_key_usage_delta_usd':usage-ledger['baseline_key_usage_usd'],
             'billing_complete':bool(len(frame)==len(manifest) and frame.cost_status.eq('reported').all() and len(review_costs)==len(reviews)),
             'key_usage_note':'Key usage may lag and fluctuate between cached snapshots; per-request records are preferred.'}
    summary['reported_request_cost_sum_usd']=summary['reported_speech_cost_usd']+summary['reported_review_cost_usd']
    save_json(OUT/'billing_summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--hidden-key',action='store_true'); args=parser.parse_args()
    if args.hidden_key: os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter credential (hidden): ')
    collect()
