"""Blind audio-model review of synthetic samples; perceptual estimates, not demographics."""
from __future__ import annotations
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import getpass
import hashlib
import json
import os
import re
import unicodedata

import pandas as pd
from openai import APIError

from openrouter_client import OpenRouterError, make_client, _safe_api_error
from synthetic_pilot import OUT, BASE_URL, public_json, key_usage, budget_spend, save_json, stamp

REVIEW_MODEL='openai/gpt-audio-mini'
INSTRUCTION='''Listen to this short synthetic voice clip. Transcribe ONLY words actually audible, in the original language/script. The intended text and voice identity are intentionally withheld. Do not translate or guess missing speech.
Also describe auditory impressions of the fictional voice, not the speaker's actual biology, identity or real age. Use uncertainty if this short clip does not support a judgment. Never infer ethnicity, nationality, health or identity.
Return ONLY a JSON object with these fields:
transcript: string;
language: Hebrew, English, other, or uncertain;
speech_intelligibility: number 0 to 1;
perceived_voice_presentation: masculine, feminine, androgynous, or uncertain;
presentation_confidence: number 0 to 1 (subjective, not a calibrated probability);
perceived_age_band: childlike, young_adult_18_29, adult_30_44, mature_45_59, older_60_plus, or uncertain;
age_confidence: number 0 to 1 (subjective, not a calibrated probability);
notes: at most 25 words describing uncertainty or pronunciation issues.
Age ranges are perceptual casting categories, not age estimates of a real person. Do not force binary gender or invent an age. If silence or no intelligible speech, return empty transcript and uncertain categories.'''


def normalized(text):
    text=unicodedata.normalize('NFKC',str(text))
    return ' '.join(''.join(c for c in text if not unicodedata.category(c).startswith(('M','P'))).casefold().split())


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def annotate_one(row):
    result={'sample_id':row['sample_id'],'review_model':REVIEW_MODEL,'review_started_utc':stamp(),
            'review_status':'failed','annotation_provenance':'unvalidated_audio_model_perceptual_estimate',
            'human_verified':False,'review_prompt_sha256':hashlib.sha256(INSTRUCTION.encode()).hexdigest()}
    try:
        audio=(OUT/row['wav_path']).read_bytes()
        with make_client(timeout=90) as client:
            response=client.chat.completions.create(
                model=REVIEW_MODEL,modalities=['text'],temperature=0,max_tokens=400,
                messages=[{'role':'user','content':[
                    {'type':'text','text':INSTRUCTION},
                    {'type':'input_audio','input_audio':{'data':base64.b64encode(audio).decode('ascii'),'format':'wav'}}]}])
        if not response.choices or not response.choices[0].message.content: raise ValueError('Empty review response')
        if response.choices[0].finish_reason=='length': raise ValueError('Truncated review response')
        content=response.choices[0].message.content.strip()
        if content.startswith('```'): content=re.sub(r'^```(?:json)?\s*|\s*```$','',content)
        data=json.loads(content)
        allowed={
            'language':{'Hebrew','English','other','uncertain'},
            'perceived_voice_presentation':{'masculine','feminine','androgynous','uncertain'},
            'perceived_age_band':{'childlike','young_adult_18_29','adult_30_44','mature_45_59','older_60_plus','uncertain'}}
        for field,values in allowed.items():
            if data.get(field) not in values: raise ValueError('Unexpected review category: '+field)
        for field in ['speech_intelligibility','presentation_confidence','age_confidence']:
            value=data.get(field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not 0<=value<=1: raise ValueError('Invalid review confidence')
        if not isinstance(data.get('transcript'),str): raise ValueError('Missing transcript')
        transcript=normalized(data['transcript']); expected=normalized(row['text'])
        cer=edit_distance(expected,transcript)/max(1,len(expected))
        letters=[c for c in data['transcript'] if c.isalpha()]
        fraction=sum('\u05d0'<=c<='\u05ea' for c in letters)/max(1,len(letters))
        language_pass=data['language']=='Hebrew' and fraction>=.8 and cer<=.25 and data['speech_intelligibility']>=.7
        result.update(data)
        result.update(review_status='reviewed',review_generation_id=response.id,returned_review_model=response.model,
                      review_usage=response.usage.model_dump() if response.usage else None,
                      normalized_character_error_rate=cer,transcript_hebrew_letter_fraction=fraction,
                      language_qc_status='automatic_hebrew_text_match_pass' if language_pass else 'needs_human_review',
                      language_qc_pass=bool(language_pass),review_completed_utc=stamp(),
                      presentation_for_summary=data['perceived_voice_presentation'] if data['presentation_confidence']>=.6 else 'uncertain',
                      age_for_summary=data['perceived_age_band'] if data['age_confidence']>=.6 else 'uncertain')
    except Exception as error:
        result['error']=str(_safe_api_error(error)) if isinstance(error,APIError) else type(error).__name__+': '+str(error)[:150] if isinstance(error,ValueError) else type(error).__name__
        if isinstance(error,APIError) and getattr(error,'status_code',None) in {401,402,403}: result['fatal_auth_or_credit_error']=True
        if isinstance(error,APIError) and getattr(error,'status_code',None)==400:
            body=getattr(error,'body',{})
            detail=body.get('message','') if isinstance(body,dict) else ''
            if isinstance(body,dict) and isinstance(body.get('error'),dict): detail=body['error'].get('message',detail)
            metadata=body.get('metadata',{}) if isinstance(body,dict) else {}
            if not metadata and isinstance(body,dict) and isinstance(body.get('error'),dict): metadata=body['error'].get('metadata',{})
            raw=metadata.get('raw') if isinstance(metadata,dict) else None
            if raw:
                try:
                    decoded=json.loads(raw) if isinstance(raw,str) else raw
                    detail=decoded.get('error',{}).get('message',detail)
                except Exception: pass
            detail=str(detail).replace(os.environ.get('OPENROUTER_API_KEY','') or 'unused-secret-marker','[REDACTED]')
            detail=re.sub(r'sk-or-[A-Za-z0-9_-]+','[REDACTED]',detail)
            result['safe_provider_error_detail']=detail[:300]
    return result


def annotate(workers=3,max_requests=None,retry_failed=False):
    manifest=pd.read_csv(OUT/'manifest.csv',keep_default_na=False)
    catalog=public_json(BASE_URL+'/models')
    chosen=next((m for m in catalog['data'] if m['id']==REVIEW_MODEL),None)
    if chosen is None or 'audio' not in chosen['architecture'].get('input_modalities',[]):
        raise OpenRouterError('The configured review model is not available with audio input.')
    save_json(OUT/'review_model_snapshot.json',{'retrieved_utc':stamp(),'model':chosen})
    (OUT/'review_instructions.txt').write_text(INSTRUCTION,encoding='utf-8')
    results_path=OUT/'automated_review.json'
    existing=json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    history_path=OUT/'review_attempt_history.jsonl'
    if not history_path.exists():
        with history_path.open('w',encoding='utf-8') as f:
            for r in existing: f.write(json.dumps(r,ensure_ascii=False)+chr(10))
    done={r['sample_id'] for r in existing if not retry_failed or r['review_status']=='reviewed'}
    remaining=[r for r in manifest.to_dict('records') if r['sample_id'] not in done and r.get('wav_path')]
    if max_requests is not None: remaining=remaining[:max_requests]
    ledger_path=OUT/'budget.json'; ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    with make_client(timeout=45) as billing:
        for offset in range(0,len(remaining),workers):
            batch=remaining[offset:offset+workers]
            if budget_spend(billing,ledger)+.1*len(batch)>18:
                print('Review stopped by budget guard.',flush=True); break
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in pool.map(annotate_one,batch):
                    existing=[r for r in existing if r['sample_id']!=result['sample_id']]+[result]
                    with history_path.open('a',encoding='utf-8') as f: f.write(json.dumps(result,ensure_ascii=False)+chr(10))
            save_json(results_path,existing)
            spent=budget_spend(billing,ledger)
            ledger.update(observed_key_usage_delta_usd=spent)
            save_json(ledger_path,ledger)
            print(json.dumps({'audio_reviews_attempted':len(existing),'status':pd.Series([r['review_status'] for r in existing]).value_counts().to_dict(),'observed_usage_usd':round(ledger['observed_key_usage_delta_usd'],6)}),flush=True)
            if any(r.get('fatal_auth_or_credit_error') for r in existing): break
    records=[]
    for result in existing:
        flat={k:v for k,v in result.items() if k!='review_usage'}
        flat['review_usage_json']=json.dumps(result.get('review_usage'))
        records.append(flat)
    pd.DataFrame(records).to_csv(OUT/'automated_review.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden-key',action='store_true')
    parser.add_argument('--workers',type=int,default=3,choices=[1,2,3])
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--retry-failed',action='store_true',help='Explicitly retry failed reviews after repairing a diagnosed request issue; history is retained.')
    args=parser.parse_args()
    if args.hidden_key: os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter credential (hidden): ')
    try: annotate(args.workers,args.max_requests,args.retry_failed)
    except OpenRouterError as error: print('Error:',error,flush=True); raise SystemExit(1)
