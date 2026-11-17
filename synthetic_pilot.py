"""Generate one Hebrew utterance for every selected stock model/voice combination.

No voice cloning, no demographic inference from names, and no automatic retries.
Use --hidden-key to enter a credential without displaying or saving it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import hashlib
import html
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import wave
import uuid
from typing import Any

import av
import numpy as np
import pandas as pd

from openrouter_client import BASE_URL, OpenRouterError, make_client, request_speech

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'synthetic_pilot'
PROMPT='היום ניפגש בשעה שלוש ליד תחנת הרכבת בירושלים.'
MODELS=[
 ('google/gemini-3.8-flash-lite-tts','google','pcm',24000),
 ('google/gemini-3.8-flash-tts','google','pcm',24000),
 ('google/gemini-3.1-flash-tts-preview','google','pcm',24000),
 ('elevenlabs/eleven-v3','elevenlabs','mp3',None),
 ('elevenlabs/eleven-v3-conversational','elevenlabs','mp3',None),
 ('elevenlabs/eleven-v4','elevenlabs','mp3',None),
 ('elevenlabs/eleven-v4-turbo','elevenlabs','mp3',None),
 ('minimax/speech-2.8-hd','minimax','mp3',None),
 ('minimax/speech-2.8-turbo','minimax','mp3',None),
 ('speechify/simba-3.5','speechify','mp3',None),
]


def stamp(): return datetime.now(timezone.utc).isoformat()


def public_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'HebrewDeepFake-research-pilot'}),timeout=45) as response:
        return json.load(response)


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def prepare():
    OUT.mkdir(exist_ok=True)
    path=OUT/'planned_samples.csv'
    if path.exists():
        print('Using frozen plan:',len(pd.read_csv(path)),'model/voice samples',flush=True)
        return
    catalog=public_json(BASE_URL+'/models?output_modalities=speech')
    save_json(OUT/'catalog_snapshot.json',{'retrieved_utc':stamp(),'url':BASE_URL+'/models?output_modalities=speech',**catalog})
    lookup={m['id']:m for m in catalog['data']}
    # Optional official provider metadata: missing metadata stays unknown.
    voice_metadata={}; enrichment={'source':'https://api.elevenlabs.io/v1/voices','status':'not_retrieved'}
    try:
        response=public_json(enrichment['source'])
        for voice in response.get('voices',[]):
            name=voice.get('name','').split(' - ')[0].strip().casefold()
            voice_metadata.setdefault(name,[]).append(voice)
        enrichment={'source':enrichment['source'],'retrieved_utc':stamp(),'status':'retrieved','voices':response.get('voices',[])}
    except Exception:
        enrichment['status']='unavailable_without_provider_authentication'
    save_json(OUT/'provider_voice_metadata.json',enrichment)
    rows=[]
    for model,family,fmt,rate in MODELS:
        if model not in lookup: raise RuntimeError('Selected model missing from the current public catalog: '+model)
        item=lookup[model]
        for voice in item.get('supported_voices') or []:
            if not isinstance(voice,str): raise RuntimeError('Voice catalog schema changed')
            sample_id=hashlib.sha256((model+'|'+voice+'|'+PROMPT).encode()).hexdigest()[:20]
            candidates=voice_metadata.get(voice.casefold(),[]) if family=='elevenlabs' else []
            matched=candidates[0] if len(candidates)==1 else {}
            labels=matched.get('labels',{})
            rows.append({'sample_id':sample_id,'model':model,'canonical_model':item.get('canonical_slug',model),
                         'family':family,'voice':voice,'voice_identity_group':family+':'+voice,'text_id':'he_shared_01',
                         'text':PROMPT,'requested_language':'he','format':fmt,'pcm_sample_rate':rate,
                         'provider_gender_label':labels.get('gender','unknown'),
                         'provider_age_label':labels.get('age','unknown'),
                         'provider_label_source':enrichment['source'] if matched else 'unavailable',
                         'provider_voice_id':matched.get('voice_id',''),
                         'input_price_usd':item.get('pricing',{}).get('prompt'),
                         'output_price_usd':item.get('pricing',{}).get('completion')})
    df=pd.DataFrame(rows)
    assert df.sample_id.is_unique
    df.to_csv(path,index=False,encoding='utf-8-sig')
    save_json(OUT/'run_config.json',{'created_utc':stamp(),'status':'planned','prompt':PROMPT,'target_language':'he',
              'requested_samples':len(df),'voice_identity_groups':df.voice_identity_group.nunique(),
              'scope':'one short utterance per stock model/voice; no volume expansion',
              'project_budget_usd':20,'normal_spend_stop_usd':18,'uncommitted_request_allowance_usd':.1,
              'demographic_policy':'provider labels and perceived audio-model estimates remain separate; no actual age or biological sex inferred',
              'automatic_retries':0,'voice_cloning':False})
    print('Frozen plan:',len(df),'samples;',df.voice_identity_group.nunique(),'provider/voice groups',flush=True)
    print(df.groupby('model').size().to_string(),flush=True)


def key_usage(client):
    try:
        result=client.get('/key',cast_to=dict[str,Any])
        return float(result['data']['usage'])
    except Exception:
        raise OpenRouterError('Could not verify current key usage. Generation stopped without another request.') from None


def budget_spend(client,ledger):
    """Use a high-water mark because cached key-usage snapshots can go backwards."""
    usage=key_usage(client)
    high=max(usage,float(ledger.get('key_usage_high_water_usd',ledger['baseline_key_usage_usd'])))
    ledger.update(latest_key_usage_usd=usage,key_usage_high_water_usd=high,last_checked_utc=stamp())
    return high-ledger['baseline_key_usage_usd']


def decode_audio(row,path):
    if row['format']=='pcm':
        source_rate=int(float(row['pcm_sample_rate']))
        raw=np.frombuffer(path.read_bytes(),dtype='<i2')
        x=raw.astype(np.float64)/32768
        if source_rate!=16000:
            from scipy.signal import resample_poly
            import math
            factor=math.gcd(source_rate,16000)
            x=resample_poly(x,16000//factor,source_rate//factor)
        codec='pcm_s16le'; channels=1
    else:
        chunks=[]
        with av.open(str(path)) as source:
            stream=source.streams.audio[0]
            source_rate=stream.codec_context.sample_rate; codec=stream.codec_context.name; channels=stream.codec_context.channels
            converter=av.AudioResampler(format='fltp',layout='mono',rate=16000)
            for frame in source.decode(stream):
                for converted in converter.resample(frame): chunks.append(converted.to_ndarray().ravel())
            for converted in converter.resample(None): chunks.append(converted.to_ndarray().ravel())
        if not chunks: raise ValueError('Empty decoded audio')
        x=np.concatenate(chunks).astype(np.float64)
    if not len(x): raise ValueError('Empty decoded audio')
    nonfinite=int((~np.isfinite(x)).sum())
    if nonfinite: raise ValueError('Nonfinite decoded samples')
    wav=path.with_suffix('.wav')
    if not wav.exists():
        pcm=np.rint(np.clip(x,-1,1)*32767).astype('<i2')
        with wave.open(str(wav),'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(pcm.tobytes())
    width=320; frames=x[:len(x)//width*width].reshape(-1,width)
    frame_db=20*np.log10(np.maximum(np.sqrt(np.mean(frames**2,axis=1)),1e-12))
    duration=len(x)/16000; rms=float(np.sqrt(np.mean(x*x))); peak=float(np.max(np.abs(x)))
    return {'duration_s':duration,'source_sample_rate':source_rate,'source_codec':codec,'source_channels':channels,
            'analysis_sample_rate':16000,'rms_dbfs':float(20*np.log10(max(rms,1e-12))),'peak':peak,
            'near_fullscale_fraction':float((np.abs(x)>=.999).mean()),'dc_offset':float(x.mean()),
            'quiet_fraction_45dbfs':float((frame_db< -45).mean()),'low_energy_p10_dbfs':float(np.percentile(frame_db,10)),
            'energy_dynamic_range_db':float(np.percentile(frame_db,90)-np.percentile(frame_db,10)),
            'nonfinite_samples':nonfinite,'zero_audio':peak==0,'wav_path':str(wav.relative_to(OUT)),
            'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),
            'technical_qc_pass':bool(1<=duration<=25 and peak>0 and rms>1e-5)}


def options_for(row):
    family=row['family']
    if family=='google':
        return {'instructions':'Read the exact input text in Hebrew, without translating or adding words.',
                'provider':{'options':{'google-ai-studio':{'maxOutputTokens':512}}}}
    if family=='speechify': return {'provider':{'options':{'speechify':{'language':'he'}}}}
    if family=='minimax': return {'provider':{'options':{'minimax':{'language_boost':'Hebrew'}}}}
    if family=='elevenlabs': return {'provider':{'options':{'elevenlabs':{'language_code':'he'}}}}
    return {}


def generate_one(row):
    folder=OUT/'audio'; folder.mkdir(exist_ok=True)
    path=folder/(row['sample_id']+'.'+row['format'])
    result={**row,'attempted_utc':stamp(),'generation_status':'failed','language_qc_status':'not_reviewed'}
    try:
        if path.exists():
            metadata=json.loads(path.with_suffix(path.suffix+'.json').read_text(encoding='utf-8'))
        else:
            metadata=request_speech(row['text'],model=row['model'],voice=row['voice'],output=path,
                      response_format=row['format'],timeout=90,extra_body=options_for(row))
        result.update(generation_status='generated',generation_id=metadata.get('generation_id'),
                      original_path=str(path.relative_to(OUT)),original_sha256=metadata['sha256'],bytes=metadata['bytes'])
        result.update(decode_audio(row,path)); result['generation_status']='decoded'
    except Exception as error:
        result['error']=str(error) if isinstance(error,OpenRouterError) else type(error).__name__
        if isinstance(error,OpenRouterError) and ('HTTP 401' in str(error) or 'HTTP 402' in str(error) or 'HTTP 403' in str(error)):
            result['fatal_auth_or_credit_error']=True
    return result


def generate(workers=3,max_requests=None):
    prepare()
    df=pd.read_csv(OUT/'planned_samples.csv',keep_default_na=False)
    manifest_path=OUT/'manifest.csv'
    existing=pd.read_csv(manifest_path,keep_default_na=False).to_dict('records') if manifest_path.exists() else []
    # Failed attempts are retained and are not silently retried on resume.
    done={r['sample_id'] for r in existing}
    remaining=[r for r in df.to_dict('records') if r['sample_id'] not in done]
    first=[]; rest=[]; seen=set()
    for row in remaining:
        if row['model'] not in seen:
            first.append(row); seen.add(row['model'])
        else: rest.append(row)
    remaining=first+rest
    if max_requests is not None: remaining=remaining[:max_requests]
    ledger_path=OUT/'budget.json'
    with make_client(timeout=45) as billing:
        usage=key_usage(billing)
        ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else {'baseline_key_usage_usd':usage,'created_utc':stamp(),'maximum_project_usd':20,'normal_stop_usd':18}
        save_json(ledger_path,ledger)
        for offset in range(0,len(remaining),workers):
            spent=budget_spend(billing,ledger)
            if spent< -1e-7: raise OpenRouterError('Usage baseline changed. Check the credential before resuming.')
            batch=remaining[offset:offset+workers]
            if spent+.1*len(batch)>18:
                print('Stopped at budget guard; no more generation requests.',flush=True); break
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures=[pool.submit(generate_one,row) for row in batch]
                for future in as_completed(futures): existing.append(future.result())
            pd.DataFrame(existing).to_csv(manifest_path,index=False,encoding='utf-8-sig')
            spent=budget_spend(billing,ledger)
            ledger.update(observed_key_usage_delta_usd=spent)
            save_json(ledger_path,ledger)
            counts=pd.DataFrame(existing).generation_status.value_counts().to_dict()
            print(json.dumps({'attempted':len(existing),'planned':len(df),'status':counts,'observed_usage_usd':round(ledger['observed_key_usage_delta_usd'],6)}),flush=True)
            if any(r.get('fatal_auth_or_credit_error') for r in existing):
                print('Stopped after an authentication, authorization or credit failure.',flush=True); break
    config=json.loads((OUT/'run_config.json').read_text(encoding='utf-8'))
    config.update(generation_finished_utc=stamp(),status='generation_attempts_complete' if len(existing)==len(df) else 'generation_incomplete')
    save_json(OUT/'run_config.json',config)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','generate'])
    parser.add_argument('--hidden-key',action='store_true')
    parser.add_argument('--workers',type=int,default=3,choices=[1,2,3])
    parser.add_argument('--max-requests',type=int,help='Limit this invocation; resume never repeats an attempted sample.')
    args=parser.parse_args()
    if args.hidden_key:
        os.environ['OPENROUTER_API_KEY']=getpass.getpass('OpenRouter credential (hidden): ')
    try:
        if args.command=='prepare': prepare()
        else: generate(args.workers,args.max_requests)
    except OpenRouterError as error:
        print('Error:',error,flush=True); raise SystemExit(1)
