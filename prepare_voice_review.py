"""Prepare auditable listening excerpts; keep original demographic metadata intact."""
from pathlib import Path
import argparse, base64, hashlib, json
import av
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'voice_review'; OUT.mkdir(exist_ok=True)
QUEUE=OUT/'review_queue.csv'
STATE=OUT/'current_excerpt.json'

def record_label(label, expected_review_number, response):
    info=json.loads(STATE.read_text(encoding='utf-8'))
    if expected_review_number is None or info['review_number']!=expected_review_number:
        raise ValueError('Specify the review number of the excerpt the user actually answered.')
    queue=pd.read_csv(QUEUE,keep_default_na=False)
    matches=queue.index[(queue.review_number==expected_review_number)&queue.session_id.eq(info['session_id'])]
    if len(matches)!=1: raise ValueError('Current excerpt does not uniquely match the review queue.')
    idx=matches[0]
    if queue.loc[idx,'review_status']!='presented': raise ValueError('Excerpt is not awaiting a response.')
    reviewed_utc=pd.Timestamp.now(tz='UTC').isoformat()
    queue.loc[idx,'perceived_voice_presentation']=label
    queue.loc[idx,'review_status']='reviewed'
    queue.loc[idx,'reviewer']='user_in_conversation'
    queue.loc[idx,'reviewed_utc']=reviewed_utc
    queue.loc[idx,'notes']=response or ''
    queue.to_csv(QUEUE,index=False,encoding='utf-8-sig')
    event={**info,'perceived_voice_presentation':label,'user_response':response,'reviewer':'user_in_conversation','reviewed_utc':reviewed_utc,'annotation_provenance':'human_auditory_impression'}
    with (OUT/'annotation_history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False)+chr(10))
    print(json.dumps({'recorded_review':expected_review_number,'label':label,'reviewed_count':int(queue.review_status.eq('reviewed').sum())}))

def prepare():
    if not QUEUE.exists():
        df=pd.read_csv(ROOT/'eda_outputs'/'recording_features.csv')
        unknown=df[df.sex_label.eq('unknown')].copy()
        unknown['audible_candidate']=unknown.peak.gt(0)&unknown.decoded_duration_s.ge(8)&unknown.decode_error.isna()
        selected=unknown.sort_values(['audible_candidate','quality_score','quiet_fraction_45dbfs'],ascending=[False,False,True]).drop_duplicates('user_id').copy()
        selected=selected[['session_id','user_id','audio_path','session_duration','decoded_duration_s','quality_score','biological_sex']].reset_index(drop=True)
        selected.insert(0,'review_number',range(1,len(selected)+1))
        selected['review_status']='pending'
        selected['perceived_voice_presentation']=''
        selected['annotation_provenance']='human_auditory_impression'
        selected['annotation_scope']='reviewed_excerpt_only'
        selected['reviewer']=''
        selected['reviewed_utc']=''
        selected['notes']=''
        selected.to_csv(QUEUE,index=False,encoding='utf-8-sig')
    queue=pd.read_csv(QUEUE,keep_default_na=False)
    pending=queue[queue.review_status.isin(['pending','presented'])]
    if pending.empty:
        print(json.dumps({'status':'complete','reviewed':len(queue)})); return
    idx=pending.index[0]; row=queue.loc[idx]
    audio=[]; target_rate=16000
    with av.open(row.audio_path) as container:
        stream=container.streams.audio[0]
        converter=av.AudioResampler(format='s16',layout='mono',rate=target_rate)
        count=0
        for frame in container.decode(stream):
            for converted in converter.resample(frame):
                chunk=converted.to_ndarray().ravel(); audio.append(chunk); count+=len(chunk)
            if count>=13*target_rate: break
    samples=np.concatenate(audio)
    start=target_rate if len(samples)>9*target_rate else 0
    excerpt=samples[start:min(len(samples),start+12*target_rate)]
    basename=f"excerpt_{int(row.review_number):02d}"
    wav=OUT/(basename+'.wav'); mp3=OUT/(basename+'.mp3')
    import wave
    with wave.open(str(wav),'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(target_rate); f.writeframes(excerpt.astype('<i2').tobytes())
    with av.open(str(mp3),'w') as output:
        encoder=output.add_stream('libmp3lame',rate=target_rate); encoder.bit_rate=32000; encoder.layout='mono'
        frame=av.AudioFrame.from_ndarray(excerpt.reshape(1,-1),format='s16',layout='mono'); frame.sample_rate=target_rate
        for packet in encoder.encode(frame): output.mux(packet)
        for packet in encoder.encode(None): output.mux(packet)
    info={'review_number':int(row.review_number),'total_speakers':len(queue),'session_id':row.session_id,'user_id':row.user_id,'source_audio':row.audio_path,'start_s':start/target_rate,'end_s':(start+len(excerpt))/target_rate,'sample_rate':target_rate,'wav_path':str(wav),'mp3_path':str(mp3),'wav_sha256':hashlib.sha256(wav.read_bytes()).hexdigest(),'annotation_scope':'reviewed_excerpt_only','original_demographic_metadata_unchanged':True}
    STATE.write_text(json.dumps(info,indent=2),encoding='utf-8')
    queue.loc[idx,'review_status']='presented'; queue.to_csv(QUEUE,index=False,encoding='utf-8-sig')
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio-base64',action='store_true')
    parser.add_argument('--label',choices=['masculine','feminine','uncertain','multiple_speakers'])
    parser.add_argument('--expected-review-number',type=int)
    parser.add_argument('--response')
    args=parser.parse_args()
    if args.label:
        record_label(args.label,args.expected_review_number,args.response)
        prepare()
    elif args.audio_base64:
        info=json.loads(STATE.read_text(encoding='utf-8'))
        print(base64.b64encode(Path(info['mp3_path']).read_bytes()).decode('ascii'))
    else: prepare()
