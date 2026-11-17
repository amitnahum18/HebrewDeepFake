"""Select and extract a bounded multi-utterance real-speaker sample."""
from pathlib import Path
import json, hashlib, wave
import av
import numpy as np
import pandas as pd
from comparative_analysis import load,normalize
from comparative_audio import features

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'voice_distribution'

def extract(path,start,end):
    # Browser-produced MKA timestamps jitter by packet. Inserting zeros at
    # every PTS gap creates artificial silence. Use the decoded sample clock,
    # consistent with preceding corpus analysis, and retain timing limitations.
    first=round(start*16000); last=round(end*16000); count=0; chunks=[]
    with av.open(str(path)) as container:
        stream=container.streams.audio[0]
        converter=av.AudioResampler(format='s16',layout='mono',rate=16000)
        for frame in container.decode(stream):
            for converted in converter.resample(frame):
                values=converted.to_ndarray().ravel()
                a=max(first,count); b=min(last,count+len(values))
                if b>a: chunks.append(values[a-count:b-count])
                count+=len(values)
            if count>=last: break
        if count<last:
            for converted in converter.resample(None):
                values=converted.to_ndarray().ravel(); a=max(first,count); b=min(last,count+len(values))
                if b>a: chunks.append(values[a-count:b-count])
                count+=len(values)
    x=np.concatenate(chunks) if chunks else np.array([],dtype=np.int16)
    coverage=len(x)/max(1,last-first)
    if coverage<.98: raise ValueError('Decoded sample coverage below 98%%: %.5f'%coverage)
    if not np.any(x): raise ValueError('Zero extracted waveform')
    return x,coverage

def main():
    OUT.mkdir(exist_ok=True); (OUT/'real_audio').mkdir(exist_ok=True)
    real,seg,syn,ref=load()
    valid=(seg.decoded_duration_s.gt(0)&seg.peak.gt(0)&seg.decode_error.isna()&seg.duration_s.between(3,8)&~seg.timestamp_invalid.astype(bool)&seg.start.ge(0)&seg.end.le(seg.decoded_duration_s+.1)&seg.word_count.between(6,14)&seg.quality_score.ge(.8)&seg.segment_alignment_score.ge(.8))
    candidates=seg.loc[valid].copy(); candidates['text_group']=candidates.text.map(normalize)
    candidates['distance_to_target']=abs(candidates.duration_s-4.5)
    rows=[]; coverage=[]
    paths=real.set_index('session_id').audio_path.to_dict()
    for user in sorted(real.user_id.unique()):
        g=candidates.loc[candidates.user_id.eq(user)].sort_values(['distance_to_target','segment_alignment_score'],ascending=[True,False])
        selected=[]; sessions=set(); texts=set()
        for require_new_session in [True,False]:
            for _,r in g.iterrows():
                if len(selected)>=3: break
                if r.text_group in texts or (require_new_session and r.session_id in sessions): continue
                selected.append(r); sessions.add(r.session_id); texts.add(r.text_group)
        successes=0
        for r in selected:
            sample='real_'+r.session_id+'_'+str(int(r.segment_index))
            path=OUT/'real_audio'/(sample+'.wav')
            row=dict(sample_id=sample,label='real',group_id=user,family='real',model='real',voice=user,session_id=r.session_id,segment_index=int(r.segment_index),source_start_s=r.start,source_end_s=r.end,word_count=r.word_count,text=r.text,sex_label=r.sex_label,age_proxy_bin=r.age_proxy_bin,source_path=paths[r.session_id],source_quality=r.quality_score,alignment_score=r.segment_alignment_score)
            try:
                x,ratio=extract(paths[r.session_id],r.start,r.end)
                with wave.open(str(path),'wb') as stream:
                    stream.setparams((1,2,16000,0,'NONE','not compressed')); stream.writeframes(x.astype('<i2').tobytes())
                row.update(status='extracted_not_human_verified',decoded_sample_coverage_fraction=ratio,extraction_clock='concatenated_decoded_samples_no_PTS_gap_insertion',absolute_wav_path=str(path),wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**features(x.astype(float)/32768))
                successes+=1
            except Exception as error: row.update(status='extraction_failed',error=str(error))
            rows.append(row)
        coverage.append(dict(user_id=user,eligible_candidates=len(g),selected=len(selected),extracted=successes,distinct_selected_sessions=len(sessions)))
        print('Real speaker',len(coverage),'/',real.user_id.nunique(),'clips',successes,flush=True)
    realdf=pd.DataFrame(rows); realdf.to_csv(OUT/'real_extraction_manifest.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(coverage).to_csv(OUT/'real_speaker_coverage.csv',index=False,encoding='utf-8-sig')
    combined=realdf.loc[realdf.status.eq('extracted_not_human_verified')].to_dict('records')
    for r in syn.itertuples():
        path=ROOT/'synthetic_pilot'/r.wav_path
        combined.append(dict(sample_id=r.sample_id,label='synthetic',group_id=r.voice_identity_group,family=r.family,model=r.model,voice=r.voice,session_id='',word_count=len(r.text.split()),text=r.text,status='existing_synthetic_not_human_verified',absolute_wav_path=str(path),wav_sha256=r.wav_sha256,duration_s=r.duration_s,rms_dbfs=r.rms_dbfs,quiet_fraction_45dbfs=r.quiet_fraction_45dbfs,sex_label='not_comparable',age_proxy_bin='not_comparable'))
    pd.DataFrame(combined).to_csv(OUT/'samples.csv',index=False,encoding='utf-8-sig')
    print('Prepared',len(combined),'samples',flush=True)

if __name__=='__main__': main()
