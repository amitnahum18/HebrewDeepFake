"""Extract a small speaker-spanning real-speech reference with comparable utterance lengths."""
from pathlib import Path
import wave
import av
import numpy as np
import pandas as pd
from synthetic_pilot import ROOT, OUT, decode_audio


def prepare_reference():
    target=OUT/'human_reference.csv'
    if target.exists(): return pd.read_csv(target)
    recordings=pd.read_csv(ROOT/'eda_outputs'/'recording_features.csv')
    segments=pd.read_csv(ROOT/'eda_outputs'/'segments.csv')
    merged=segments.merge(recordings[['session_id','user_id','audio_path','quality_score','peak','sex_label','age_proxy_bin']],on='session_id',validate='many_to_one')
    candidates=merged[(merged.duration_s.between(3,8))&merged.word_count.between(6,10)&(merged.start>=0)&(merged.end<=20)&(merged.quality_score>=.8)&(merged.peak>0)].copy()
    candidates['word_count_distance']=(candidates.word_count-8).abs()
    selected=candidates.sort_values(['word_count_distance','quality_score'],ascending=[True,False]).drop_duplicates('user_id')
    folder=OUT/'human_reference_audio'; folder.mkdir(parents=True,exist_ok=True)
    results=[]
    for _,row in selected.iterrows():
        path=folder/(str(row.session_id)+'_'+str(int(row.segment_index))+'.wav')
        try:
            if not path.exists():
                chunks=[]; count=0; end=int(np.ceil(row.end*16000))
                with av.open(str(row.audio_path)) as source:
                    stream=source.streams.audio[0]
                    converter=av.AudioResampler(format='s16',layout='mono',rate=16000)
                    for frame in source.decode(stream):
                        for converted in converter.resample(frame):
                            chunk=converted.to_ndarray().ravel(); chunks.append(chunk); count+=len(chunk)
                        if count>=end: break
                clip=np.concatenate(chunks)[round(row.start*16000):round(row.end*16000)]
                if not len(clip): raise ValueError('Empty reference clip')
                with wave.open(str(path),'wb') as output:
                    output.setnchannels(1); output.setsampwidth(2); output.setframerate(16000); output.writeframes(clip.astype('<i2').tobytes())
            metrics=decode_audio({'format':'container'},path)
            results.append({'session_id':row.session_id,'segment_index':int(row.segment_index),'user_id':row.user_id,
                            'source_start_s':row.start,'source_end_s':row.end,'word_count':row.word_count,
                            'sex_label':row.sex_label,'age_proxy_bin':row.age_proxy_bin,'quality_score':row.quality_score,
                            'text':row.text,'reference_status':'decoded',**metrics})
        except Exception as error:
            results.append({'session_id':row.session_id,'user_id':row.user_id,'reference_status':'failed','error':type(error).__name__})
    frame=pd.DataFrame(results); frame.to_csv(target,index=False,encoding='utf-8-sig')
    print('Human reference excerpts:',len(frame),'distinct recorder IDs',frame.user_id.nunique(),flush=True)
    return frame


if __name__=='__main__': prepare_reference()
