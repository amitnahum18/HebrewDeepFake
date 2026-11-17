"""Corpus-level quality/duplication audit and representative listening manifests.

No API calls, new audio or destructive filtering. Near-text similarities are
screening candidates, not confirmed audio duplicates or semantic equivalence.
"""
from pathlib import Path
import json, hashlib, wave
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import comparative_analysis as comparison
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'quality_audit'
OUT.mkdir(exist_ok=True)

def save(df,name):
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    return df

def exact_text_audit(frame,text_column,id_columns,prefix,min_words=1):
    work=frame.copy()
    work['audit_normalized_text']=work[text_column].map(comparison.normalize)
    work['audit_words']=work.audit_normalized_text.str.split().str.len()
    work=work.loc[work.audit_words.ge(min_words)]
    stats=work.groupby('audit_normalized_text').agg(rows=('audit_words','size'),word_count=('audit_words','first'),speakers=('user_id','nunique')).reset_index()
    repeated=stats.loc[stats.rows.gt(1)].sort_values(['speakers','word_count','rows'],ascending=False)
    repeated['text_group_id']=repeated.audit_normalized_text.map(lambda s:hashlib.sha256(s.encode()).hexdigest()[:16])
    save(repeated,prefix+'_groups')
    occurrences=work.merge(repeated[['audit_normalized_text','text_group_id']],on='audit_normalized_text')
    save(occurrences[id_columns+['text_group_id','audit_words','audit_normalized_text']],prefix+'_occurrences')
    return repeated,occurrences

def near_sessions(real):
    work=real.loc[real.whitespace_words.ge(20)].copy().reset_index(drop=True)
    text=work.text.map(comparison.normalize)
    matrix=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True).fit_transform(text)
    rows=[]
    for start in range(0,len(work),100):
        similarities=(matrix[start:start+100]@matrix.T).tocoo()
        for i,j,score in zip(similarities.row,similarities.col,similarities.data):
            i+=start
            if j<=i or score<.92 or text.iloc[i]==text.iloc[j]: continue
            a=work.iloc[i]; b=work.iloc[j]
            rows.append(dict(session_a=a.session_id,session_b=b.session_id,user_a=a.user_id,user_b=b.user_id,cosine_similarity=score,words_a=a.whitespace_words,words_b=b.whitespace_words,same_document=bool(a.document_title==b.document_title),text_a=a.text,text_b=b.text,status='near_text_candidate_not_audio_duplicate'))
    return save(pd.DataFrame(rows,columns=['session_a','session_b','user_a','user_b','cosine_similarity','words_a','words_b','same_document','text_a','text_b','status']).sort_values('cosine_similarity',ascending=False),'near_session_text_candidates')

def main():
    real,segments,synth,ref=comparison.load()
    eligible=(segments.decoded_duration_s.gt(0)&segments.peak.gt(0)&segments.decode_error.isna()&segments.duration_s.gt(0)&~segments.timestamp_invalid.astype(bool)&segments.start.ge(0)&segments.end.le(segments.decoded_duration_s+.1)&segments.word_count.gt(0)&segments.normalized_text.str.strip().ne('')&segments.quality_score.ge(.3)&segments.segment_alignment_score.ge(.5))
    pool=segments.loc[eligible].copy()
    groups,occ=exact_text_audit(segments,'text',['session_id','segment_index','user_id','start','end','duration_s'],'real_segment_exact')
    valid_groups,valid_occ=exact_text_audit(pool,'text',['session_id','segment_index','user_id','start','end','duration_s'],'eligible_segment_exact',6)
    session_groups,session_occ=exact_text_audit(real,'text',['session_id','user_id','document_title','audio_path'],'real_session_exact')
    near=near_sessions(real)
    # Keep failures/flags separate; flags are deliberately overlapping.
    flags=[]
    for row in real.itertuples():
        reasons=[]
        if not row.decoded_duration_s>0 or not row.peak>0: reasons.append('zero_audio')
        if row.quality_score<.3: reasons.append('source_alignment_score_below_0.3')
        if row.near_fullscale_fraction>.001: reasons.append('near_fullscale_over_0.1pct_review_not_confirmed_clipping')
        if row.quiet_fraction_45dbfs>.5: reasons.append('quiet_majority_absolute_energy_review')
        if abs(row.duration_discrepancy_s)>.5: reasons.append('duration_difference_over_0.5s_review')
        if row.hebrew_letter_fraction<.8: reasons.append('hebrew_letter_fraction_below_0.8_review')
        for reason in reasons:
            flags.append(dict(session_id=row.session_id,user_id=row.user_id,reason=reason,quality_score=row.quality_score,duration_s=row.decoded_duration_s,source_path=row.audio_path,text_excerpt=str(row.text)[:240]))
    flagtable=save(pd.DataFrame(flags),'real_priority_review')
    save(flagtable.groupby('reason').agg(sessions=('session_id','nunique'),speakers=('user_id','nunique')).reset_index(),'real_priority_review_summary')
    save(synth.loc[synth.review_flag_count.gt(0),['sample_id','family','model','voice','voice_identity_group','duration_s','rms_dbfs','review_reasons','wav_path']],'synthetic_priority_review')
    # Reconcile exact audio fingerprints from full preceding inventories.
    fingerprints=[]
    for label,df,column,idcol in [('real_canonical',real,'pcm16_16k_mono_sha256','session_id'),('synthetic_canonical',synth,'wav_sha256','sample_id'),('synthetic_original',synth,'original_sha256','sample_id')]:
        counts=df.groupby(column,dropna=False).size()
        fingerprints.append(dict(scope=label,files=len(df),missing_fingerprints=int(df[column].isna().sum()),duplicate_hash_groups=int(counts.gt(1).sum()),duplicate_rows=int(counts.loc[counts.gt(1)].sum()),hash_semantics='decoded PCM digest' if label=='real_canonical' else 'file digest'))
    save(pd.DataFrame(fingerprints),'exact_audio_fingerprint_audit')
    saved_audio=[]
    for label,df,idcol in [('real_reference',ref,'session_id'),('synthetic',synth,'sample_id')]:
        for row in df.to_dict('records'):
            path=ROOT/'synthetic_pilot'/row['wav_path']
            with wave.open(str(path),'rb') as stream:
                assert (stream.getnchannels(),stream.getsampwidth(),stream.getframerate())==(1,2,16000)
                digest=hashlib.sha256(stream.readframes(stream.getnframes())).hexdigest()
            saved_audio.append(dict(label=label,sample_id=row[idcol],wav_path=row['wav_path'],pcm16_sha256=digest))
    saved_audio=save(pd.DataFrame(saved_audio),'saved_audio_pcm_fingerprints')
    save(saved_audio.loc[saved_audio.pcm16_sha256.duplicated(False)],'saved_audio_exact_pcm_duplicates')
    # Whole corpus vs eligible vs proposal vs short reference: same source labels,
    # different weights explicitly named. Never mix synthetic casting with sex.
    proposal=pd.read_csv(ROOT/'comparison_outputs/proposed_same_text_pairs.csv')
    populations=[('full_real_recordings_decoded_seconds',real,'decoded_duration_s'),('eligible_real_segments_aligned_seconds',pool,'duration_s'),('proposed_real_pairs_segment_seconds',proposal,'duration_s'),('selected_real_reference_seconds',ref,'duration_s')]
    demographics=[]
    for scope,df,duration in populations:
        for field in ['sex_label','age_proxy_bin']:
            table=df.groupby(field,dropna=False).agg(rows=('user_id','size'),speakers=('user_id','nunique'),seconds=(duration,'sum')).reset_index()
            for r in table.itertuples(index=False,name=None):
                value,n,speakers,seconds=r
                demographics.append(dict(scope=scope,field=field,value=value,rows=n,speakers=speakers,minutes=seconds/60,time_share=seconds/df[duration].sum(),speaker_share=speakers/df.user_id.nunique()))
    demo=save(pd.DataFrame(demographics),'real_demographic_coverage_by_scope')
    distance=[]
    for field in ['sex_label','age_proxy_bin']:
        for weight in ['time_share','speaker_share']:
            t=demo.loc[demo.field.eq(field)].pivot(index='value',columns='scope',values=weight).fillna(0)
            base=t['full_real_recordings_decoded_seconds']
            for col in t:
                distance.append(dict(field=field,weight=weight,comparison=col,total_variation_vs_full=.5*abs(t[col]-base).sum(),interpretation='descriptive selection shift; full corpus is not population truth'))
    save(pd.DataFrame(distance),'reference_selection_shift')
    # Speaker exposure audit retains all source IDs, even if no eligible segment.
    exposure=real.groupby('user_id').agg(source_seconds=('decoded_duration_s','sum'),source_sessions=('session_id','size'),sex_label=('sex_label','first'),age_proxy_bin=('age_proxy_bin','first')).reset_index()
    exposure['eligible_seconds']=exposure.user_id.map(pool.groupby('user_id').duration_s.sum()).fillna(0)
    exposure['eligible_segments']=exposure.user_id.map(pool.groupby('user_id').size()).fillna(0).astype(int)
    exposure['source_time_share']=exposure.source_seconds/exposure.source_seconds.sum()
    exposure['equal_speaker_target_share']=1/len(exposure)
    save(exposure.sort_values('source_seconds',ascending=False),'real_speaker_exposure')
    # Identity-coverage listening sample, NOT claimed population-representative.
    listen=[]
    for user in sorted(real.user_id.unique()):
        choices=pool.loc[pool.user_id.eq(user)&pool.duration_s.between(2,10)&pool.word_count.between(6,14)].copy()
        if len(choices):
            # Deterministic median-duration example, favour higher alignment on ties.
            choices['distance_to_median']=abs(choices.duration_s-choices.duration_s.median())
            row=choices.sort_values(['distance_to_median','segment_alignment_score'],ascending=[True,False]).iloc[0]
            source=real.loc[real.session_id.eq(row.session_id)].iloc[0]
            listen.append(dict(user_id=user,status='listen_source_interval_before_extraction',session_id=row.session_id,segment_index=row.segment_index,start_s=row.start,end_s=row.end,text=row.text,sex_label=row.sex_label,age_proxy_bin=row.age_proxy_bin,source_path=source.audio_path))
        else: listen.append(dict(user_id=user,status='no_eligible_prototype_interval_manual_review_required'))
    save(pd.DataFrame(listen),'real_identity_coverage_listening')
    voice_sample=[]
    for family,family_df in synth.groupby('family'):
        for i,(voice,g) in enumerate(family_df.groupby('voice_identity_group')):
            choices=g.sort_values('model')
            voice_sample.append(choices.iloc[i%len(choices)].to_dict())
    voice_sample=pd.DataFrame(voice_sample)
    save(voice_sample[['sample_id','family','model','voice','voice_identity_group','wav_path','duration_s','provider_gender_label','provider_age_label','presentation_for_summary','review_reasons']],'synthetic_identity_coverage_listening')
    assert voice_sample.voice_identity_group.nunique()==len(voice_sample)==116
    metrics=dict(real_segment_repeated_text_groups=len(groups),real_segment_rows_in_repeated_groups=len(occ),real_segment_excess_rows_over_one_per_text=len(occ)-len(groups),eligible_sixplus_word_repeated_groups=len(valid_groups),eligible_sixplus_word_repeated_rows=len(valid_occ),eligible_sixplus_word_cross_speaker_groups=int(valid_groups.speakers.gt(1).sum()),real_session_repeated_text_groups=len(session_groups),real_session_rows_in_repeated_groups=len(session_occ),near_session_text_candidate_pairs=len(near),synthetic_same_text_rows=len(synth),synthetic_unique_texts=synth.text.map(comparison.normalize).nunique(),synthetic_voice_groups=synth.voice_identity_group.nunique(),real_identity_listening_rows=len(listen),synthetic_identity_listening_rows=len(voice_sample),saved_362_audio_pcm_duplicate_rows=int(saved_audio.pcm16_sha256.duplicated(False).sum()),api_calls=0)
    (OUT/'key_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    inputs=[ROOT/'eda_outputs/recording_features.csv',ROOT/'eda_outputs/segments.csv',ROOT/'eda_outputs/segment_quality.csv',ROOT/'synthetic_pilot/analysis/sample_features.csv',ROOT/'synthetic_pilot/human_reference.csv',ROOT/'comparison_outputs/proposed_same_text_pairs.csv',Path(__file__)]
    (OUT/'provenance.json').write_text(json.dumps({'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'normalization':'lowercase Hebrew/Latin/digits, punctuation removed, whitespace collapsed; diacritics removed','near_text_method':'all session pairs with >=20 source whitespace words; word unigram/bigram TF-IDF sublinear_tf; cosine>=0.92; exclude normalized exact duplicates','limitations':['No new human annotations','No perceptual near-audio duplicate search','No phonetic near-text matching','Existing real canonical fingerprints hash decoded PCM; synthetic fingerprints hash WAV bytes, not cross-class comparable','Source demographic descriptors are not synthetic perceived age/sex','Listening manifests cover identities, not estimated population prevalence'],'no_audio_mutations':True},indent=2),encoding='utf-8')
    figures=OUT/'figures'; figures.mkdir(exist_ok=True)
    for field,name in [('sex_label','source_sex_representation'),('age_proxy_bin','source_age_representation')]:
        part=demo.loc[demo.field.eq(field)]
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        for ax,weight in zip(axes,['speaker_share','time_share']):
            chart=part.pivot(index='value',columns='scope',values=weight).fillna(0)*100
            chart.columns=[c.replace('full_real_recordings_decoded_seconds','full corpus').replace('eligible_real_segments_aligned_seconds','eligible pool').replace('proposed_real_pairs_segment_seconds','308 proposals').replace('selected_real_reference_seconds','78 references') for c in chart]
            chart.plot.bar(ax=ax); ax.set_title(weight); ax.set_ylabel('Percent'); ax.tick_params(axis='x',rotation=30)
        fig.suptitle('SOURCE labels only; descriptive selection shift, not population truth')
        fig.tight_layout(); fig.savefig(figures/(name+'.png'),dpi=150); plt.close(fig)
    return metrics

if __name__=='__main__': print(json.dumps(main(),indent=2))
