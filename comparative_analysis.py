"""Offline comparative EDA. No generation, annotations, or source audio mutations."""
from pathlib import Path
import hashlib, json, re, sys, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wasserstein_distance
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from comparative_audio import analyze_one

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'synthetic_pilot'
OUT=ROOT/'comparison_outputs'
FIG=OUT/'figures'
FEATURES=['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction','energy_dynamic_range_db','spectral_centroid_hz','highband_energy_fraction','spectral_flatness','zero_crossing_fraction']

def export(df,name):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    return df

def normalize(text):
    return ' '.join(re.sub(r'[^א-תa-z0-9 ]',' ',str(text).lower()).split())

def load():
    real=pd.read_csv(ROOT/'eda_outputs/recording_features.csv')
    seg=pd.read_csv(ROOT/'eda_outputs/segments.csv',keep_default_na=False)
    qual=pd.read_csv(ROOT/'eda_outputs/segment_quality.csv').rename(columns={'quality_index':'segment_index','probability':'segment_alignment_score','start':'quality_start','end':'quality_end'})
    seg=seg.merge(qual,on=['session_id','segment_index'],validate='one_to_one',how='left')
    assert np.allclose(seg.start,seg.quality_start,equal_nan=False) and np.allclose(seg.end,seg.quality_end,equal_nan=False)
    seg=seg.merge(real[['session_id','user_id','decoded_duration_s','peak','quality_score','sex_label','age_proxy_bin','decode_error']],on='session_id',validate='many_to_one')
    synth=pd.read_csv(BASE/'analysis/sample_features.csv')
    ref=pd.read_csv(BASE/'human_reference.csv')
    return real,seg,synth,ref

def eligibility(real,seg):
    steps=[('all aligned rows',pd.Series(True,index=seg.index)),
           ('source decoded and nonzero',seg.decoded_duration_s.gt(0)&seg.peak.gt(0)&seg.decode_error.isna()),
           ('positive valid in-bounds timing',seg.duration_s.gt(0)&~seg.timestamp_invalid.astype(bool)&seg.start.ge(0)&seg.end.le(seg.decoded_duration_s+.1)),
           ('nonempty transcript',seg.word_count.gt(0)&seg.normalized_text.str.strip().ne('')),
           ('session score >= 0.3',seg.quality_score.ge(.3)),
           ('segment alignment score >= 0.5',seg.segment_alignment_score.ge(.5))]
    mask=pd.Series(True,index=seg.index); rows=[]; previous=len(seg)
    for name,condition in steps:
        mask &= condition
        part=seg.loc[mask]
        rows.append(dict(step=name,segments=len(part),removed_at_step=previous-len(part),speakers=part.user_id.nunique(),sessions=part.session_id.nunique(),aligned_hours=part.duration_s.sum()/3600))
        previous=len(part)
    export(pd.DataFrame(rows),'real_eligibility_funnel')
    sensitivity=[]
    base=mask.copy()
    for session_q,segment_q in [(.3,.5),(.8,.8)]:
        selected=seg.loc[base & seg.quality_score.ge(session_q)&seg.segment_alignment_score.ge(segment_q)]
        for restricted in [False,True]:
            p=selected.loc[selected.duration_s.between(2,10)&selected.word_count.between(6,14)] if restricted else selected
            sensitivity.append(dict(session_threshold=session_q,alignment_threshold=segment_q,prototype_lengths=restricted,segments=len(p),speakers=p.user_id.nunique(),aligned_hours=p.duration_s.sum()/3600))
    export(pd.DataFrame(sensitivity),'eligibility_sensitivity')
    strict=seg.loc[base & seg.quality_score.ge(.8)&seg.segment_alignment_score.ge(.8)&seg.duration_s.between(2,10)&seg.word_count.between(6,14)].copy()
    strict['hebrew_letter_fraction']=strict.text.map(lambda s:sum('\u05d0'<=c<='\u05ea' for c in s)/max(1,sum(c.isalpha() for c in s)))
    strict=strict.loc[strict.hebrew_letter_fraction.ge(.8)].copy()
    strict['exact_text_group']=strict.text.map(normalize)
    strict=strict.sort_values(['segment_alignment_score','quality_score'],ascending=False)
    chosen=[]; seen=set(); per_speaker={u:0 for u in real.user_id.unique()}
    # Round-robin prevents prolific speakers consuming globally shared texts first.
    groups={u:g for u,g in strict.groupby('user_id')}
    for round_index in range(4):
        for user in sorted(groups):
            options=groups[user].loc[~groups[user].exact_text_group.isin(seen)]
            if len(options):
                row=options.iloc[0].to_dict(); row['selection_round']=round_index+1
                chosen.append(row); seen.add(row['exact_text_group']); per_speaker[user]+=1
    proposal=pd.DataFrame(chosen)
    families=['elevenlabs','google','minimax','speechify']
    # Counterbalanced across selection rounds and speakers; proposal only.
    proposal['desired_family']=[families[i%4] for i in range(len(proposal))]
    proposal['proposed_pair_id']=['proposed_%04d'%i for i in range(1,len(proposal)+1)]
    proposal['assigned_model']=''; proposal['assigned_voice']=''; proposal['split']=''
    proposal['status']='proposal_requires_listening_and_boundary_QC_not_generated'
    export(proposal,'proposed_same_text_pairs')
    coverage=real.groupby('user_id').agg(source_sessions=('session_id','size'),source_hours=('decoded_duration_s',lambda s:s.sum()/3600)).reset_index()
    coverage['eligible_prototype_segments']=coverage.user_id.map(strict.groupby('user_id').size()).fillna(0).astype(int)
    coverage['proposed_pairs']=coverage.user_id.map(per_speaker)
    export(coverage,'proposed_speaker_coverage')
    export(proposal.groupby('desired_family').agg(proposed_pairs=('proposed_pair_id','size'),real_seconds=('duration_s','sum')).reset_index(),'proposed_family_allocation')
    return mask,proposal

def acoustic(synth,ref):
    rows=[]
    for _,r in ref.iterrows():
        rows.append(dict(sample_id='real_'+r.session_id+'_'+str(r.segment_index),label='real',group_id=str(r.user_id),family='real',word_count=int(r.word_count),text=r.text,absolute_wav_path=str(BASE/r.wav_path)))
    for _,r in synth.iterrows():
        rows.append(dict(sample_id=r.sample_id,label='synthetic',group_id=r.voice_identity_group,family=r.family,word_count=len(r.text.split()),text=r.text,absolute_wav_path=str(BASE/r.wav_path)))
    result=pd.DataFrame([analyze_one(row) for row in rows])
    # Store paths and hashes; transforms are in memory only.
    result['wav_sha256']=result.absolute_wav_path.map(lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    export(result,'common_audio_features')
    return result

def effects(frame,view,prefix=''):
    group=frame.groupby(['label','group_id'])[ [prefix+f for f in FEATURES] ].median()
    rng=np.random.default_rng(20261116); rows=[]
    for feature in FEATURES:
        key=prefix+feature
        a=group.loc['real',key].dropna().to_numpy(); b=group.loc['synthetic',key].dropna().to_numpy()
        u=mannwhitneyu(b,a,alternative='two-sided').statistic
        auc=u/(len(a)*len(b)); delta=2*auc-1
        sd=np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
        differences=b[rng.integers(0,len(b),(1000,len(b)))].mean(axis=1)-a[rng.integers(0,len(a),(1000,len(a)))].mean(axis=1)
        lo,hi=np.quantile(differences,[.025,.975])
        rows.append(dict(view=view,feature=feature,real_groups=len(a),synthetic_groups=len(b),real_median=np.median(a),synthetic_median=np.median(b),median_difference=np.median(b)-np.median(a),mean_difference=b.mean()-a.mean(),smd=(b.mean()-a.mean())/sd if sd else np.nan,cliffs_delta=delta,wasserstein_distance=wasserstein_distance(a,b),bootstrap_mean_difference_low=lo,bootstrap_mean_difference_high=hi,in_sample_orientation_free_auc=max(auc,1-auc)))
    return pd.DataFrame(rows)

def match_diagnostic(frame):
    a=frame.loc[frame.label.eq('real')&frame.word_count.eq(8)].reset_index(drop=True)
    b=frame.loc[frame.label.eq('synthetic')].reset_index(drop=True)
    fam=sorted(b.family.unique()); cap=min(20,len(a)//len(fam))
    edges=[(i,j,abs(r.duration_s-s.duration_s)) for i,r in a.iterrows() for j,s in b.iterrows() if abs(r.duration_s-s.duration_s)<=.5]
    groups=sorted(b.group_id.unique()); nr=len(a); ng=len(groups); nf=len(fam); n=len(edges)
    A=lil_matrix((nr+ng+nf,n),dtype=float)
    gi={g:i for i,g in enumerate(groups)}; fi={f:i for i,f in enumerate(fam)}
    for k,(i,j,d) in enumerate(edges):
        A[i,k]=1; A[nr+gi[b.iloc[j].group_id],k]=1; A[nr+ng+fi[b.iloc[j].family],k]=1
    # Try descending common family count: cardinality first, cost second.
    solution=None; common=0
    for target in range(cap,-1,-1):
        low=np.r_[np.zeros(nr+ng),np.full(nf,target)]
        high=np.r_[np.ones(nr+ng),np.full(nf,target)]
        solution=milp(np.array([e[2] for e in edges]),integrality=np.ones(n),bounds=Bounds(0,1),constraints=LinearConstraint(A.tocsr(),low,high),options={'time_limit':60})
        if solution.success: common=target; break
    assert solution is not None and solution.success, 'Matching optimizer did not certify optimum'
    pairs=[]
    for k in np.flatnonzero(solution.x>.5):
        i,j,d=edges[k]; r=a.iloc[i]; s=b.iloc[j]
        pairs.append(dict(real_sample_id=r.sample_id,real_group_id=r.group_id,synthetic_sample_id=s.sample_id,synthetic_group_id=s.group_id,family=s.family,real_duration_s=r.duration_s,synthetic_duration_s=s.duration_s,duration_difference_s=d,real_words=r.word_count,synthetic_words=s.word_count,text_matched=False))
    pairs=pd.DataFrame(pairs)
    assert pairs.real_group_id.is_unique and pairs.synthetic_group_id.is_unique
    assert pairs.duration_difference_s.le(.5+1e-9).all()
    assert pairs.groupby('family').size().eq(common).all()
    export(pairs,'diagnostic_duration_word_pairs')
    subset=frame.loc[frame.sample_id.isin(set(pairs.real_sample_id)|set(pairs.synthetic_sample_id))]
    return pairs,subset

def plots(frame,effect,pairs):
    FIG.mkdir(parents=True,exist_ok=True)
    for columns,name in [(FEATURES[:6],'acoustic_distributions'),(FEATURES[6:],'spectral_distributions')]:
        fig,axes=plt.subplots(2,3,figsize=(14,8)); axes=axes.ravel()
        grouped=frame.groupby(['label','group_id'])[columns].median().reset_index()
        for ax,key in zip(axes,columns):
            for label,color in [('real','tab:blue'),('synthetic','tab:orange')]:
                v=np.sort(grouped.loc[grouped.label.eq(label),key].dropna())
                ax.step(v,np.arange(1,len(v)+1)/len(v),label=label,color=color)
            ax.set_title(key); ax.set_ylabel('Group-median empirical CDF'); ax.legend()
        for ax in axes[len(columns):]: ax.set_visible(False)
        fig.tight_layout(); fig.savefig(FIG/(name+'.png'),dpi=150); plt.close(fig)
    table=effect.pivot(index='feature',columns='view',values='smd')
    fig,ax=plt.subplots(figsize=(12,6)); table.plot.barh(ax=ax); ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('Standardized mean difference: synthetic minus real; observed group medians')
    fig.tight_layout(); fig.savefig(FIG/'effect_sensitivities.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(pairs.real_duration_s,pairs.synthetic_duration_s)
    ax.plot([2,8],[2,8],color='black',linestyle='--'); ax.set(xlabel='Real duration (s)',ylabel='Synthetic duration (s)',title='Diagnostic pairs: 8 words, <=0.5 s caliper; DIFFERENT TEXT')
    fig.tight_layout(); fig.savefig(FIG/'diagnostic_pairs.png',dpi=150); plt.close(fig)

def audit(real,seg,synth,ref):
    text=normalize(synth.text.iloc[0]); overlap=int(seg.text.map(normalize).eq(text).sum())
    codec=pd.concat([real.assign(label='real')[['label','codec']].rename(columns={'codec':'native_codec'}),synth.assign(label='synthetic')[['label','source_codec']].rename(columns={'source_codec':'native_codec'})])
    export(codec.groupby(['label','native_codec']).size().rename('samples').reset_index(),'native_codec_confound')
    family=synth.groupby('family').agg(configurations=('sample_id','size'),nominal_voices=('voice_identity_group','nunique'),seconds=('duration_s','sum')).reset_index()
    family['configuration_share']=family.configurations/len(synth); family['audio_time_share']=family.seconds/family.seconds.sum()
    export(family,'synthetic_family_exposure')
    export(synth.groupby(['family','voice_identity_group']).agg(configurations=('sample_id','size'),seconds=('duration_s','sum')).reset_index(),'synthetic_voice_exposure')
    review=synth.loc[synth.review_flag_count.gt(0),['sample_id','model','voice','duration_s','rms_dbfs','quiet_fraction_45dbfs','review_reasons']]
    export(review,'synthetic_review_queue')
    export(ref.groupby(['sex_label','age_proxy_bin'],dropna=False).size().rename('reference_excerpts').reset_index(),'reference_metadata_coverage')
    def summarize_labels(values):
        known=sorted(set(values.dropna().astype(str))-{'','unknown','nan'})
        return known[0] if len(known)==1 else ('conflicting' if len(known)>1 else 'unknown')
    real_labels=real.groupby('user_id').agg(source_sex=('sex_label',summarize_labels),age_proxy_bins=('age_proxy_bin',summarize_labels)).reset_index()
    export(real_labels,'real_speaker_source_labels')
    synth_labels=synth.groupby(['family','voice_identity_group']).agg(provider_gender=('provider_gender_label',summarize_labels),provider_age=('provider_age_label',summarize_labels),estimated_presentation=('presentation_for_summary',summarize_labels),raw_estimated_age=('raw_age_for_summary',summarize_labels)).reset_index()
    export(synth_labels,'synthetic_nominal_voice_labels')
    label_rows=[]
    for unit,table,columns in [('real_speakers',real_labels,['source_sex','age_proxy_bins']),('synthetic_nominal_voices',synth_labels,['provider_gender','provider_age','estimated_presentation','raw_estimated_age'])]:
        for column in columns:
            for value,count in table[column].value_counts(dropna=False).items():
                label_rows.append(dict(unit=unit,field=column,value=value,count=int(count)))
    export(pd.DataFrame(label_rows),'label_distributions')
    return {'exact_real_segment_text_matches':overlap,'real_recordings':len(real),'real_speakers':real.user_id.nunique(),'real_hours':real.session_duration.sum()/3600,'real_reference_speakers':ref.user_id.nunique(),'synthetic_configurations':len(synth),'synthetic_nominal_voices':synth.voice_identity_group.nunique(),'synthetic_minutes':synth.duration_s.sum()/60,'synthetic_unique_texts':synth.text.map(normalize).nunique(),'synthetic_human_verified':int(synth.human_verified.fillna(False).astype(bool).sum()),'synthetic_age_raw_categories':int(synth.raw_age_for_summary.nunique()),'synthetic_review_candidates':len(review),'native_codec_overlap':sorted(set(real.codec.dropna())&set(synth.source_codec.dropna()))}

def run():
    real,seg,synth,ref=load()
    mask,proposal=eligibility(real,seg)
    metrics=audit(real,seg,synth,ref)
    frame=acoustic(synth,ref)
    pairs,matched=match_diagnostic(frame)
    effects_table=pd.concat([effects(frame,'original'),effects(frame,'edge_trimmed','edge_trimmed_'),effects(frame,'level_normalized','level_normalized_'),effects(matched,'duration_word_matched')],ignore_index=True)
    export(effects_table,'group_effect_sizes')
    export(frame.groupby(['label','family'])[FEATURES].agg(['count','median','min','max']).reset_index(),'acoustic_summary')
    plots(frame,effects_table,pairs)
    metrics.update(diagnostic_pairs=len(pairs),diagnostic_pairs_per_family=pairs.groupby('family').size().to_dict(),proposal_pairs=len(proposal),proposal_real_speakers=proposal.user_id.nunique(),proposal_real_seconds=float(proposal.duration_s.sum()),eligible_segments=int(mask.sum()))
    (OUT/'key_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    inputs=[ROOT/'eda_outputs/recording_features.csv',ROOT/'eda_outputs/segments.csv',ROOT/'eda_outputs/segment_quality.csv',BASE/'analysis/sample_features.csv',BASE/'human_reference.csv',ROOT/'comparative_analysis.py',ROOT/'comparative_audio.py',ROOT/'COMPARATIVE_EDA_PLAN.md']
    provenance={'seed':20261116,'python':sys.version,'platform':platform.platform(),'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'no_api_calls':True,'bootstrap_repetitions':1000,'note':'Reference is a selected 78-speaker short-clip subset, not representative corpus estimates. Spectral masking and fixed absolute energy thresholds are not VAD/SNR. No originals modified.'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return metrics

if __name__=='__main__': print(json.dumps(run(),indent=2))
