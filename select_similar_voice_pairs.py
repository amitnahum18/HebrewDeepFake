"""Select existing real/synthetic listening pairs in three feature spaces.

No generation/API calls. Same candidate controls across methods; identity- and
diverse six-pair selections. Consensus is a separate extra view.
"""
from pathlib import Path
import json,hashlib,shutil,wave
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
from cluster_voice_features import CORE,ALL,transform
from comparative_audio import read_wav,features

ROOT=Path(__file__).resolve().parent;BASE=ROOT/'voice_distribution';OUT=ROOT/'similar_voice_pairs'
METHODS=['acoustic','ecapa','hybrid']
PAIRS_PER_METHOD=6

def choose_six(df,score):
    # Within a real-ID/synthetic-ID edge, only the cheapest clip configuration
    # can improve this objective; remove dominated alternatives before solving.
    g=df.sort_values([score,'pair_id']).drop_duplicates(['real_group','synthetic_group']).reset_index(drop=True)
    real=sorted(g.real_group.unique());syn=sorted(g.synthetic_group.unique());families=sorted(g.family.unique())
    ri={k:i for i,k in enumerate(real)};si={k:i for i,k in enumerate(syn)};fi={k:i for i,k in enumerate(families)}
    nr,ns,nf=len(real),len(syn),len(families)
    a=lil_matrix((nr+ns+nf+1,len(g)),dtype=float)
    for j,r in enumerate(g.itertuples()):
        a[ri[r.real_group],j]=1;a[nr+si[r.synthetic_group],j]=1;a[nr+ns+fi[r.family],j]=1;a[-1,j]=1
    lo=np.r_[np.zeros(nr+ns+nf),PAIRS_PER_METHOD]
    hi=np.r_[np.ones(nr+ns),np.full(nf,3),PAIRS_PER_METHOD]
    result=milp(g[score].to_numpy(),integrality=np.ones(len(g)),bounds=Bounds(0,1),constraints=LinearConstraint(a.tocsr(),lo,hi),options={'time_limit':90})
    if not result.success:raise ValueError('Six-pair optimum not certified: '+result.message)
    chosen=g.loc[result.x>.5].sort_values([score,'pair_id']).copy()
    assert len(chosen)==6 and chosen.real_group.nunique()==6 and chosen.synthetic_group.nunique()==6
    assert chosen.family.value_counts().max()<=3
    return chosen

def choose_two(df,score):
    g=df.sort_values([score,'pair_id']).reset_index(drop=True)
    best=None;cost=np.inf
    for i,r in g.iterrows():
        if r[score]*2>cost:break
        for j in range(i+1,len(g)):
            s=g.iloc[j]
            if r[score]+s[score]>=cost:break
            if r.real_group==s.real_group or r.synthetic_group==s.synthetic_group or r.family==s.family:continue
            best=(i,j);cost=r[score]+s[score];break
    if best is None:raise ValueError('No two identity/family-distinct candidates')
    return g.iloc[list(best)].copy()

def wav_write(path,x):
    with wave.open(str(path),'wb') as stream:
        stream.setparams((1,2,16000,0,'NONE','not compressed'))
        stream.writeframes(np.round(np.clip(x,-1,32767/32768)*32768).astype('<i2').tobytes())

def main():
    OUT.mkdir(exist_ok=True);(OUT/'audio').mkdir(exist_ok=True);(OUT/'metadata').mkdir(exist_ok=True)
    samples=pd.read_csv(BASE/'samples.csv',keep_default_na=False)
    acoustics=pd.read_csv(BASE/'sample_acoustic_features.csv')
    acoustic=acoustics.set_index('sample_id').loc[samples.sample_id].reset_index()
    group=acoustics.groupby(['label','group_id'],sort=True)[ALL].median().reset_index()
    group_z,fit,_,_=transform(group,CORE)
    z,_,_,_=transform(acoustic,CORE,fit)
    block_scale=float(np.sqrt(np.mean(np.sum(group_z*group_z,axis=1))))
    cache=np.load(BASE/'embeddings.npz',allow_pickle=False)
    lookup={str(k):e for k,v,e in zip(cache['sample_ids'],cache['views'],cache['embeddings']) if v=='original'}
    e=np.array([lookup[k] for k in samples.sample_id])
    hybrid=np.column_stack([z/block_scale,e])/np.sqrt(2)
    real=np.flatnonzero(samples.label.eq('real'));syn=np.flatnonzero(samples.label.eq('synthetic'))
    distances={'acoustic':cdist(z[real],z[syn]),'ecapa':1-np.clip(e[real]@e[syn].T,-1,1),'hybrid':cdist(hybrid[real],hybrid[syn])}
    rows=[]
    for i,a in enumerate(real):
        r=samples.iloc[a]
        for j,b in enumerate(syn):
            s=samples.iloc[b]
            rows.append(dict(pair_id=r.sample_id+'__'+s.sample_id,real_sample=r.sample_id,synthetic_sample=s.sample_id,real_group=r.group_id,synthetic_group=s.group_id,family=s.family,model=s.model,voice=s.voice,real_words=int(r.word_count),synthetic_words=int(s.word_count),real_duration_s=r.duration_s,synthetic_duration_s=s.duration_s,duration_gap_s=abs(r.duration_s-s.duration_s),text_matched=bool(r.text==s.text),**{m+'_distance':float(d[i,j]) for m,d in distances.items()}))
    all_pairs=pd.DataFrame(rows)
    controlled=all_pairs.loc[all_pairs.real_words.eq(8)&all_pairs.synthetic_words.eq(8)&all_pairs.duration_gap_s.le(.5)].copy()
    for m in METHODS:controlled[m+'_percentile']=controlled[m+'_distance'].rank(method='average',pct=True)
    controlled['consensus_max_percentile']=controlled[[m+'_percentile' for m in METHODS]].max(axis=1)
    selected=[]
    for method in METHODS+['consensus']:
        score=method+'_percentile' if method!='consensus' else 'consensus_max_percentile'
        chosen=choose_six(controlled,score)
        for rank,r in enumerate(chosen.to_dict('records'),1):
            percentile=r[method+'_percentile'] if method!='consensus' else r['consensus_max_percentile']
            selected.append({'selection_method':method,'within_method_rank':rank,'relative_match_score':100*(1-percentile),**r})
    selected=pd.DataFrame(selected)
    all_pairs.to_csv(OUT/'all_cross_class_distances.csv',index=False,encoding='utf-8-sig')
    controlled.to_csv(OUT/'controlled_candidate_pairs.csv',index=False,encoding='utf-8-sig')
    selected.to_csv(OUT/'selected_pairs.csv',index=False,encoding='utf-8-sig')
    # Save unconstrained six-pair choices for transparency, without claiming they
    # satisfy the primary duration/word controls or adding them to the gallery.
    unrestricted=[]
    for method in METHODS:
        g=choose_six(all_pairs,method+'_distance');g['selection_method']=method;unrestricted.append(g)
    pd.concat(unrestricted,ignore_index=True).to_csv(OUT/'uncontrolled_selection_reference.csv',index=False,encoding='utf-8-sig')
    sample_map=samples.set_index('sample_id');assets={}
    for sid in sorted(set(selected.real_sample)|set(selected.synthetic_sample)):
        row=sample_map.loc[sid];source=Path(row.absolute_wav_path)
        assert hashlib.sha256(source.read_bytes()).hexdigest()==row.wav_sha256
        raw=OUT/'audio'/(sid+'.wav');level=OUT/'audio'/(sid+'__listening_level.wav')
        shutil.copy2(source,raw)
        x=read_wav(source);rms=np.sqrt(np.mean(x*x));gain=min(10**(-23/20)/max(rms,1e-12),.99/max(np.abs(x).max(),1e-12))
        wav_write(level,x*gain)
        assets[sid]={'sample_id':sid,'label':row.label,'group_id':row.group_id,'model':row.model,'voice':row.voice,'family':row.family,'text':row.text,'duration_s':row.duration_s,'source_sha256':row.wav_sha256,'raw_audio':'audio/'+raw.name,'level_audio':'audio/'+level.name,'listening_gain_db':float(20*np.log10(gain)),'listening_rms_dbfs':features(read_wav(level))['rms_dbfs'],'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'level_sha256':hashlib.sha256(level.read_bytes()).hexdigest(),'normalization_note':'Listening-only gain toward -23 dBFS RMS, peak capped at 0.99. No speed or boundary change; selection uses original features.'}
    (OUT/'audio_assets.json').write_text(json.dumps(assets,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in selected.to_dict('records'):
        data={**r,'real':assets[r['real_sample']],'synthetic':assets[r['synthetic_sample']],'limits':['Different Hebrew texts: not content-matched or cloned speaker identity.','Relative match score = 100*(1-method percentile), not human quality or identity probability.','Six selected pairs are an illustration, not a statistically valid method-quality comparison.','Audio has not received new human verification.']}
        (OUT/'metadata'/(r['selection_method']+'_'+str(r['within_method_rank'])+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    metrics={'all_cross_class_pairs':len(all_pairs),'controlled_candidate_pairs':len(controlled),'eligible_real_clips':controlled.real_sample.nunique(),'eligible_real_speakers':controlled.real_group.nunique(),'eligible_synthetic_clips':controlled.synthetic_sample.nunique(),'eligible_synthetic_groups':controlled.synthetic_group.nunique(),'pairs_per_group':6,'method_selections':18,'additional_consensus_selections':6,'unique_selected_pairs':selected.pair_id.nunique(),'distinct_selected_original_recordings':len(assets),'selection_controls':'exactly 8 aligned/whitespace words; duration gap <=0.5 s; six distinct real IDs and nominal synthetic groups per method; max 3 pairs per family; certified maximum sum of relative match scores (minimum sum of percentile ranks)','score_definition':'100*(1-percentile rank within the common controlled pool); higher is closer, not a human quality or identity probability','not_text_matched':not selected.text_matched.any(),'api_calls':0,'new_audio_generated':False,'word_count_caution':'Real aligned word count and synthetic whitespace words use existing source definitions; boundary/transcript validation remains pending.'}
    (OUT/'key_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    provenance={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE/'samples.csv',BASE/'sample_acoustic_features.csv',BASE/'embeddings.npz',ROOT/'cluster_voice_features.py',Path(__file__)]},'acoustic_scaler':'identity-fitted original core acoustic transform from clustering; no class labels in fit','ecapa':'original per-clip L2 vectors; cosine distance','hybrid':'Euclidean on equal-expected-energy acoustic+ECAPA blocks, using identity-fitted block scale','human_similarity_not_yet_rated':True}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
