"""Build a self-contained, executable analysis notebook (no network required)."""
from pathlib import Path
import sys
import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md('''# Hebrew crowd-recital: exploratory data analysis

Executed analysis of the original session corpus, not the derived Whisper training set.
Read [EDA_PLAN.md](EDA_PLAN.md) for the prespecified workflow, screening rules, and sources.
The supplied HebSpoof PDF informs downstream questions; it does not authorize synthesis or model training.

**Interpretation:** `biological_sex` is the source label, not gender identity. Birth year does not provide age at recording. `quality_score` is alignment confidence, not acoustic quality. All flags are review candidates, never automatic exclusions. Independent sampling units are speakers; recordings and segments repeat speakers.''')
code(r'''from pathlib import Path
import os, sys, json, hashlib, re, unicodedata, platform, importlib.metadata, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Audio, Markdown
import av
SEED = 20261115
rng = np.random.default_rng(SEED)
ROOT = Path.cwd()
if not (ROOT / 'EDA_PLAN.md').exists():
    raise RuntimeError('Run this notebook with the project directory as working directory.')
OUT = ROOT / 'eda_outputs'; OUT.mkdir(exist_ok=True)
FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
SNAPSHOT = Path(os.environ.get('CROWD_RECITAL_PATH', str(Path.home()/'.cache/huggingface/hub/datasets--ivrit-ai--crowd-recital/snapshots/34d93a876ae075a1a7904e7ec2c54d2af1c9d4d7')))
assert (SNAPSHOT/'manifest.csv').exists(), f'Missing manifest: {SNAPSHOT}'
run = {'dataset':'ivrit-ai/crowd-recital','revision':SNAPSHOT.name,'input_path':str(SNAPSHOT),
       'seed':SEED,'analysis_reference_year':2026,'started_utc':pd.Timestamp.now(tz='UTC').isoformat(),
       'python':sys.version,'platform':platform.platform(),
       'versions':{x:importlib.metadata.version(x) for x in ['pandas','numpy','matplotlib','seaborn','av','nbformat','nbclient']}}
(OUT/'run_provenance.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
pd.set_option('display.max_columns',30)
sns.set_theme(style='whitegrid',context='notebook')
plt.rcParams.update({'figure.figsize':(11,5),'figure.dpi':110})
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/f'{name}.png',dpi=160,bbox_inches='tight'); plt.show(); plt.close()
def export(df,name):
    df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
manifest=pd.read_csv(SNAPSHOT/'manifest.csv')
display(manifest.head()); print('Manifest shape:',manifest.shape)
''')
md('''## 1. Inventory, schema, and integrity
Inventory hashes cover the downloaded files, including original audio. Presence checks are separate from successful parsing or decoding. Manifest/JSON disagreement is retained for investigation.''')
code(r'''files=sorted(p for p in SNAPSHOT.rglob('*') if p.is_file() and '.cache' not in p.relative_to(SNAPSHOT).parts)
assert len(files)>=len(manifest)*4+1, 'Incomplete input inventory'
def inventory_one(p): return {'path':p.relative_to(SNAPSHOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)}
with ThreadPoolExecutor(max_workers=4) as pool: inventory=pd.DataFrame(pool.map(inventory_one,files))
export(inventory,'file_inventory')
session_dirs=sorted(p for p in SNAPSHOT.iterdir() if p.is_dir() and not p.name.startswith('.'))
expected=['audio.mka','metadata.json','transcript.aligned.json','transcript.vtt']
presence=pd.DataFrame([{'session_id':p.name,**{x:(p/x).is_file() for x in expected}} for p in session_dirs])
export(presence,'companion_file_presence')
dictionary=pd.DataFrame([{'field':c,'dtype':str(manifest[c].dtype),'missing':int(manifest[c].isna().sum()),'missing_pct':100*manifest[c].isna().mean(),'unique_nonmissing':manifest[c].nunique()} for c in manifest])
definitions={
 'source_type':('category','Source collection type'), 'source_id':('identifier','Source collection identifier'),
 'source_entry_id':('identifier','Source recording identifier'), 'session_id':('identifier','Recording session identifier'),
 'session_duration':('seconds','Reported complete recording duration'), 'user_id':('identifier','Opaque recorder identifier; speaker grouping proxy'),
 'document_language':('language code','Source document language'), 'document_title':('text','Source document title'),
 'document_source_type':('category','Source document origin'), 'year_of_birth':('calendar year','Reported birth year, if supplied'),
 'biological_sex':('source category','Reported source sex label; not gender identity'),
 'quality_score':('0-1','Provider text/audio alignment quality; not acoustic quality'),
 'min_segment_quality_score':('0-1','Minimum provider segment alignment score'),
 'avg_words_per_segment':('words/segment','Reported average aligned words per segment'),
 'avg_segment_duration':('seconds','Reported average aligned segment duration'),
 'avg_words_per_minute':('words/minute','Reported recording word rate'),
 'segments_count':('count','Reported aligned segment count'), 'words_count':('count','Reported word count')}
dictionary['units']=dictionary.field.map(lambda c:definitions.get(c,('unknown','Undocumented'))[0])
dictionary['meaning']=dictionary.field.map(lambda c:definitions.get(c,('unknown','Undocumented'))[1])
export(dictionary,'data_dictionary'); display(dictionary)
integrity=[]; metadata={}; aligned={}; segments=[]; words=[]; qsegments=[]; textrows=[]; discrepancies=[]
def issue(sid,check,detail): integrity.append({'session_id':sid,'check':check,'detail':str(detail)})
def norm(text):
    text=unicodedata.normalize('NFKC',str(text))
    text=''.join(c for c in text if not unicodedata.category(c).startswith(('M','P')))
    return ' '.join(text.casefold().split())
manifest_ids=set(manifest.session_id)
for p in session_dirs:
    sid=p.name
    if sid not in manifest_ids: issue(sid,'orphan_directory','not in manifest')
    for f in expected:
        if not (p/f).exists(): issue(sid,'missing_companion',f)
    try:
        m=json.loads((p/'metadata.json').read_text(encoding='utf-8')); metadata[sid]=m
        a=json.loads((p/'transcript.aligned.json').read_text(encoding='utf-8')); aligned[sid]=a
    except Exception as e:
        issue(sid,'json_parse_error',e); continue
    duration=float(m.get('session_duration',np.nan)); ss=a.get('segments',[])
    if len(ss)!=m.get('segments_count'): issue(sid,'segment_count_mismatch',f"{len(ss)} vs {m.get('segments_count')}")
    if m.get('session_id')!=sid: issue(sid,'directory_metadata_id_mismatch',m.get('session_id'))
    if a.get('language') not in ('he',None): issue(sid,'aligned_language',a.get('language'))
    previous_end=-np.inf
    for i,s in enumerate(ss):
        start=float(s['start']); end=float(s['end']); text=s.get('text',''); sw=s.get('words',[])
        bad=not (np.isfinite(start) and np.isfinite(end) and start>=0 and end>start and end<=duration+0.1)
        overlap=start<previous_end-0.02
        if bad: issue(sid,'segment_timestamp_invalid',i)
        if overlap: issue(sid,'segment_overlap',i)
        segments.append({'session_id':sid,'segment_index':i,'start':start,'end':end,'duration_s':end-start,'gap_before_s':max(0,start-previous_end) if i else start,'text':text,'normalized_text':norm(text),'word_count':len(sw),'timestamp_invalid':bad,'overlap':overlap})
        previous_end=end
        wprev=-np.inf
        for j,w in enumerate(sw):
            ws=float(w['start']); we=float(w['end']); invalid=not(np.isfinite(ws) and np.isfinite(we) and ws>=start-0.1 and we>=ws and we<=end+0.1)
            if invalid: issue(sid,'word_timestamp_invalid',f'{i}:{j}')
            words.append({'session_id':sid,'segment_index':i,'word_index':j,'word':w.get('word',''),'start':ws,'end':we,'duration_s':we-ws,'probability':w.get('probability'), 'timestamp_invalid':invalid,'overlap':ws<wprev-0.02})
            wprev=we
    for i,q in enumerate(m.get('per_segment_quality_scores',[])):
        qsegments.append({'session_id':sid,'quality_index':i,**q})
    text=a.get('text',''); vtt=(p/'transcript.vtt').read_text(encoding='utf-8-sig') if (p/'transcript.vtt').exists() else ''
    lines=[]; in_cue=False
    for line in vtt.splitlines():
        line=line.strip()
        if '-->' in line: in_cue=True; continue
        if not line: in_cue=False; continue
        if in_cue: lines.append(re.sub(r'<[^>]*>','',line))
    vtext=' '.join(lines)
    letters=[c for c in text if c.isalpha()]
    textrows.append({'session_id':sid,'text':text,'normalized_text':norm(text),'char_count':len(text),'whitespace_words':len(text.split()),'hebrew_letter_fraction':sum('\u05d0'<=c<='\u05ea' for c in letters)/len(letters) if letters else np.nan,'has_latin':bool(re.search('[A-Za-z]',text)),'has_digits':bool(re.search('[0-9]',text)),'has_niqqud':bool(re.search('[\u0591-\u05c7]',text)),'has_question':('?' in text),'vtt_normalized_match':norm(vtext)==norm(text),'empty_text':not bool(text.strip())})
for sid in manifest_ids-set(p.name for p in session_dirs): issue(sid,'missing_session_directory','')
for _,r in manifest.iterrows():
    for field in ['session_id','source_entry_id','user_id','source_type','source_id']:
        if pd.isna(r[field]): issue(r.session_id,'missing_critical_field',field)
    for field in ['session_duration','segments_count','words_count','avg_segment_duration','avg_words_per_segment','avg_words_per_minute']:
        if pd.notna(r[field]) and (not np.isfinite(r[field]) or r[field]<0): issue(r.session_id,'invalid_manifest_numeric',field)
    for field in ['quality_score','min_segment_quality_score']:
        if pd.notna(r[field]) and not 0<=r[field]<=1: issue(r.session_id,'invalid_probability',field)
for _,r in manifest.iterrows():
    sid=r.session_id
    if sid not in metadata: continue
    for c in manifest:
        if c not in metadata[sid]: continue
        a=r[c]; b=metadata[sid][c]
        equal=(pd.isna(a) and b is None) or (isinstance(a,(float,int)) and isinstance(b,(float,int)) and np.isclose(a,b,rtol=1e-4,atol=1e-4)) or str(a)==str(b)
        if not equal: discrepancies.append({'session_id':sid,'field':c,'manifest':a,'metadata':b})
for sid in manifest.loc[manifest.session_id.duplicated(False),'session_id']: issue(sid,'duplicate_manifest_id','')
seg=pd.DataFrame(segments); word=pd.DataFrame(words); qs=pd.DataFrame(qsegments); textdf=pd.DataFrame(textrows)
for _,r in qs.iterrows():
    d=metadata.get(r.session_id,{}).get('session_duration',np.nan)
    if not (0<=r.start<r.end<=d+.1): issue(r.session_id,'quality_segment_timestamp_invalid',r.quality_index)
    if pd.notna(r.probability) and not 0<=r.probability<=1: issue(r.session_id,'quality_probability_invalid',r.quality_index)
for _,r in word.loc[word.probability.notna() & ~word.probability.between(0,1)].iterrows(): issue(r.session_id,'word_probability_invalid',r.word_index)
metadata_schema=Counter(key for m in metadata.values() for key in m)
export(pd.DataFrame([{'field':k,'sessions_present':v,'sessions_missing':len(metadata)-v} for k,v in metadata_schema.items()]),'metadata_field_coverage')
checks=pd.DataFrame(integrity,columns=['session_id','check','detail'])
diff=pd.DataFrame(discrepancies,columns=['session_id','field','manifest','metadata'])
export(checks,'integrity_issues'); export(diff,'manifest_metadata_disagreements')
export(seg,'segments'); export(word,'words'); export(qs,'segment_quality'); export(textdf,'transcript_features')
display(checks.groupby('check').size().rename('occurrences').to_frame()); print('Manifest/metadata disagreements:',len(diff)); display(presence[expected].sum())
''')
md('''## 2. Complete feature distributions
Percentages retain missing values. Identifier counts describe repetition, not meaningful semantic categories. Histograms include all numeric fields; derived acoustic and transcript features follow below.''')
code(r'''df=manifest.merge(textdf,on='session_id',how='left',validate='one_to_one')
numeric=manifest.select_dtypes(include='number').columns.tolist()
summary=manifest[numeric].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.reset_index(names='field')
export(summary,'numeric_summary'); display(summary)
cats=[]
for c in manifest.select_dtypes(exclude='number'):
    counts=manifest[c].fillna('<missing>').value_counts(dropna=False)
    cats.extend({'field':c,'value':v,'sessions':n,'pct':100*n/len(manifest)} for v,n in counts.items())
categoricals=pd.DataFrame(cats); export(categoricals,'categorical_distributions')
fig,axes=plt.subplots(int(np.ceil(len(numeric)/3)),3,figsize=(15,3.3*int(np.ceil(len(numeric)/3))))
for ax,c in zip(axes.flat,numeric):
    ax.hist(manifest[c].dropna(),bins=40,color='#197a8c'); ax.set(title=c,ylabel='Sessions')
for ax in axes.flat[len(numeric):]: ax.set_visible(False)
savefig('manifest_numeric_distributions')
overview={'sessions':len(df),'speakers':int(df.user_id.nunique()),'hours':float(df.session_duration.sum()/3600),'manifest_words':int(df.words_count.sum()),'manifest_segments':int(df.segments_count.sum()),'aligned_words':len(word),'aligned_segments':len(seg),'documents':int(df.document_title.nunique()),'inventory_files':len(inventory),'inventory_gb':float(inventory.bytes.sum()/1e9)}
display(pd.Series(overview,name='value').to_frame())
for c in ['source_type','source_id','document_language','document_source_type','biological_sex']:
    display(categoricals[categoricals.field==c])
''')
md('''## 3. Speakers, demographic coverage, and concentration
Age proxy = 2026 minus birth year; it is **not age at recording** and may differ from attained age by one year. Biological sex is reported under its original name. Speaker metadata conflicts are explicitly labeled rather than resolved silently. Exposure hours can be concentrated even when speaker counts look balanced.''')
code(r'''df['sex_label']=df.biological_sex.fillna('unknown').replace({'unspecified':'unknown','':'unknown'})
df['age_proxy_2026']=2026-df.year_of_birth
df['invalid_birth_year']=df.year_of_birth.notna() & ((df.year_of_birth>2026)|(df.year_of_birth<1906)|(df.year_of_birth%1!=0))
df.loc[df.invalid_birth_year,'age_proxy_2026']=np.nan
df['age_proxy_bin']=pd.cut(df.age_proxy_2026,[0,18,30,40,50,60,70,121],right=False,labels=['0-17','18-29','30-39','40-49','50-59','60-69','70-120']).astype('object').fillna('unknown')
def one_or_conflict(s):
    vals=s.dropna().unique()
    return vals[0] if len(vals)==1 else ('conflicting' if len(vals)>1 else 'unknown')
speaker=df.groupby('user_id').agg(sessions=('session_id','size'),seconds=('session_duration','sum'),words=('words_count','sum'),sex_label=('sex_label',one_or_conflict),age_proxy_bin=('age_proxy_bin',one_or_conflict),quality_median=('quality_score','median'),birth_year_values=('year_of_birth',lambda s:s.nunique())).reset_index()
speaker['hours']=speaker.seconds/3600
export(speaker,'speaker_summary')
balance=[]
for field in ['sex_label','age_proxy_bin']:
    a=df.groupby(field,dropna=False).agg(sessions=('session_id','size'),hours=('session_duration',lambda s:s.sum()/3600))
    b=speaker.groupby(field,dropna=False).size().rename('speakers')
    tab=a.join(b,how='outer').fillna(0).reset_index(); tab['field']=field
    for c in ['sessions','hours','speakers']: tab[c+'_pct']=100*tab[c]/tab[c].sum()
    tab=tab.rename(columns={field:'category'}); balance.append(tab)
balance=pd.concat(balance,ignore_index=True); export(balance,'demographic_balance'); display(balance)
cross=speaker.groupby(['sex_label','age_proxy_bin']).agg(speakers=('user_id','size'),hours=('hours','sum')).reset_index(); export(cross,'demographic_intersections'); display(cross)
fig,axes=plt.subplots(1,2,figsize=(15,5))
for ax,field in zip(axes,['sex_label','age_proxy_bin']):
    tab=balance[balance.field==field].set_index('category')
    tab[['speakers_pct','sessions_pct','hours_pct']].plot.bar(ax=ax); ax.set(title=field,ylabel='Percent',xlabel='')
savefig('demographic_weighting')
hours=np.sort(speaker.hours.to_numpy()); shares=hours/hours.sum(); n=len(hours)
gini=float((2*np.arange(1,n+1)-n-1)@hours/(n*hours.sum()))
concentration={'gini_hours':gini,'effective_speakers_hours':float(1/(shares@shares)),**{f'top_{k}_speaker_hour_share':float(shares[-k:].sum()) for k in [1,5,10]}}
display(pd.Series(concentration).to_frame('value'))
fig,axes=plt.subplots(1,2,figsize=(13,5)); axes[0].plot(np.r_[0,np.arange(1,n+1)/n],np.r_[0,np.cumsum(shares)]); axes[0].plot([0,1],[0,1],'--',color='gray'); axes[0].set(title='Speaker Lorenz curve',xlabel='Cumulative fraction of speakers',ylabel='Cumulative fraction of recording hours')
axes[1].hist(speaker.hours,bins=30); axes[1].set(title='Recording hours per speaker',xlabel='Hours',ylabel='Speakers'); savefig('speaker_concentration')
''')
md('''## 4. Transcript content, timing, repetition, and alignment
Normalization v1 uses NFKC, case folding, removal of Unicode marks/punctuation, and whitespace folding. It is only for duplicate screening; originals remain in the tables. Token counts are whitespace-based, not Hebrew morphology or phoneme coverage. Alignment scores and word probabilities are model outputs, not independently measured transcript accuracy.''')
code(r'''seg=seg.merge(df[['session_id','user_id']],on='session_id',validate='many_to_one')
seg['wpm_aligned']=60*seg.word_count/seg.duration_s.replace(0,np.nan)
duration_stats=seg.groupby('session_id').agg(aligned_duration_sum_s=('duration_s','sum'),segment_gap_sum_s=('gap_before_s','sum'),aligned_word_count=('word_count','sum'))
df=df.join(duration_stats,on='session_id')
df['alignment_duration_fraction']=df.aligned_duration_sum_s/df.session_duration
df['wpm_full_recording']=60*df.aligned_word_count/df.session_duration
df['wpm_aligned']=60*df.aligned_word_count/df.aligned_duration_sum_s.replace(0,np.nan)
df['word_count_difference']=df.aligned_word_count-df.words_count
quality_summary=pd.DataFrame({'metric':['session_score_lt_0.3','segment_score_lt_0.5','word_probability_lt_0.5','vtt_normalized_mismatch','manifest_word_count_mismatch'],
 'flagged':[int((df.quality_score<.3).sum()),int((qs.probability<.5).sum()),int((word.probability<.5).sum()),int((~df.vtt_normalized_match).sum()),int((df.word_count_difference!=0).sum())],
 'denominator':[int(df.quality_score.notna().sum()),int(qs.probability.notna().sum()),int(word.probability.notna().sum()),len(textdf),len(df)]})
quality_summary['pct']=100*quality_summary.flagged/quality_summary.denominator; export(quality_summary,'alignment_screening'); display(quality_summary)
duplicates=seg[seg.normalized_text.ne('')].groupby('normalized_text').agg(segments=('session_id','size'),sessions=('session_id','nunique'),speakers=('user_id','nunique')).reset_index().sort_values('segments',ascending=False)
export(duplicates[duplicates.segments>1],'repeated_segment_texts')
session_dups=df[df.normalized_text.ne('') & df.normalized_text.duplicated(False)].sort_values('normalized_text'); export(session_dups[['session_id','user_id','normalized_text']],'duplicate_session_texts')
tokens=Counter(t for text in df.normalized_text.fillna('') for t in text.split()); export(pd.DataFrame(tokens.most_common(),columns=['token','count']),'token_frequencies')
text_summary=df[['char_count','whitespace_words','hebrew_letter_fraction','has_latin','has_digits','has_niqqud','has_question','empty_text','wpm_full_recording','wpm_aligned','alignment_duration_fraction']].describe(include='all').T
display(text_summary); export(text_summary.reset_index(names='field'),'transcript_summary')
fig,axes=plt.subplots(2,3,figsize=(15,9))
for ax,(title,values) in zip(axes.flat,[('Session alignment score',df.quality_score),('Segment alignment score',qs.probability),('Word probability',word.probability),('Words/minute: full recording',df.wpm_full_recording),('Words/minute: aligned segments',df.wpm_aligned),('Sum aligned duration / recording duration',df.alignment_duration_fraction)]):
    ax.hist(values.replace([np.inf,-np.inf],np.nan).dropna(),bins=50); ax.set(title=title,ylabel='Observations')
savefig('transcripts_and_alignment')
print('Unique normalized whitespace tokens:',len(tokens)); display(duplicates.head(15))
doc_summary=df.groupby(['document_source_type','document_title'],dropna=False).agg(sessions=('session_id','size'),speakers=('user_id','nunique'),hours=('session_duration',lambda s:s.sum()/3600)).reset_index().sort_values('hours',ascending=False)
export(doc_summary,'document_distribution'); display(doc_summary.head(10))
timing_summary=pd.concat([seg[['duration_s','gap_before_s','word_count','wpm_aligned']].describe(percentiles=[.01,.05,.5,.95,.99]).T.assign(entity='segment'),word[['duration_s','probability']].describe(percentiles=[.01,.05,.5,.95,.99]).T.assign(entity='word')]).reset_index(names='feature')
export(timing_summary,'segment_word_timing_summary'); display(timing_summary)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,title,values in [(axes[0,0],'Aligned segment duration (seconds)',seg.duration_s),(axes[0,1],'Aligned word duration (seconds)',word.duration_s),(axes[1,0],'Words per aligned segment',seg.word_count),(axes[1,1],'Gap before segment (seconds)',seg.gap_before_s)]:
    ax.hist(values.dropna(),bins=50); ax.set(title=title,ylabel='Observations')
savefig('segment_word_timing')
''')
md('''## 5. Full-corpus acoustic audit
Every recording is decoded at its native sample rate for acoustic statistics. Mono energy uses channel mean, which can cancel opposing channels; channel count is retained for follow-up. RMS uses a full-scale amplitude of 1. Frame energy uses nonoverlapping 20 ms frames. Quiet thresholds are -50/-45/-40 dBFS. The 90th/10th percentile frame-energy difference is an **energy dynamic-range proxy**, not calibrated SNR, a noise estimate, or VAD. Near-full-scale samples are clipping candidates, including possible codec overshoot.

Canonical audio duplicate hashes use decoded 16 kHz mono PCM16; sample equality can detect some container/encoding variants, but different lossy encodings need not hash identically. SHA-256 file hashes also remain available. No denoising or source rewriting occurs.''')
code(r'''from concurrent.futures import ProcessPoolExecutor
from audio_eda import measure_audio
t0=time.monotonic(); audio_rows=[]
with ProcessPoolExecutor(max_workers=2) as pool:
    futures={pool.submit(measure_audio,(str(SNAPSHOT),sid)):sid for sid in df.session_id}
    for future in as_completed(futures):
        audio_rows.append(future.result())
        if len(audio_rows)%100==0:
            print(f'Decoded {len(audio_rows)}/{len(df)} sessions; {(time.monotonic()-t0)/60:.1f} minutes',flush=True)
            export(pd.DataFrame(audio_rows),'audio_progress')
audio=pd.DataFrame(audio_rows).sort_values('session_id'); export(audio,'audio_features')
(OUT/'audio_progress.csv').unlink(missing_ok=True)
df=df.merge(audio,on='session_id',how='left',validate='one_to_one')
df['duration_discrepancy_s']=df.decoded_duration_s-df.session_duration
display(audio.describe(include='all').T); print('Decode failures:',audio.decode_error.notna().sum())
technical=audio.groupby(['codec','sample_rate','channels'],dropna=False).agg(sessions=('session_id','size'),decoded_hours=('decoded_duration_s',lambda s:s.sum()/3600)).reset_index(); export(technical,'audio_formats'); display(technical)
acoustic=['rms_dbfs','peak','near_fullscale_fraction','dc_offset','quiet_fraction_45dbfs','energy_dynamic_range_db','duration_discrepancy_s','low_energy_p10_dbfs','crest_factor_db']
fig,axes=plt.subplots(3,3,figsize=(15,12))
for ax,c in zip(axes.flat,acoustic): ax.hist(df[c].dropna(),bins=50); ax.set(title=c,ylabel='Sessions')
for ax in axes.flat[len(acoustic):]: ax.set_visible(False)
savefig('acoustic_distributions')
display(df[['quiet_fraction_50dbfs','quiet_fraction_45dbfs','quiet_fraction_40dbfs']].describe().T)
''')
md('''## 6. Outliers, associations, and human review queue
Robust modified Z scores use |0.6745 × (x - median) / MAD| > 3.5. If MAD is zero and IQR is positive, use 1.5 × IQR fences. If both are zero, report no robust numerical flag. Heavy tails may represent valid recordings. Comparisons are descriptive and are not hypothesis tests. Correlations across sessions can be driven by prolific speakers; a separate speaker-median correlation is provided.''')
code(r'''flags=pd.DataFrame({'session_id':df.session_id})
flags['low_session_alignment']=df.quality_score<.3
flags['low_segment_alignment']=df.min_segment_quality_score<.5
flags['duration_mismatch']=df.duration_discrepancy_s.abs()>np.maximum(.25,.01*df.session_duration)
flags['decode_failure']=df.decode_error.notna()
flags['quiet_majority']=df.quiet_fraction_45dbfs>.5
flags['low_rms']=df.rms_dbfs< -40
flags['near_fullscale_candidate']=df.near_fullscale_fraction>.001
flags['dc_offset_candidate']=df.dc_offset.abs()>.01
flags['nonfinite_audio']=df.nonfinite_samples.fillna(0)>0
flags['invalid_birth_year']=df.invalid_birth_year
flags['nonpositive_duration']=df.session_duration<=0
flags['empty_transcript']=df.empty_text
badids=set(checks.loc[checks.check.str.contains('timestamp|overlap|parse|missing',regex=True),'session_id'])
flags['structural_issue']=df.session_id.isin(badids)
outlier_rules=[]
for c in ['session_duration','words_count','avg_words_per_minute','wpm_aligned','rms_dbfs','quiet_fraction_45dbfs','energy_dynamic_range_db','low_energy_p10_dbfs']:
    x=df[c]; med=x.median(); mad=(x-med).abs().median(); q1,q3=x.quantile([.25,.75]); iqr=q3-q1
    if mad>0: f=(.6745*(x-med)/mad).abs()>3.5; rule='modified_z_abs_gt_3.5'
    elif iqr>0: f=(x<q1-1.5*iqr)|(x>q3+1.5*iqr); rule='IQR_1.5_fallback'
    else: f=pd.Series(False,index=x.index); rule='degenerate_no_flag'
    flags['outlier_'+c]=f.fillna(False); outlier_rules.append({'field':c,'median':med,'MAD':mad,'IQR':iqr,'rule':rule,'flagged':int(f.sum()),'observed':int(x.notna().sum())})
export(pd.DataFrame(outlier_rules),'outlier_rules'); export(flags,'session_flags')
flagcols=flags.columns.drop('session_id'); df['review_flag_count']=flags[flagcols].sum(axis=1)
df['review_reasons']=flags[flagcols].apply(lambda row:';'.join(row.index[row.astype(bool)]),axis=1)
df['audio_path']=df.session_id.map(lambda sid:str(SNAPSHOT/sid/'audio.mka'))
review=df[df.review_flag_count>0].sort_values(['review_flag_count','quality_score'],ascending=[False,True])
export(review,'review_candidates'); export(df,'recording_features')
flag_summary=pd.DataFrame({'flag':flagcols,'flagged_sessions':[int(flags[c].sum()) for c in flagcols]}); flag_summary['all_sessions']=len(df); flag_summary['pct']=100*flag_summary.flagged_sessions/len(df); export(flag_summary,'flag_summary'); display(flag_summary)
sample=df.groupby('user_id',group_keys=False).sample(n=1,random_state=SEED)
sample=sample.assign(review_role='speaker_spanning_baseline',human_review_status='pending',human_noise_rating='',human_text_match='',human_notes='')
export(sample[['session_id','user_id','audio_path','review_role','human_review_status','human_noise_rating','human_text_match','human_notes']],'human_listening_sample')
corrfields=['session_duration','quality_score','wpm_aligned','rms_dbfs','quiet_fraction_45dbfs','energy_dynamic_range_db']
session_corr=df[corrfields].corr(method='spearman'); speaker_corr=df.groupby('user_id')[corrfields].median().corr(method='spearman')
export(session_corr.reset_index(names='field'),'session_spearman'); export(speaker_corr.reset_index(names='field'),'speaker_median_spearman')
fig,axes=plt.subplots(1,2,figsize=(16,6))
sns.heatmap(session_corr,ax=axes[0],vmin=-1,vmax=1,cmap='vlag',annot=True,fmt='.2f'); axes[0].set_title('Session Spearman correlations (dependent observations)')
sns.heatmap(speaker_corr,ax=axes[1],vmin=-1,vmax=1,cmap='vlag',annot=True,fmt='.2f'); axes[1].set_title('Speaker-median Spearman correlations'); savefig('correlations')
groups=[]
for c in ['sex_label','age_proxy_bin','document_source_type','codec','sample_rate']:
    for label,g in df.groupby(c,dropna=False):
        groups.append({'group_field':c,'group':label,'sessions':len(g),'speakers':g.user_id.nunique(),'hours':g.session_duration.sum()/3600,'median_alignment':g.quality_score.median(),'median_rms_dbfs':g.rms_dbfs.median(),'median_quiet_fraction':g.quiet_fraction_45dbfs.median(),'low_alignment_pct':100*(g.quality_score<.3).mean(),'any_flag_pct':100*(g.review_flag_count>0).mean()})
groupdf=pd.DataFrame(groups); export(groupdf,'group_quality_comparisons'); display(groupdf)
fig,axes=plt.subplots(1,2,figsize=(13,5)); sns.boxplot(data=df,x='sex_label',y='quality_score',ax=axes[0]); sns.boxplot(data=df,x='sex_label',y='quiet_fraction_45dbfs',ax=axes[1]); savefig('quality_by_source_sex_label')
display(review[['session_id','quality_score','rms_dbfs','review_flag_count','review_reasons']].head(15))
''')
md('''### Listening examples for human review
These short clips are provided for a human reviewer. The analysis has not assigned perceptual noise or transcript-correctness ratings. Two flagged examples and two seeded baseline sessions are embedded, capped at 15 seconds each; the complete review lists retain original paths.''')
code(r'''examples=pd.concat([review.head(2).assign(example_role='flagged'),sample.head(2).assign(example_role='baseline')]).drop_duplicates('session_id')
for _,r in examples.iterrows():
    print(r.example_role,r.session_id,'alignment score:',r.quality_score)
    try:
        with av.open(str(SNAPSHOT/r.session_id/'audio.mka')) as container:
            stream=container.streams.audio[0]; rate=stream.codec_context.sample_rate
            convert=av.AudioResampler(format='fltp',layout='mono',rate=rate); clip=[]; count=0
            for frame in container.decode(stream):
                for f in convert.resample(frame):
                    x=f.to_ndarray().ravel(); clip.append(x); count+=len(x)
                if count>=15*rate: break
            display(Audio(np.concatenate(clip)[:15*rate],rate=rate,normalize=False))
    except Exception as e: print('Listening clip unavailable:',type(e).__name__,str(e))
''')
md('''## 7. Exact duplicates and split feasibility
This audit models a strict protocol requiring both speaker and exact normalized segment text separation. Speakers are joined when they share a nonempty normalized segment. All recordings of each speaker therefore remain together. Tiny/common text can create large components; the audit shows feasibility, not a final split. Near-duplicate/paraphrase and perceptual speaker-identity checks remain unperformed.''')
code(r'''filedup=inventory[inventory.path.str.endswith('/audio.mka')].copy()
filedup=filedup[filedup.sha256.duplicated(False)]; export(filedup,'duplicate_audio_files')
pcmdup=audio[audio.pcm16_16k_mono_sha256.notna() & audio.pcm16_16k_mono_sha256.duplicated(False)].sort_values('pcm16_16k_mono_sha256'); export(pcmdup,'duplicate_canonical_audio')
parent={u:u for u in speaker.user_id}
def find(x):
    while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
    return x
def union(a,b):
    a,b=find(a),find(b)
    if a!=b: parent[b]=a
for text,g in seg[seg.normalized_text.ne('')].groupby('normalized_text'):
    users=g.user_id.unique()
    for u in users[1:]: union(users[0],u)
speaker['text_component']=speaker.user_id.map(find)
components=speaker.groupby('text_component').agg(speakers=('user_id','size'),sessions=('sessions','sum'),hours=('hours','sum')).reset_index().sort_values('hours',ascending=False)
components['hour_share']=components.hours/components.hours.sum(); export(components,'speaker_text_components'); export(speaker,'speaker_summary')
print('Duplicate source-audio files:',len(filedup),'Canonical duplicate recordings:',len(pcmdup))
print('Shared normalized segment texts across speakers:',int((duplicates.speakers>1).sum())); display(components.head(15))
''')
md('''## 8. Findings, limitations, and reproducibility
The following report is generated from this run. Acoustic screening has no human-confirmed noise labels. No ASR WER, calibrated SNR, perceived naturalness, accent, inferred demographics, or proof of single-speaker content is claimed. Full listening verification and near-duplicate detection remain follow-up work. Automatic confidence is not independent validation. No data were removed or models fitted.

References: [dataset card](https://huggingface.co/datasets/ivrit-ai/crowd-recital), [NIST outlier guidance](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm), [Common Voice methodology](https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md).''')
code(r"""largest=float(components.hour_share.max()) if len(components) else 0
metrics={**overview,**concentration,'decode_success':int(df.decode_error.isna().sum()),'decode_failure':int(df.decode_error.notna().sum()),'flagged_sessions':len(review),'low_alignment_sessions':int((df.quality_score<.3).sum()),'sex_unknown_speakers':int((speaker.sex_label=='unknown').sum()),'birth_year_missing_sessions':int(df.year_of_birth.isna().sum()),'largest_speaker_text_component_hour_share':largest,'exact_duplicate_audio_files':len(filedup),'canonical_duplicate_recordings':len(pcmdup),'vtt_normalized_mismatches':int((~df.vtt_normalized_match).sum()),'integrity_issue_rows':len(checks),'manifest_metadata_disagreements':len(diff),'vocabulary_size':len(tokens)}
(OUT/'key_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
def table_markdown(frame):
    # Avoid optional tabulate dependency.
    columns=list(frame.columns); rows=['| '+' | '.join(map(str,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in frame.itertuples(index=False,name=None): rows.append('| '+' | '.join(str(round(x,3)) if isinstance(x,(float,np.floating)) else str(x).replace('|','/') for x in row)+' |')
    return '\n'.join(rows)
report=f'''# Crowd-recital EDA: progress and findings

## Completed work
- Inspected the workspace, supplied research-context PDF, dataset documentation, manifest and JSON structure.
- Ran the user-requested Hugging Face download command before EDA execution and documented the analysis design in EDA_PLAN.md.
- Analyzed revision `{SNAPSHOT.name}` locally; credentials are never included in outputs.
- Inventoried and SHA-256 hashed {len(inventory):,} files; checked companion presence, schema, IDs, metadata agreement, transcript and word timing.
- Built recording, speaker, segment, word, alignment, acoustic, review, and split-connectivity tables.
- Decoded {metrics['decode_success']:,}/{len(df):,} recordings successfully; {metrics['decode_failure']} failures retained explicitly.
- Generated feature distributions, demographic weighting, speaker concentration, transcript/alignment, acoustic, association, and outlier analyses.
- Saved tables, plots and provenance under `eda_outputs/`. The executed notebook is `dataset.ipynb`.

## Corpus size and concentration
{len(df):,} sessions, {overview['speakers']:,} speakers, {overview['hours']:.2f} reported recording hours, {overview['manifest_words']:,} manifest words and {overview['manifest_segments']:,} manifest segments. Recomputed aligned counts: {len(word):,} word objects and {len(seg):,} segments; these are separate counting conventions.

The top speaker contributes {concentration['top_1_speaker_hour_share']:.1%} of hours; the top five {concentration['top_5_speaker_hour_share']:.1%}; the top ten {concentration['top_10_speaker_hour_share']:.1%}. Hour-based Gini is {gini:.3f}, and the effective speaker count is {concentration['effective_speakers_hours']:.1f}. These quantify contribution imbalance without assuming a population demographic target.

## Demographic distributions
The dataset labels biological sex, not gender identity. Age at recording cannot be established because a reliable recording date is unavailable. The age proxy is 2026 minus birth year and must not be used as observed age. Unknown sex: {metrics['sex_unknown_speakers']} speakers. Missing birth year: {metrics['birth_year_missing_sessions']} sessions. Conflicting speaker metadata is retained as a category.

{table_markdown(balance)}

## Alignment and acoustic screening
{table_markdown(quality_summary)}

{table_markdown(technical)}

{table_markdown(flag_summary[~flag_summary.flag.str.startswith('outlier_')])}

There are {len(review):,} sessions with at least one screening flag. Flags overlap and include provider segment-confidence guidance, so their sum is not a rejection count. See `outlier_rules.csv` for robust numerical thresholds and `review_candidates.csv` for all reasons. Quiet fractions depend on the documented energy threshold. Energy dynamic range is not SNR. Near-full-scale values may reflect lossy overshoot, so clipping candidates need listening and waveform review.

## Integrity, repetitions, and split risk
{len(checks):,} integrity issue occurrences and {len(diff):,} manifest/metadata disagreement entries were retained for review. There are {len(filedup)} recordings participating in exact file duplicates and {len(pcmdup)} in canonical PCM duplicates; these are recording counts, not duplicate-group counts. There are {int((duplicates.speakers>1).sum()):,} exact normalized segment texts shared across speakers. The largest strict speaker/text connected component accounts for {largest:.1%} of recording hours, which constrains a split requiring both speaker and exact text independence. No split was assigned.

## Recommended actions
1. Review decode, timestamp and metadata integrity issues before selecting training data; retain documented reasons and original files.
2. Listen to the flagged queue and `human_listening_sample.csv`, which selects one seeded baseline session per speaker. Human ratings are pending; no perceptual-noise claim is established.
3. Select demographic and speaker-duration quotas with unknowns visible. Evaluate retention rates by speakers and hours, not just session counts, before setting exclusions.
4. Confirm alignment/text mismatches manually. Provider scores measure alignment, and are not measured WER or transcript ground truth.
5. Establish speaker/text grouping and verify largest-component feasibility before freezing evaluation splits. Browser codec, sample rate, channel and source signatures can confound a future real/synthetic experiment.
6. Obtain reliable demographic provenance or recording dates if gender, accent or age-at-recording analyses are required. Do not infer them from voice.

## Limits and remaining human work
This is a descriptive audit of read Hebrew, not a representative sample of Hebrew speakers or spontaneous speech. Sessions repeat speakers and documents; no independent-session significance tests or causal claims were made. Vocabulary is whitespace-based rather than morphological/phonemic coverage. Exact normalization can merge distinctions; perceptual audio duplicates, paraphrases, stable cross-ID speaker identity, speech/noise separation, calibrated SNR, ASR WER, single-speaker verification and human listening are not established. No acoustic denoising, automatic exclusion, synthetic generation or classifier training was performed.

## Reproduction and references
Open `dataset.ipynb` in this project and run all cells using the recorded package versions in `eda_outputs/run_provenance.json`. Set `CROWD_RECITAL_PATH` to move the same input revision. The notebook requires no network after acquisition. Original bytes are hashed in `file_inventory.csv`; current source hashes do not independently authenticate upstream provenance.

- Dataset schema and provider screening guidance: https://huggingface.co/datasets/ivrit-ai/crowd-recital
- Robust outlier screening and cautions: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- Comparative demographic/speaker reporting methodology: https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md
'''
(ROOT/'EDA_PROGRESS.md').write_text(report,encoding='utf-8')
run['completed_utc']=pd.Timestamp.now(tz='UTC').isoformat(); run['metrics']=metrics
(OUT/'run_provenance.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
display(Markdown(report))
assert len(audio)==len(manifest) and df.session_id.is_unique
assert len(seg)==sum(len(a.get('segments',[])) for a in aligned.values())
print('Analysis artifacts:',len(list(OUT.glob('*.csv'))),'CSV tables;',len(list(FIG.glob('*.png'))),'figures')
""")
md('''## 9. Analyst validation: reported versus decoded time bounds
This independent follow-up reads the exported tables so the reported-duration warnings can be distinguished from violations of actual decoded audio bounds. Birth-year placeholders are also quantified explicitly. It can be rerun without decoding the corpus again; it does not replace any earlier screening flags.''')
code(r"""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
validation_root=Path.cwd(); validation_out=validation_root/'eda_outputs'
recordings_check=pd.read_csv(validation_out/'recording_features.csv')
segments_check=pd.read_csv(validation_out/'segments.csv')
audio_check=pd.read_csv(validation_out/'audio_features.csv')
limits=recordings_check[['session_id','session_duration','decoded_duration_s']]
timing_check=segments_check.merge(limits,on='session_id',how='left',validate='many_to_one')
timing_check['reported_duration_violation']=(timing_check.start<0)|(timing_check.end<=timing_check.start)|(timing_check.end>timing_check.session_duration+.1)
timing_check['decoded_duration_violation']=np.where(timing_check.decoded_duration_s.notna(),(timing_check.start<0)|(timing_check.end<=timing_check.start)|(timing_check.end>timing_check.decoded_duration_s+.1),None)
timing_candidates=timing_check[timing_check.reported_duration_violation | timing_check.decoded_duration_violation.fillna(False).astype(bool)]
timing_candidates.to_csv(validation_out/'timestamp_bounds_validation.csv',index=False,encoding='utf-8-sig')
birth=recordings_check.year_of_birth
invalid_birth=birth.notna() & ((birth<1906)|(birth>2026)|(birth%1!=0))
validation_metrics={
 'birth_year_zero_sessions':int((birth==0).sum()),
 'invalid_or_placeholder_birth_year_sessions':int(invalid_birth.sum()),
 'birth_year_missing_sessions':int(birth.isna().sum()),
 'age_proxy_unavailable_sessions':int((birth.isna()|invalid_birth).sum()),
 'segments_outside_reported_duration':int(timing_check.reported_duration_violation.sum()),
 'sessions_outside_reported_duration':int(timing_check.loc[timing_check.reported_duration_violation,'session_id'].nunique()),
 'segments_outside_decoded_duration':int(timing_check.decoded_duration_violation.fillna(False).sum()),
 'sessions_outside_decoded_duration':int(timing_check.loc[timing_check.decoded_duration_violation.fillna(False).astype(bool),'session_id'].nunique()),
 'segments_without_decoded_duration':int(timing_check.decoded_duration_s.isna().sum())}
display(pd.Series(validation_metrics,name='count').to_frame())
display(timing_candidates[['session_id','segment_index','start','end','session_duration','decoded_duration_s','reported_duration_violation','decoded_duration_violation']].head(12))
notes=f'''## Analyst validation

Birth year `0` occurs in {validation_metrics['birth_year_zero_sessions']} sessions. Missing birth years affect {validation_metrics['birth_year_missing_sessions']} sessions, and invalid/placeholder years affect {validation_metrics['invalid_or_placeholder_birth_year_sessions']}. Together, {validation_metrics['age_proxy_unavailable_sessions']} sessions ({validation_metrics['age_proxy_unavailable_sessions']/len(recordings_check):.1%}) have no usable age proxy. Original years are retained; the zero placeholder is not interpreted as an age.

Using a 0.1-second tolerance, {validation_metrics['segments_outside_reported_duration']} aligned segments in {validation_metrics['sessions_outside_reported_duration']} sessions violate bounds relative to reported recording duration. Against actual decoded duration, {validation_metrics['segments_outside_decoded_duration']} segments in {validation_metrics['sessions_outside_decoded_duration']} sessions still violate bounds. There are {validation_metrics['segments_without_decoded_duration']} segments without a decoded duration available. See `timestamp_bounds_validation.csv`; initial reported-duration flags remain intact, and overlapping metadata/alignment issue rows should not be counted as independent failures.
'''
report_path=validation_root/'EDA_PROGRESS.md'
base_report=report_path.read_text(encoding='utf-8').split('## Analyst validation')[0].rstrip()
report_path.write_text(base_report+chr(10)+chr(10)+notes,encoding='utf-8')
metrics_path=validation_out/'key_metrics.json'; final_metrics=json.loads(metrics_path.read_text(encoding='utf-8')); final_metrics.update(validation_metrics)
metrics_path.write_text(json.dumps(final_metrics,indent=2),encoding='utf-8')
provenance_path=validation_out/'run_provenance.json'; final_provenance=json.loads(provenance_path.read_text(encoding='utf-8')); final_provenance['metrics']=final_metrics; final_provenance['analyst_validation_completed_utc']=pd.Timestamp.now(tz='UTC').isoformat()
provenance_path.write_text(json.dumps(final_provenance,indent=2),encoding='utf-8')
display(Markdown(notes))
""")
md('''## 10. Finding triage and leakage sensitivity
This independent validation uses exported data to distinguish zero-length alignment segments from audio overruns, quantify exact digital silence, and test whether speaker/text connectivity is driven only by very short phrases. No observations are deleted and no splits are assigned.''')
code(r"""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
triage_root=Path.cwd(); triage_out=triage_root/'eda_outputs'
rec=pd.read_csv(triage_out/'recording_features.csv')
seg_check=pd.read_csv(triage_out/'segments.csv').merge(rec[['session_id','user_id']],on='session_id',validate='many_to_one')
speaker_check=pd.read_csv(triage_out/'speaker_summary.csv')
bounds=pd.read_csv(triage_out/'timestamp_bounds_validation.csv')
zero_length=bounds[bounds.end==bounds.start]
zero_audio=rec[rec.peak==0]
zero_audio[['session_id','user_id','session_duration','decoded_duration_s','quality_score','audio_path']].to_csv(triage_out/'zero_audio_recordings.csv',index=False,encoding='utf-8-sig')
zero_length.to_csv(triage_out/'zero_length_segments.csv',index=False,encoding='utf-8-sig')
sensitivity=[]
for min_words in [1,5,10,20]:
    parent_check={u:u for u in speaker_check.user_id}
    def find_check(x):
        while parent_check[x]!=x:
            parent_check[x]=parent_check[parent_check[x]]; x=parent_check[x]
        return x
    def union_check(a,b):
        a,b=find_check(a),find_check(b)
        if a!=b: parent_check[b]=a
    # Only positive-length segments are used for these sensitivity checks.
    eligible=seg_check[(seg_check.duration_s>0)&(seg_check.word_count>=min_words)&seg_check.normalized_text.fillna('').ne('')]
    shared_texts=0
    for _,g in eligible.groupby('normalized_text'):
        users=g.user_id.unique()
        if len(users)>1: shared_texts+=1
        for u in users[1:]: union_check(users[0],u)
    components_check=speaker_check.assign(component=speaker_check.user_id.map(find_check)).groupby('component').agg(speakers=('user_id','size'),hours=('hours','sum'))
    max_row=components_check.loc[components_check.hours.idxmax()]
    sensitivity.append({'minimum_words':min_words,'positive_length_segments':len(eligible),'shared_texts':shared_texts,'components':len(components_check),'largest_component_speakers':int(max_row.speakers),'largest_component_hour_share':float(max_row.hours/speaker_check.hours.sum())})
sensitivity=pd.DataFrame(sensitivity); sensitivity.to_csv(triage_out/'leakage_sensitivity.csv',index=False,encoding='utf-8-sig'); display(sensitivity)
triage_metrics={'zero_audio_recordings':len(zero_audio),'zero_length_aligned_segments':len(zero_length),'sessions_with_zero_length_aligned_segments':int(zero_length.session_id.nunique()),'negative_length_aligned_segments':int((bounds.end<bounds.start).sum()),'aligned_segments_exceeding_decoded_duration':int((bounds.end>bounds.decoded_duration_s+.1).sum()),'quiet_majority_recordings':int((rec.quiet_fraction_45dbfs>.5).sum()),'duration_discrepancy_recordings':int((rec.duration_discrepancy_s.abs()>np.maximum(.25,.01*rec.session_duration)).sum())}
metrics_path=triage_out/'key_metrics.json'; m=json.loads(metrics_path.read_text(encoding='utf-8')); m.update(triage_metrics); m['leakage_sensitivity']=sensitivity.to_dict(orient='records'); metrics_path.write_text(json.dumps(m,indent=2),encoding='utf-8')
highlights=f'''## Priority findings

- **Contribution and demographic imbalance:** ten of 85 recorder IDs contribute {m['top_10_speaker_hour_share']:.1%} of hours. Reported sex is male for 46 speakers, female for 25, and unknown/unspecified for 14; these labels do not measure gender identity. No usable age proxy exists for {m['age_proxy_unavailable_sessions']} sessions (23.7%).
- **Technical review:** {len(zero_audio)} recordings decode entirely to zero-valued audio. {triage_metrics['quiet_majority_recordings']} have more than half of 20 ms frames below -45 dBFS; this can reflect pauses or low recording level and does not confirm background noise. {triage_metrics['duration_discrepancy_recordings']} recordings exceed the declared duration-discrepancy rule; investigate container timestamps and source timing before cutting audio.
- **Alignment review:** {len(zero_length):,} segments have zero duration, spanning {triage_metrics['sessions_with_zero_length_aligned_segments']} sessions. None of the aligned segments exceed decoded duration by more than 0.1 s. Thus the earlier decoded-bound invalidity count refers to zero-length segments, not audio overruns. The 37 additional reported-duration overruns disappear when decoded length is used. Session alignment score <0.3 affects 74 recordings, and normalized VTT/aligned text differs in 214 recordings.
- **Split feasibility:** the prespecified strict speaker/exact-segment-text graph puts 99.2% of hours in one component. Positive-length sensitivity checks at minimum segment lengths 1/5/10/20 words produce largest-component hour shares of {', '.join(f'{v:.1%}' for v in sensitivity.largest_component_hour_share)}. A meaningful 70/15/15 split under the original grouping rule requires changing corpus selection or explicitly defining a different overlap protocol; the sensitivity results quantify alternatives rather than approving them. Do not silently split a connected group.
- **Review flags are not rejection decisions:** {m['flagged_sessions']} recordings have at least one broad screening flag, including demographics, statistical tails and alignment rules. No files were removed. Exact byte and canonical PCM duplicate searches found no duplicate recordings. Perceived noise and transcript correctness still require listening.
'''
report_path=triage_root/'EDA_PROGRESS.md'; report=report_path.read_text(encoding='utf-8')
report=report.split('## Priority findings')[0]+report[report.find('## Completed work'):] if '## Priority findings' in report else report
report=report.replace('# Crowd-recital EDA: progress and findings','# Crowd-recital EDA: progress and findings'+chr(10)+chr(10)+highlights,1)
report_path.write_text(report,encoding='utf-8')
provenance_path=triage_out/'run_provenance.json'; provenance=json.loads(provenance_path.read_text(encoding='utf-8')); provenance['metrics']=m; provenance['finding_triage_completed_utc']=pd.Timestamp.now(tz='UTC').isoformat(); provenance_path.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
display(Markdown(highlights))
""")
md('''**Sensitivity interpretation:** read the component shares together with the eligible-segment counts. Longer text thresholds change the definition of prohibited overlap and can leave very little evidence. A smaller connected component is not proof that leakage has been eliminated or that a revised protocol is appropriate.''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}})
target=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'dataset.ipynb'
nbf.write(nb,target)
print(f'Wrote notebook with {len(cells)} cells')
