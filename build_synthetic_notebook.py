"""Author the offline, reproducible synthetic-pilot EDA notebook."""
from pathlib import Path
import nbformat as nb

cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))

md('''# Hebrew synthetic-voice pilot EDA

One fixed Hebrew sentence per selected stock model/voice. This is a controlled voice inventory, not a 20-hour corpus or a representative demographic survey.

**Keep three distinct concepts separate:** provider voice labels; unvalidated audio-model estimates of perceived presentation/age; and human listening ratings. A synthetic voice has no biological age or sex. Perceived age bins are casting impressions from a few seconds, not measured ages. Unknown and conflicting assessments stay visible. Models sharing a provider/voice identifier are dependent observations.

Generation uses the OpenRouter speech API. Automated review is blind to the intended sentence and to model/voice IDs. No audio is generated and no API calls are made when executing this notebook.''')
code(r'''from pathlib import Path
import json, hashlib, sys, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
ROOT=Path.cwd(); BASE=ROOT/'synthetic_pilot'; ANALYSIS=BASE/'analysis'; FIG=ANALYSIS/'figures'
FIG.mkdir(parents=True,exist_ok=True)
planned=pd.read_csv(BASE/'planned_samples.csv',keep_default_na=False)
generated=pd.read_csv(BASE/'manifest.csv',keep_default_na=False)
reviews=pd.read_csv(BASE/'automated_review.csv',keep_default_na=False)
assert planned.sample_id.is_unique and generated.sample_id.is_unique and reviews.sample_id.is_unique
df=planned.merge(generated.drop(columns=[c for c in generated if c in planned and c!='sample_id']),on='sample_id',how='left',validate='one_to_one')
df=df.drop(columns=['language_qc_status'],errors='ignore').merge(reviews,on='sample_id',how='left',validate='one_to_one',suffixes=('','_review'))
for c in ['technical_qc_pass','language_qc_pass']:
    df[c]=df[c].map(lambda x:x is True or str(x).lower()=='true')
df['pilot_automatic_qc_pass']=df.technical_qc_pass & df.language_qc_pass
for c in ['duration_s','rms_dbfs','peak','near_fullscale_fraction','quiet_fraction_45dbfs','energy_dynamic_range_db','normalized_character_error_rate','presentation_confidence','age_confidence']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df['presentation_for_summary']=df.presentation_for_summary.fillna('uncertain').replace('','uncertain')
df['age_for_summary']=df.age_for_summary.fillna('uncertain').replace('','uncertain')
df['raw_age_for_summary']=df.age_for_summary
age_estimator_uninformative=bool(df.review_status.eq('reviewed').sum()>=20 and df.loc[df.review_status.eq('reviewed'),'age_for_summary'].nunique()<=1)
if age_estimator_uninformative: df['age_for_summary']='uniform_estimator_not_validated'
df['nominal_prompt_wpm']=60*df.text.str.split().str.len()/df.duration_s
def export(frame,name): frame.to_csv(ANALYSIS/(name+'.csv'),index=False,encoding='utf-8-sig')
def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight'); plt.show(); plt.close()
def markdown_table(frame):
    result=['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None): result.append('| '+' | '.join(str(round(x,3)) if isinstance(x,float) else str(x).replace('|','/') for x in row)+' |')
    return chr(10).join(result)
sns.set_theme(style='whitegrid'); plt.rcParams.update({'figure.figsize':(12,5),'figure.dpi':110})
display(df[['model','voice','generation_status','provider_gender_label','provider_age_label','language_qc_status']].head(12))
export(df,'sample_features')
''')
md('''## Coverage, completeness and provenance
Count provider/voice groups separately from model/voice configurations. A shared name across versions is conservatively treated as one identity group; it is not a biometric identity guarantee. Raw API audio and analysis WAV are separately hashed. Failed and unverified samples remain in the audit.''')
code(r'''coverage=df.groupby(['family','model']).agg(planned=('sample_id','size'),generated=('generation_status',lambda s:s.isin(['generated','decoded']).sum()),decoded=('generation_status',lambda s:s.eq('decoded').sum()),technical_pass=('technical_qc_pass','sum'),automatic_hebrew_pass=('language_qc_pass','sum'),combined_pass=('pilot_automatic_qc_pass','sum'),seconds=('duration_s','sum')).reset_index()
coverage['minutes']=coverage.seconds/60
export(coverage,'model_coverage'); display(coverage)
missing=pd.DataFrame({'field':df.columns,'missing_or_blank':[int((df[c].isna()|df[c].eq('')).sum()) for c in df]})
export(missing,'missingness')
inventory=[]
for row in df.itertuples():
    for kind,relative,expected in [('original',getattr(row,'original_path',''),getattr(row,'original_sha256','')),('wav',getattr(row,'wav_path',''),getattr(row,'wav_sha256',''))]:
        if isinstance(relative,str) and relative:
            p=BASE/relative; actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            inventory.append({'sample_id':row.sample_id,'kind':kind,'path':relative,'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else 0,'sha256':actual,'matches_generation_hash':actual==expected})
inventory=pd.DataFrame(inventory); export(inventory,'file_inventory')
assert inventory.exists.all() and inventory.matches_generation_hash.all()
duplicates=df[df.wav_sha256.duplicated(False)].sort_values('wav_sha256')
export(duplicates,'exact_analysis_audio_duplicates')
overview={'planned_samples':len(planned),'attempted_samples':len(generated),'decoded_samples':int(df.generation_status.eq('decoded').sum()),'models':df.model.nunique(),'families':df.family.nunique(),'provider_voice_groups':df.voice_identity_group.nunique(),'minutes_generated':float(df.duration_s.sum()/60),'technical_qc_pass':int(df.technical_qc_pass.sum()),'automatic_hebrew_match_pass':int(df.language_qc_pass.sum()),'combined_qc_pass':int(df.pilot_automatic_qc_pass.sum()),'human_verified_samples':0,'exact_pcm_duplicate_recordings':len(duplicates)}
display(pd.Series(overview).to_frame('value'))
ax=coverage.set_index('model')[['planned','combined_pass']].plot.barh(figsize=(12,7)); ax.set(xlabel='Model/voice configurations',title='Coverage and automatic Hebrew/audio screening'); savefig('coverage')
''')
md('''## Provider labels and perceptual estimates
Provider gender/age labels are read only from retrievable official metadata, not guessed from names. Audio-model outputs are subjective, not calibrated probabilities. For the displayed perceptual summaries, confidence below 0.6 becomes uncertain. Conflicting estimates for the same provider/voice across model versions are explicitly marked.

The review model transcribes the audio independently of the intended text. It also judges perceived casting age, but these estimates remain unvalidated: the model has not been calibrated on Hebrew voices with human ratings. They must not be interpreted as demographic ground truth.''')
code(r'''valid=df[df.pilot_automatic_qc_pass].copy()
def conservative_category(series):
    values=set(series.dropna().replace('','uncertain'))
    known=values-{'uncertain'}
    if len(known)>1: return 'conflicting_across_models'
    if 'uncertain' in values or not known: return 'uncertain'
    return next(iter(known))
def source_category(series):
    values=set(series.fillna('unknown').replace('','unknown'))
    return next(iter(values)) if len(values)==1 else 'conflicting_provider_labels'
voices=[]
for identity,g in df.groupby('voice_identity_group'):
    good=g[g.pilot_automatic_qc_pass]
    voices.append({'voice_identity_group':identity,'family':g.family.iloc[0],'voice':g.voice.iloc[0],
                   'configurations':len(g),'valid_configurations':len(good),
                   'provider_gender_label':source_category(g.provider_gender_label),
                   'provider_age_label':source_category(g.provider_age_label),
                   'perceived_presentation':conservative_category(good.presentation_for_summary) if len(good) else 'not_validated',
                   'perceived_age_band':conservative_category(good.age_for_summary) if len(good) else 'not_validated'})
voice_summary=pd.DataFrame(voices); export(voice_summary,'voice_summary'); display(voice_summary.head(15))
distributions=[]
for unit,frame,fields in [('model_voice_configurations',valid,['presentation_for_summary','age_for_summary']),('provider_voice_groups',voice_summary,['perceived_presentation','perceived_age_band','provider_gender_label','provider_age_label'])]:
    for field in fields:
        counts=frame[field].value_counts(dropna=False)
        for value,count in counts.items(): distributions.append({'unit':unit,'field':field,'category':value,'count':int(count),'denominator':len(frame),'pct':100*count/len(frame) if len(frame) else np.nan})
distributions=pd.DataFrame(distributions); export(distributions,'voice_distributions'); display(distributions)
raw_age=df.raw_age_for_summary.value_counts().rename_axis('raw_model_age_category').reset_index(name='configurations'); export(raw_age,'raw_age_model_outputs'); display(raw_age)
if age_estimator_uninformative: display(Markdown('**Age estimation failed an informativeness check:** the audio model assigned every reviewed clip the same age category. This is not accepted as an age distribution. Use provider labels where supplied and human listening ratings; raw model outputs remain in the audit.'))
fig,axes=plt.subplots(1,2,figsize=(16,6))
for ax,field,title in [(axes[0],'perceived_presentation','Perceived voice presentation (unvalidated audio model)'),(axes[1],'perceived_age_band','Age estimator uninformative: human review required' if age_estimator_uninformative else 'Perceived casting age (unvalidated; not actual age)')]:
    voice_summary[field].value_counts().plot.bar(ax=ax,color='#367d9c'); ax.set(title=title,ylabel='Provider/voice groups',xlabel=''); ax.tick_params(axis='x',labelrotation=35)
savefig('perceptual_voice_groups')
fig,axes=plt.subplots(1,2,figsize=(15,6))
for ax,field in zip(axes,['provider_gender_label','provider_age_label']):
    voice_summary[field].value_counts().plot.bar(ax=ax,color='#668d66'); ax.set(title=field+' (source metadata)',ylabel='Provider/voice groups',xlabel=''); ax.tick_params(axis='x',labelrotation=35)
savefig('provider_labels')
fig,ax=plt.subplots(figsize=(13,7)); pd.crosstab(valid.model,valid.presentation_for_summary,normalize='index').mul(100).plot.barh(stacked=True,ax=ax); ax.set(title='Per-model perceived presentation; repeated voices across versions',xlabel='Percent of automatically screened configurations'); savefig('presentation_by_model')
''')
md('''## Acoustic distributions and outlier candidates
Analysis uses 16 kHz mono WAV, 20 ms nonoverlapping energy frames and a -45 dBFS quiet threshold. Quiet frames are not VAD. Energy dynamic range is not calibrated SNR. Near-full-scale samples are clipping candidates, not proof of clipping. Nominal prompt WPM assumes the intended words were spoken; use only with transcript checks and do not interpret it as independently measured speaking rate.''')
code(r'''metrics=['duration_s','rms_dbfs','peak','quiet_fraction_45dbfs','energy_dynamic_range_db','near_fullscale_fraction','dc_offset','nominal_prompt_wpm']
numeric=df[metrics].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.reset_index(names='feature'); export(numeric,'numeric_summary'); display(numeric)
fig,axes=plt.subplots(2,4,figsize=(18,9))
for ax,field in zip(axes.flat,metrics): ax.hist(df[field].dropna(),bins=25,color='#4386a2'); ax.set(title=field,ylabel='Configurations'); ax.tick_params(axis='x',labelrotation=20)
savefig('acoustic_distributions')
fig,axes=plt.subplots(1,2,figsize=(17,8))
sns.boxplot(data=df,y='model',x='duration_s',ax=axes[0],color='#73b0b7'); axes[0].set_title('Fixed Hebrew sentence: duration by model')
sns.boxplot(data=df,y='model',x='rms_dbfs',ax=axes[1],color='#94b973'); axes[1].set_title('Analysis waveform RMS by model'); savefig('model_audio_comparison')
flag=pd.DataFrame({'sample_id':df.sample_id})
flag['technical_fail']=~df.technical_qc_pass
flag['hebrew_text_match_needs_review']=~df.language_qc_pass
flag['quiet_majority']=df.quiet_fraction_45dbfs>.5
flag['near_fullscale_candidate']=df.near_fullscale_fraction>.001
flag['low_rms']=df.rms_dbfs< -40
rules=[]
for field in ['duration_s','rms_dbfs','quiet_fraction_45dbfs','energy_dynamic_range_db']:
    x=df[field]; median=x.median(); mad=(x-median).abs().median(); q1,q3=x.quantile([.25,.75]); iqr=q3-q1
    if mad>0: flagged=(.6745*(x-median)/mad).abs()>3.5; method='modified_z_abs_gt_3.5'
    elif iqr>0: flagged=(x<q1-1.5*iqr)|(x>q3+1.5*iqr); method='IQR_1.5_fallback'
    else: flagged=pd.Series(False,index=x.index); method='degenerate_no_flag'
    flag['outlier_'+field]=flagged.fillna(False)
    rules.append({'feature':field,'median':median,'MAD':mad,'IQR':iqr,'method':method,'flagged':int(flagged.sum())})
export(pd.DataFrame(rules),'outlier_rules'); export(flag,'sample_flags')
df['review_flag_count']=flag.drop(columns='sample_id').sum(axis=1)
df['review_reasons']=flag.drop(columns='sample_id').apply(lambda row:';'.join(row.index[row.astype(bool)]),axis=1)
review_candidates=df[df.review_flag_count>0].sort_values('review_flag_count',ascending=False); export(review_candidates,'review_candidates')
display(review_candidates[['model','voice','review_reasons']].head(20))
''')
md('''## Hebrew content checks and human-speech reference
The normalized character edit distance compares an audio-model transcription to the intended sentence; it is not gold-standard ASR CER. Numeric/spelling alternatives can create differences even when speech is correct. Automatic passage requires Hebrew language identification, at least 80% Hebrew letters in the transcription, normalized edit distance <= 0.25, and model intelligibility >= 0.7. Every accepted clip still permits human correction.

The human reference has one short 6-10-word excerpt per eligible recorder, selected at 3-8 seconds, within the first 20 seconds of a source recording with alignment confidence >=0.8. It spans fewer than all 85 recorders because these are explicit eligibility rules. Both references and synthetic clips use 16 kHz mono analysis. Text, channel history and selection differ, so comparisons are descriptive and cannot isolate a synthesis effect.''')
code(r'''language=df.groupby('model').agg(configurations=('sample_id','size'),reviewed=('review_status',lambda s:s.eq('reviewed').sum()),automatic_hebrew_match=('language_qc_pass','sum'),median_transcript_distance=('normalized_character_error_rate','median')).reset_index()
export(language,'hebrew_validation'); display(language)
language_failed=df[~df.language_qc_pass]; export(language_failed,'hebrew_review_candidates')
fig,axes=plt.subplots(1,2,figsize=(14,5)); axes[0].hist(df.normalized_character_error_rate.dropna(),bins=25); axes[0].set(title='Audio-model transcript distance (not gold-standard CER)',xlabel='Normalized character edit distance',ylabel='Configurations'); df.language_qc_status.fillna('not_reviewed').value_counts().plot.bar(ax=axes[1]); axes[1].set_title('Automatic Hebrew content screening'); axes[1].tick_params(axis='x',labelrotation=20); savefig('hebrew_screening')
human=pd.read_csv(BASE/'human_reference.csv'); human=human[human.reference_status.eq('decoded')].copy()
comparison=pd.concat([human[['duration_s','rms_dbfs','quiet_fraction_45dbfs','energy_dynamic_range_db']].assign(source='Real utterance reference'),valid[['duration_s','rms_dbfs','quiet_fraction_45dbfs','energy_dynamic_range_db']].assign(source='Synthetic fixed utterance')])
comparison_summary=comparison.groupby('source').agg(['count','median','mean','std']).reset_index(); comparison_summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in comparison_summary.columns]; export(comparison_summary,'real_synthetic_summary'); display(comparison_summary)
fig,axes=plt.subplots(1,3,figsize=(16,5))
for ax,field in zip(axes,['duration_s','rms_dbfs','quiet_fraction_45dbfs']): sns.boxplot(data=comparison,x='source',y=field,ax=ax); ax.set_title(field); ax.set_xlabel(''); ax.tick_params(axis='x',labelrotation=15)
savefig('real_synthetic_reference')
''')
md('''## Cost, conclusions and limits
Per-request reported costs are preferred to key usage snapshots, which can lag or fluctuate. A missing cost is not zero. Provider labels are not verified demographics, and audio-model perceived age/presentation estimates have no validated accuracy for this pilot. The listening workbench saves human ratings separately. One short example per configuration does not estimate within-voice variation, recording-session variation, naturalness reliability, or demographic representativeness.''')
code(r"""billing=json.loads((BASE/'billing_summary.json').read_text(encoding='utf-8')) if (BASE/'billing_summary.json').exists() else {'billing_complete':False,'status':'pending'}
display(pd.Series(billing).to_frame('value'))
overview.update({'reference_speakers':int(human.user_id.nunique()),'review_candidates':len(review_candidates),'perceptual_presentation_conflicting_groups':int(voice_summary.perceived_presentation.eq('conflicting_across_models').sum()),'perceptual_age_conflicting_groups':int(voice_summary.perceived_age_band.eq('conflicting_across_models').sum()),'age_estimator_uninformative':age_estimator_uninformative,'billing':billing})
(ANALYSIS/'key_metrics.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
export(df,'sample_features')
baseline=df.groupby('voice_identity_group',group_keys=False).sample(n=1,random_state=20261116)
export(baseline[['sample_id','model','voice','voice_identity_group','wav_path']],'speaker_spanning_listening_sample')
perceived=distributions[(distributions.unit=='provider_voice_groups')&distributions.field.isin(['perceived_presentation','perceived_age_band'])]
report=f'''# Synthetic Hebrew voice pilot: findings and progress

## What was completed
{overview['decoded_samples']} of {overview['planned_samples']} planned configurations decoded successfully across {overview['models']} models, {overview['families']} provider families and {overview['provider_voice_groups']} nominal provider/voice groups. Generated audio totals {overview['minutes_generated']:.2f} minutes. Every request used the same Hebrew text: {planned.text.iloc[0]}

Original audio and derived analysis WAV have separate hashes. No reference-audio cloning or corpus-volume expansion occurred. The recorded stock-voice catalog, exact requests, generation IDs, billing records and failed review attempts are retained. Initial audio-review JSON-mode failures were diagnosed; after removing the incompatible parameter, explicit repair attempts were logged rather than silently replacing history.

## Coverage and Hebrew screening
{markdown_table(coverage)}

{overview['technical_qc_pass']} configurations passed technical screening and {overview['automatic_hebrew_match_pass']} passed automatic Hebrew/text screening. The combined passed count is {overview['combined_qc_pass']}. Language passage is an audio-model transcription/recognition check, not human verification; corrections remain possible in review.html. Failed or unverified clips remain in the manifest and are not counted as automatically accepted Hebrew samples.

## Perceptual voice groups — unvalidated estimates
{markdown_table(perceived)}

These are casting impressions from very short synthetic clips, not real ages, biological sex or verified gender identities. Below-0.6 self-confidence is mapped to uncertain. The model's confidence is not calibrated. {overview['perceptual_presentation_conflicting_groups']} nominal voice groups have conflicting presentation estimates across model versions. Do not hide those conflicts with majority voting. Source-provided voice labels are reported separately in provider_labels.png and voice_summary.csv.

Age estimator uninformative: **{age_estimator_uninformative}**. The audio model returned the same age category for all clips, including voices with different source age descriptions. This failed informativeness check prevents presenting those outputs as a usable age distribution. All raw estimates remain in raw_age_model_outputs.csv; perceived age requires human listening or a separately validated estimator.

## Acoustic patterns and outliers
The notebook includes numerical percentiles, model comparisons, quiet-frame fractions, waveform levels, format history, duplicate screening and robust outlier flags. There are {len(review_candidates)} configurations with one or more broad review flags. No recordings were deleted. Silence thresholds are not VAD, dynamic range is not SNR, and near-full-scale samples do not establish clipping. Repeated text is deliberate and lexical coverage is therefore fixed, unlike the real corpus.

## Real-speech reference
{len(human)} utterances from {human.user_id.nunique()} real recorder IDs were processed at the same analysis sample rate. Each was selected to be 3-8 seconds and 6-10 aligned words, with source alignment score >=0.8. They are not the same sentence and have different recording/channel histories. Comparisons must not be presented as proof of a synthesis effect or used to claim classifier accuracy. This subset is not a replacement for the full 85-recorder dataset.

## Costs
{json.dumps(billing,ensure_ascii=False,indent=2)}

Missing or incomplete billing is explicitly marked and must not be interpreted as free generation. API usage snapshots can be eventually consistent. The project limit remains $20; this pilot does not authorize spending the remaining balance on a 20-hour dataset.

## Next steps
Listen to review candidates and the seeded one-example-per-voice baseline. Rate perceived presentation, casting-age category, Hebrew correctness and intelligibility; export human ratings from the workbench. Double-rate a subset before treating perceptual labels as reliable. Choose text and voice/engine split rules before any training experiment, and retain the repeated-text pilot as a clearly tagged calibration/EDA subset. The four provider families are not proof of architectural independence, and models sharing voice identifiers are dependent.
'''
(ROOT/'SYNTHETIC_EDA_PROGRESS.md').write_text(report,encoding='utf-8')
provenance={'executed_utc':pd.Timestamp.now(tz='UTC').isoformat(),'python':sys.version,'versions':{name:importlib.metadata.version(name) for name in ['openai','numpy','pandas','av','nbformat','nbclient','matplotlib','seaborn']},'seed':20261116,'metrics':overview,'input_sha256':{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ['planned_samples.csv','manifest.csv','automated_review.json','catalog_snapshot.json']}}
(ANALYSIS/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
display(Markdown(report))
assert len(df)==len(planned) and set(generated.sample_id)==set(planned.sample_id)
assert len(voice_summary)==planned.voice_identity_group.nunique()
print('Exports:',len(list(ANALYSIS.glob('*.csv'))),'CSV tables;',len(list(FIG.glob('*.png'))),'figures')
""")

notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
nb.write(notebook,Path(__file__).resolve().parent/'synthetic_eda.ipynb')
print('Wrote synthetic_eda.ipynb with',len(cells),'cells')
