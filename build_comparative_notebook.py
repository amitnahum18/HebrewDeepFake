"""Build an executable comparative EDA with explicit interpretation and limits."""
from pathlib import Path
import nbformat as nb

root=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))

md('''# Hebrew real / synthetic audio: comparative EDA

This extends `dataset.ipynb` and `synthetic_eda.ipynb`. Read `COMPARATIVE_EDA_PLAN.md` for the prespecified units, exclusions and matching rules, and `DATASET_MATCHING_RECOMMENDATIONS_HE.md` for the Hebrew interpretation and proposed experiment.

**Decision:** the current synthetic pilot is a voice/configuration inventory and calibration set. It cannot support a credible text-disjoint real/fake benchmark: all 284 synthetic clips read one sentence, absent from the real aligned utterances. Source codec and clip-boundary preparation also differ by label. No paid calls, model training, source deletion, or audio generation occur here.''')
code('''from pathlib import Path
import json, sys
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
import comparative_analysis as analysis
OUT=ROOT/'comparison_outputs'
metrics=analysis.run()
display(pd.Series(metrics,name='value').to_frame())
def table(name): return pd.read_csv(OUT/(name+'.csv'))
def figure(name): display(Image(filename=str(OUT/'figures'/(name+'.png'))))
''')
md('''## 1. Eligibility and coverage of real aligned utterances

The funnel is sequential: a row removed earlier is not counted again later. Sum of aligned segment durations is neither original recorded hours nor guaranteed nonoverlapping speech time. Source session quality and segment alignment scores are dataset-provided heuristics, **not calibrated audio SNR or human intelligibility scores**. Thresholds are operational sensitivity choices, not validated universal cutoffs. Missing ages do not exclude audio. The full real corpus remains frozen.''')
code('''display(table('real_eligibility_funnel'))
display(table('eligibility_sensitivity'))
coverage=table('proposed_speaker_coverage')
display(coverage.describe(include='all'))
display(coverage.loc[coverage.proposed_pairs.eq(0)])
''')
md('''## 2. Voice exposure and demographic information

Model/voice configurations are repeated measurements, not independent speakers. Provider:voice IDs are conservative nominal identity groups, not biometric identity guarantees. Stock voice gender/age descriptors, real source biological-sex/birth-year fields and auditory impressions represent different concepts and must remain separate.

The automated age reviewer returned one adult category for every synthetic sample; this is rejected as uninformative. Thirty-five nominal voice groups also have conflicting presentation estimates across model versions. Neither result supports assigning synthetic biological ages/sexes. Human listening is still pending for all synthetic samples; the 14 previous real excerpt judgments remain excerpt-scoped and are not propagated.''')
code('''display(table('synthetic_family_exposure'))
display(table('label_distributions'))
display(table('synthetic_review_queue'))
display(table('reference_metadata_coverage'))
''')
md('''## 3. Text and codec confounds

Zero exact normalized real-segment matches were found for the one synthetic sentence. Native real codecs are Opus/AAC; synthetic native outputs are MP3/PCM. Conversion to the same WAV format does not undo lossy codec history. Original source metadata must not be given to a detector.

The acoustic comparison below uses all 284 synthetic WAVs and an existing, deliberately selected reference of 78 real speakers, one short excerpt each. This is a descriptive reference, **not a random representative sample** of the 50-hour real corpus. Real boundaries follow alignment, synthetic boundaries retain full generated output; edge differences are therefore inseparable from extraction method here. Transcript content and phonetic composition remain uncontrolled.''')
code('''display(table('native_codec_confound'))
assert metrics['exact_real_segment_text_matches']==0
assert metrics['synthetic_unique_texts']==1
common=table('common_audio_features')
display(common.groupby('label')[['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction']].agg(['count','median','min','max']))
''')
md('''## 4. Common acoustic measurements and group-weighted effects

Identical code reads saved mono 16 kHz PCM16 WAVs. Spectral summaries use 20 ms Hann-window frames above -45 dBFS. Quiet masks are absolute-energy proxies, not speech activity labels or noise estimates. The energy percentile range is **not SNR**, and digital zeros can inflate it.

Each synthetic provider:voice group contributes its median over versions, preventing four versions of one voice from being counted as four independent people. Effect signs are synthetic minus real. Cliff's delta and Wasserstein distances are descriptive. Bootstrap intervals resample observed real speaker/nominal voice groups (1,000 repetitions, fixed seed); correlated voices within providers and selected real references limit interpretation. No hypothesis tests or population claims are made.

`in_sample_orientation_free_auc` measures one-variable separability on the same observed group medians. It is a shortcut warning, **not detector performance, validation AUC, EER, or generalization**.''')
code('''effects=table('group_effect_sizes')
display(effects.loc[effects.view.eq('original')].round(4))
figure('acoustic_distributions')
figure('spectral_distributions')
''')
md('''## 5. Sensitivity to preparation choices

Two separate in-memory views: remove at most 0.25 s of estimated quiet frames per edge; or scale RMS toward -23 dBFS with peak capped at 0.99. No transformed audio is saved over originals. These are sensitivity analyses, **not a finalized preprocessing policy**. Normalization alone leaves the edge shortcut; capped trimming reduces it but does not establish a fair benchmark. Frame threshold effects and crop phase can change spectral summaries.''')
code('''display(effects.pivot(index='feature',columns='view',values='smd').round(3))
figure('effect_sensitivities')
''')
md('''## 6. Duration / word-count matching diagnostic

Eligibility: exactly eight real aligned words; duration difference <=0.5 s; each real speaker and nominal synthetic voice used at most once. Four synthetic families receive equal counts. Mixed-integer optimization maximizes feasible equal-family cardinality, then minimizes total absolute duration difference. It certifies **44 pairs, 11 per family** under this selected reference and caliper; this is not the maximum matching capacity of the whole real corpus.

Texts still differ. This diagnostic cannot be used as same-text training/evaluation data. Duration effects shrink, but edge/level differences persist and some effects grow. Matching covariates does not prove removal of confounds. Bootstrap intervals in this table are the same separate-group composition diagnostic, not paired treatment-effect inference.''')
code('''pairs=table('diagnostic_duration_word_pairs')
display(pairs.groupby('family').agg(pairs=('real_sample_id','size'),mean_duration_gap=('duration_difference_s','mean'),max_duration_gap=('duration_difference_s','max')))
display(effects.loc[effects.view.eq('duration_word_matched')].round(4))
figure('diagnostic_pairs')
assert pairs.real_group_id.is_unique and pairs.synthetic_group_id.is_unique
assert pairs.duration_difference_s.le(.5+1e-9).all()
assert pairs.groupby('family').size().nunique()==1
''')
md('''## 7. Proposed true same-text experiment — not generated

Select up to four globally normalized-text-unique real utterances per eligible speaker, round-robin. Require positive in-bounds timing, session and alignment scores >=0.8, 2–10 s, 6–14 words, and >=80% Hebrew letters among letters. This yields **308 proposed real utterances from 81 speakers**, assigned provisionally 77 per synthetic family. Around 17.10 minutes of real excerpts; synthetic duration must be measured after generation and is not assumed equal.

Proposed source cuts still need listening, boundary and transcript validation. These thresholds retain 81/85 source IDs; four remain in the coverage audit, not deleted. Model/voice and split columns intentionally remain unassigned. Each target should later have one authentic excerpt and a synthetic rendering of that exact Hebrew text. Balance family and voice exposure within frozen partitions; reserve an independent evaluation pool first. Different versions sharing a voice must stay in the same identity partition.

The one-sentence pilot belongs in calibration only. It must not be randomly spread across train/dev/test. For text-disjoint evaluation of the full real corpus, repeated document passages can connect many speakers: retain one partition's copies and explicitly discard conflicting copies, or build a curated text-unique subset and audit normalized text, near-duplicates, recording and speaker groups after splitting. Never promise full simultaneous speaker/text disjointness while silently retaining leakage.''')
code('''display(table('proposed_family_allocation'))
proposal=table('proposed_same_text_pairs')
display(proposal[['proposed_pair_id','user_id','duration_s','word_count','desired_family','text']].head(12))
assert proposal.exact_text_group.is_unique
assert proposal.groupby('user_id').size().le(4).all()
assert proposal.quality_score.ge(.8).all() and proposal.segment_alignment_score.ge(.8).all()
assert proposal.duration_s.between(2,10).all() and proposal.word_count.between(6,14).all()
assert proposal.hebrew_letter_fraction.ge(.8).all()
assert proposal['split'].isna().all() and proposal.assigned_model.isna().all()
''')
md('''## 8. Reproducibility and next decision

Read the detailed Hebrew report for handling policy, matching quotas, demographic limitations, split protocols and priority order. No additional synthesis or budget expenditure was performed here.

Primary guidance: [ASVspoof 2021 evaluation plan](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf) motivates auditing coding/transmission conditions; [Smeu et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.pdf) documents leading-silence shortcuts; [ivrit-ai dataset card](https://huggingface.co/datasets/ivrit-ai/crowd-recital) supplies schema context. These publications motivate checks; their benchmark results are not evidence about this pilot.

Run: `python build_comparative_notebook.py`, `python execute_comparative_eda.py`, `python verify_comparative_eda.py`. The notebook regenerates offline tables/plots. Source audio hashes are checked against the frozen preceding EDA manifests.''')
code('''display(json.loads((OUT/'provenance.json').read_text()))
assert len(common)==362
assert common.sample_id.is_unique
assert common.groupby('label').group_id.nunique().to_dict()=={'real':78,'synthetic':116}
print('Completed offline comparison: no API requests, no source audio mutations, no new human labels.')
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
for cell in cells:
    if cell.cell_type=='code': compile(cell.source,'<comparison notebook>','exec')
nb.write(notebook,root/'real_synthetic_comparison.ipynb')
print('Created comparison notebook:',len(cells),'cells')
