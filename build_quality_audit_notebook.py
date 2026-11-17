"""Build and optionally execute the corpus quality / duplicate audit."""
from pathlib import Path
import sys
import nbformat as nb

ROOT=Path(__file__).resolve().parent
cells=[]
def md(s): cells.append(nb.v4.new_markdown_cell(s))
def code(s): cells.append(nb.v4.new_code_cell(s))
md('''# Hebrew corpus quality, duplicates and representation audit

Companion to `real_synthetic_comparison.ipynb`. This audit examines full-corpus metadata and text, then measures how selected reference/proposal subsets differ. **No subset is claimed to represent all Hebrew speech.** Full corpus hours, eligible segment durations and equal-speaker coverage answer different questions.

See `QUALITY_AND_MATCHING_SPEC_HE.md` for concrete duplicate examples, priorities and the proposed dataset contract. This notebook is offline: no new audio, API calls, training, deletions or human annotations.''')
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
import quality_matching_audit as audit
ROOT=Path.cwd(); OUT=ROOT/'quality_audit'
metrics=audit.main()
display(pd.Series(metrics,name='value').to_frame())
def table(name): return pd.read_csv(OUT/(name+'.csv'))
''')
md('''## 1. Exact audio versus repeated transcript

Real whole-corpus canonical PCM fingerprints are inherited from the frozen previous full decode; they were not recomputed from all original recordings here. Synthetic original/WAV file fingerprints are separately reconciled. Additionally **all 362 saved real-reference/synthetic WAVs are reopened**, checked for 16 kHz mono PCM16, and hashed over sample payloads, allowing a common cross-class exact PCM check.

Exact hashes do not detect shifted, level-scaled, differently coded or trimmed copies. A full perceptual near-audio duplicate search remains future work. Repeated text is not evidence of identical audio. Text counts use one documented punctuation/diacritic normalization and include short/invalid rows in the broad audit.''')
code('''display(table('exact_audio_fingerprint_audit'))
display(table('saved_audio_exact_pcm_duplicates'))
display(table('real_session_exact_groups').drop(columns='audit_normalized_text').head(12))
groups=table('eligible_segment_exact_groups')
display(groups.head(12))
occurrences=table('eligible_segment_exact_occurrences')
display(occurrences.loc[occurrences.text_group_id.isin(groups.head(3).text_group_id)].head(20))
''')
md('''## 2. Near-text candidates and leakage

All session pairs with at least 20 source whitespace words are screened using word unigram/bigram TF-IDF with sublinear term frequency. Cosine >=0.92 and normalized text not exactly equal yields 12 candidate pairs. This is lexical similarity, not semantic/phonetic matching or confirmed duplicates. The threshold can miss partial overlap and shorter shared passages.

Same-text utterances from different speakers remain useful, but require common text-group assignment in a text-disjoint benchmark. Session/speaker and source-derived variants must stay in one split. The existing proposal is exact-text-unique, but near-text and document leakage must still be checked after splitting. No train/test split is created here.''')
code('''near=table('near_session_text_candidates')
display(near.drop(columns=['text_a','text_b']).head(12))
display(near[['session_a','session_b','text_a','text_b']].head(2))
''')
md('''## 3. Quality review queues and identity coverage

Flags overlap and must not be summed as failed-file counts. Low source alignment score is not SNR. Absolute quiet fractions depend on level. Sample near-fullscale statistics do not prove absence of past clipping. Hebrew-letter ratio is not an intelligibility score.

The real listening manifest accounts for every one of 85 source IDs, with a source interval where available and an explicit manual-review status otherwise. The synthetic manifest covers 116 nominal provider:voice groups, rotating versions within a family. It does not validate every version or estimate natural population prevalence. Previously collected human labels are not propagated.''')
code('''display(table('real_priority_review_summary'))
display(table('synthetic_priority_review'))
real_listen=table('real_identity_coverage_listening')
synth_listen=table('synthetic_identity_coverage_listening')
display(real_listen.status.value_counts().rename('speaker_IDs').to_frame())
display(synth_listen.groupby(['family','model']).size().rename('nominal_voices').to_frame())
display(table('real_speaker_exposure').head(12))
''')
md('''## 4. Is the reference representative?

Compare source sex/age descriptors on **the same real source**, using both speaker shares and audio-time shares. Unknown labels stay visible. Full-corpus shares are a descriptive baseline, not population truth. The 308 proposal's equal-speaker objective changes its time composition, particularly unknown labels. This is not automatically a flaw; target definition determines whether to reselect or weight.

Total variation is half the sum of absolute category-share differences. It describes total selection shift, not a per-category error or a fairness metric. Source birth-year age relative to 2026 is not age at recording; do not match it to rejected synthetic age estimates.''')
code('''display(table('real_demographic_coverage_by_scope'))
display(table('reference_selection_shift'))
for name in ['source_sex_representation','source_age_representation']:
    display(Image(filename=str(OUT/'figures'/(name+'.png'))))
''')
md('''## 5. Recommended next experiment and verification

Use the current single-sentence synthetic pilot for calibration. Validate proposed real intervals before new generation; assign stock voices within fixed partitions and generate the same text. Balance identity/family exposure, monitor minutes and duration bins, use symmetric boundary/channel processing, and keep controlled and deployment-oriented evaluations separate. Do not force every acoustic feature to match or use detector errors to curate evaluation data.

Concrete priorities and acceptance rules are in `QUALITY_AND_MATCHING_SPEC_HE.md`. Review all seven synthetic outlier candidates and identity coverage before expansion; no human approval is implied by successful automated QC.

Guidance: [ASVspoof 2021](https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf) and [Smeu et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.pdf). Publication results are not detector results on this corpus.

Reproduce: `python build_quality_audit_notebook.py --execute`, then `python verify_quality_audit.py`.''')
code('''assert len(real_listen)==85 and real_listen.user_id.is_unique
assert len(synth_listen)==116 and synth_listen.voice_identity_group.is_unique
assert metrics['saved_362_audio_pcm_duplicate_rows']==0
assert metrics['eligible_sixplus_word_cross_speaker_groups']==145
display(json.loads((OUT/'provenance.json').read_text()))
print('Completed offline quality/duplication audit; no source changes or API calls.')
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
for cell in cells:
    if cell.cell_type=='code': compile(cell.source,'<quality audit>','exec')
path=ROOT/'quality_matching_audit.ipynb'; nb.write(notebook,path)
if '--execute' in sys.argv:
    from nbclient import NotebookClient
    def progress(cell,cell_index,**kwargs):
        print('Quality audit cell',cell_index+1,'of',len(cells),flush=True)
    try: NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress).execute()
    finally: nb.write(notebook,path)
    assert not [o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
    print('Executed quality audit with zero errors')
