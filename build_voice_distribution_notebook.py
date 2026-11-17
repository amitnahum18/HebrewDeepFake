"""Build / execute offline multi-method voice distribution EDA."""
from pathlib import Path
import sys
import nbformat as nb
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Hebrew real / synthetic voice distributions — multiple views

This notebook analyzes **actual ECAPA embeddings**, acoustic features, nominal identity consistency and coverage. No inference download or paid API request occurs in notebook execution; cached embeddings are required. The local CPU extraction is in `ecapa_embeddings.py`. See `VOICE_DISTRIBUTION_PLAN.md` for prespecified analyses and `VOICE_DISTRIBUTIONS_REPORT_HE.md` for Hebrew results.

Scope: up to three 3–8 s excerpts per qualified real speaker, preferring distinct sessions/texts, versus all stock synthetic clips. Selection targets identity coverage, not corpus-hour or population representation. Five real IDs have no eligible excerpt under these criteria; keep them in the audit. There are no human-verified new labels. Synthetic samples read one text while real samples vary; text/phonetics, native codec and boundaries remain confounders.''')
code('''from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image
import analyze_voice_distributions as analysis
ROOT=Path.cwd(); OUT=ROOT/'voice_distribution'
metrics=analysis.main()
def table(name): return pd.read_csv(OUT/(name+'.csv'))
def figure(name): display(Image(filename=str(OUT/'figures'/(name+'.png'))))
samples=table('samples')
display(pd.Series(metrics,name='value').to_frame())
display(samples.groupby('label').agg(clips=('sample_id','size'),identities=('group_id','nunique'),seconds=('duration_s','sum')))
''')
md('''## 1. Selection, extraction and technical provenance

Real selection requires source/alignment scores >=0.8, 3–8 s, 6–14 words and positive in-bounds timing. Different sessions and normalized texts are preferred, then duration near 4.5 s. This does not validate speaker identity or transcript by listening.

Initial timestamp-addressed slicing exposed jittered MKA packet clocks. Inserting zeros at every packet gap would create artificial silence. The final extraction therefore uses concatenated decoded sample counts, consistent with prior EDA. Initial attempt records remain separate and must not be interpreted as source audio failures. No attempt was made to repair actual clock discontinuities; listening/boundary validation remains pending.

Official SpeechBrain Fbank (80 mel bins), sentence mean normalization and pretrained ECAPA layers output 192 dimensions. We use strict weight loading, no speaker classifier or gender/age model, and L2-normalize each vector. Global speaker-verification thresholds and VoxCeleb benchmark scores are not transferred to this Hebrew dataset.''')
code('''coverage=table('real_speaker_coverage')
display(coverage.describe())
display(coverage.loc[coverage.extracted.eq(0)])
display(json.loads((OUT/'model_provenance.json').read_text()))
display(json.loads((OUT/'inference_environment.json').read_text()))
display(samples.loc[samples.label.eq('real')].groupby('sex_label').agg(speakers=('group_id','nunique'),clips=('sample_id','size')))
display(samples.loc[samples.label.eq('real')].groupby('age_proxy_bin').agg(speakers=('group_id','nunique'),clips=('sample_id','size')))
''')
md('''## 2. Hebrew sanity check: same real identity versus different identity

Compare cosine distributions among the sampled real clips. Separately report same-speaker pairs from different sessions, because shared microphones/session conditions can inflate same-session agreement. Different-speaker negative pairs include the observed recording-condition mix. Pairs are dependent; AUC is an exploratory same-ID ranking statistic on this sample, **not deepfake detector performance or a validated identity threshold**.

Some real IDs have only one eligible clip. Within-identity claims apply only where repeated clips exist. Strong separation supports local utility of the representation; it does not establish unbiased Hebrew speaker verification.''')
code('''summary=table('cosine_distribution_summary')
display(summary.loc[summary.label.eq('real')])
figure('identity_consistency_original')
figure('identity_consistency_edge_trimmed')
display(pd.Series({k:v for k,v in metrics.items() if 'auc' in k or 'cross_session' in k}))
''')
md('''## 3. Synthetic consistency, diversity and listening candidates

A nominal provider:voice may occur in two to four model versions. Compare these pairs with different-voice pairs, and inspect both low-consistency same-name groups and high-similarity different-name groups. These are **listening candidates, not duplicate or identity proof**. Content is the same for every synthetic clip, which may itself elevate similarities. Cross-provider voices can also share origins unknown to us.

Centroids are L2-normalized means of L2 vectors. Each nominal identity gets one centroid regardless of its number of versions. This is a conservative grouping, not a biometric identity declaration.''')
code('''display(summary.loc[summary.label.eq('synthetic')])
within=table('within_identity_pairs')
display(within.loc[within.label.eq('synthetic')].sort_values('cosine_similarity').head(16))
display(table('closest_synthetic_groups_original').head(15))
display(table('closest_real_groups_original').head(10))
''')
md('''## 4. Shared PCA and t-SNE maps

Both groups are projected together using one centroid per identity, coloured by real/family. PCA reports retained variance; t-SNE uses a fixed seed and perplexity 30. Plots may reveal local clusters/provider effects, but apparent overlap/separation is not a hypothesis test. All reported cosine/kernel metrics use the full original 192-dimensional space.''')
code('''figure('speaker_projections')
print('Fraction of variance retained by 2-D PCA:',metrics['pca_2d_variance_fraction'])
display(table('projection_coordinates').head())
''')
md('''## 5. Coverage: nearest acoustic neighbours

Compare real-to-synthetic and synthetic-to-real nearest centroid distances, with a within-real reference. Candidate sets differ in size, so nearest-distance differences alone cannot establish better coverage. Nearest synthetic identity for a real speaker is an acoustic neighbour, not a cloned identity or prescribed matching assignment.

Do not select only nearest neighbours to create the main dataset: that changes the target to similar-sounding voices and may discard legitimate diversity. Use these candidates to find uncovered regions and for a separate controlled sensitivity subset.''')
code('''neighbours=table('cross_class_nearest_neighbours')
display(neighbours.groupby(['view','direction']).cosine_distance.agg(['count','median','min','max']))
display(neighbours.loc[(neighbours.view=='original')&(neighbours.direction=='real_to_other_class')].sort_values('cosine_distance',ascending=False).head(12))
figure('nearest_coverage_original')
figure('nearest_coverage_edge_trimmed')
''')
md('''## 6. Kernel distribution distance and family weighting

Biased weighted RBF MMD squared compares identity-centroid distributions at 0.5x, 1x and 2x the pooled median squared-distance bandwidth scale. Lower means closer under that representation/kernel; it is **not a percentage of realism or a universal quality score**. Finite sample sizes and domain effects matter. Report both equal-identity and equal-family weights.

For family comparisons, sample the same number of real and synthetic identities per family comparison (up to 20 each), repeat 200 times and report the observed composition range. Those ranges are not confidence intervals for a population, significance tests or causal synthesis effects.''')
code('''display(table('kernel_distribution_distances'))
display(table('equal_size_family_sensitivity'))
display(table('equal_size_family_coverage'))
display(table('one_clip_per_identity_sensitivity').groupby('view').agg({'mmd_squared_one_clip_per_identity':['median','min','max'],'median_between_real_cosine':'median','median_between_synthetic_cosine':'median'}))
''')
md('''## 7. Independent acoustic-feature view and practical limits

Recompute identical duration, level, absolute-energy quiet, edge and spectral features for all sampled audio. Use per-identity medians and pooled median/IQR scaling for a compact multivariate comparison. Scaling is for visualization, not a finalized preprocessing policy. These features are not SNR, human quality judgments, or demographic labels.

Interpret alongside prior native-unit effect tables and the corpus-level sex/age/time coverage audits. ECAPA does not replace those. If distribution differences persist after capped trim, that does not identify a synthesis artifact: text, recording conditions, language domain and sampling can still explain them.''')
code('''acoustic=table('identity_acoustic_features')
display(acoustic.groupby('label').median(numeric_only=True))
display(table('selected_acoustic_effect_sizes'))
display(table('embedding_boundary_sensitivity').groupby('label').original_trimmed_cosine.agg(['count','median','min']))
figure('acoustic_multivariate_view')
display(json.loads((OUT/'analysis_provenance.json').read_text()))
assert len(samples)==515 and samples.sample_id.is_unique
assert metrics['real_speakers']==80 and metrics['synthetic_nominal_voices']==116
print('Completed cached offline analysis. No TTS generation, API spend, source changes or new human ratings.')
''')
md('''Primary sources: [ECAPA-TDNN paper](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf), [official SpeechBrain checkpoint/model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), [model architecture configuration](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/hyperparams.yaml).

Reproduce extraction/inference using the dedicated environment as described in `README_VOICE_DISTRIBUTIONS.md`. Re-execute analysis: `python build_voice_distribution_notebook.py --execute`, then `python verify_voice_distributions.py`.''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
for cell in cells:
    if cell.cell_type=='code':compile(cell.source,'<voice distributions>','exec')
path=ROOT/'voice_distributions.ipynb'; nb.write(notebook,path)
if '--execute' in sys.argv:
    from nbclient import NotebookClient
    def progress(cell,cell_index,**kwargs): print('Voice distribution cell',cell_index+1,'of',len(cells),flush=True)
    try:NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress).execute()
    finally:nb.write(notebook,path)
    assert not [o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
    print('Executed voice distribution notebook with zero errors')
