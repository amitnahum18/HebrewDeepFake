"""Create and execute the detailed cached-results clustering notebook."""
from pathlib import Path
import sys
import nbformat as nb
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Clustering Hebrew speech: acoustic features, ECAPA and hybrid

This is exploratory unsupervised clustering, **not a real/fake detector**. Same selected 515 clips and 196 nominal identity representatives as `voice_distributions.ipynb`. Primary units: 80 real source IDs and 116 provider:voice groups, giving each identity one observation regardless of how many clips it has.

Four spaces: nine core acoustic features; all 16 acoustic descriptors as a redundancy sensitivity; 192-D ECAPA; and acoustic core + ECAPA with equal expected squared block norms. Original and capped-edge-trimmed views are compared. Read `CLUSTERING_PLAN.md` and `CLUSTERING_REPORT_HE.md` for methods, limitations and conclusions.

Class/provider/sex/age/text/ID fields are excluded from fitting, scaling, cluster-number selection and stability checks. Real/synthetic and family labels are added **after** fitting. Source demographic metadata cannot be treated as equivalent to synthetic casting/estimated labels.''')
code('''from pathlib import Path
import json, subprocess, sys
import pandas as pd
import numpy as np
from IPython.display import display, Image
ROOT=Path.cwd(); OUT=ROOT/'clustering_outputs'
# Set True to rerun all fits. Cached results were computed by this script;
# notebook reruns never regenerate speech or invoke paid APIs.
RECOMPUTE=False
if RECOMPUTE or not (OUT/'key_metrics.json').exists():
    subprocess.run([sys.executable,str(ROOT/'cluster_voice_features.py')],check=True)
def table(name):return pd.read_csv(OUT/(name+'.csv'))
def figure(name):display(Image(filename=str(OUT/'figures'/(name+'.png'))))
metrics=json.loads((OUT/'key_metrics.json').read_text())
selected=table('selected_models')
assign=table('identity_assignments')
display(selected)
display(assign.loc[(assign.view=='original')&(assign.space=='acoustic_core')].groupby('label').agg(identities=('group_id','size'),clips=('clips','sum')))
''')
md('''## 1. Feature definitions and preprocessing

Core: duration, RMS level, quiet fraction at -45 dBFS, quiet-edge fraction, percentile energy range, spectral centroid, highband fraction, flatness and zero crossings. All-descriptor sensitivity additionally includes peak, DC, near-fullscale fraction, other quiet thresholds and leading/trailing quiet seconds. The extra features overlap conceptually and are not independent new evidence.

Group medians; prespecified log transforms for skewed nonnegative spectral/range features; median imputation if needed; drop constant dimensions; RobustScaler median/IQR; cap transformed acoustic values at +/-5. ECAPA stays unit-normalized, without per-coordinate scaling. Hybrid concatenates the acoustic block scaled to expected squared norm 1 and the unit ECAPA block, then divides by sqrt(2). This balances raw block energy, not every coordinate or all centered variance.

All transforms are fitted on the pooled exploratory sample without class labels. They are not a fitted training/test preprocessing pipeline. Each boundary view is separately fitted, so boundary sensitivity includes preprocessing changes. No clipping or source-waveform change is performed by this acoustic feature cap.''')
code('''transforms=table('acoustic_transform_parameters')
display(transforms)
display(transforms.groupby(['view','space']).agg(retained_dimensions=('retained','sum'),clipped_cells=('total_clipped_cells','first'),missing_cells=('total_missing_cells','first')))
display(table('acoustic_core_pca_axis_coefficients'))
display(table('acoustic_all16_pca_axis_coefficients'))
''')
md('''## 2. Cluster-number selection and PCA

KMeans k=2..10, n_init=30, fixed seed. Select maximum silhouette among partitions with minimum cluster size >=8 and largest <=90% of identities. If none meet support requirements, flag a fallback, not a natural number of clusters. This grid searches only 2..10; it does not establish that any clusters exist or rule out other k.

Silhouette measures internal compactness/separation under the specified feature geometry. It is not accuracy. Scores across different spaces do not form a quality leaderboard. Primary fits use full transformed feature spaces; PCA and t-SNE are displays only. A separate >=90% PCA sensitivity checks the clustering effect of reduction. PCA coefficient signs are arbitrary and refer to transformed/scaled features.''')
code('''display(table('k_selection'))
figure('k_selection_overview')
figure('pca_variance_overview')
display(table('preprocessing_sensitivity'))
''')
md('''## 3. Acoustic core: clusters without ECAPA

Same 196 identities, nine numeric descriptors. Maps show the same coordinates coloured by fitted cluster, then real/synthetic and source family. Silhouette plots retain negative values. The acoustic profile heatmap is post-hoc and in pooled native-feature robust units, with colours capped at +/-3 for display; the full native-unit table is also saved. A profile describes conditions/features, not a cause or human quality category.

The average-linkage dendrogram uses a different clustering method. Its branch labels are aggregated groups, not original clip positions; parentheses count identities. Do not read branch heights as an identity-verification threshold.''')
code('''for name in ['acoustic_core_original_maps','acoustic_core_cluster_diagnostics','acoustic_core_acoustic_cluster_profiles','acoustic_core_hierarchy']:
    figure(name)
display(table('cluster_native_feature_profiles').query("view == 'original' and space == 'acoustic_core'").pivot(index='cluster',columns='feature',values='identity_median'))
''')
md('''## 4. ECAPA: voice-representation clusters

KMeans uses Euclidean distances on unit-normalized 192-D identity centroids. This is ordinary KMeans on unit vectors, not a separately implemented spherical KMeans. At point-to-point level, Euclidean and cosine distance have a fixed relationship, but unconstrained cluster means are not unit vectors.

The selected original k=2 partition has low silhouette (~0.057). It nearly tracks real/synthetic origin in this selected sample, despite labels not being used to fit it. Text, codec, channel and the single synthetic prompt remain confounders. That association is neither two verified natural voice categories nor independent deepfake detection performance.''')
code('''for name in ['ecapa_original_maps','ecapa_cluster_diagnostics','ecapa_acoustic_cluster_profiles','ecapa_hierarchy']:
    figure(name)
display(pd.crosstab(assign.query("view == 'original' and space == 'ecapa'").cluster,assign.query("view == 'original' and space == 'ecapa'").family))
''')
md('''## 5. Hybrid: created acoustic features plus ECAPA

This directly compares clustering with the existing acoustic core plus the learned voice representation. Block scaling was fixed before clustering. Appending 192 unscaled coordinates to nine acoustic values without a defined weighting would make the comparison harder to interpret. A different scientifically justified weight could still produce another partition.

Hybrid clusters are a feature-combination sensitivity, not proof that more features improve quality. Observe the low internal separation and changes across boundaries, algorithms and subsamples.''')
code('''for name in ['hybrid_original_maps','hybrid_cluster_diagnostics','hybrid_acoustic_cluster_profiles','hybrid_hierarchy']:
    figure(name)
''')
md('''## 6. All 16 created descriptors: redundancy sensitivity

Several quiet and edge descriptors describe related aspects of the waveform. Their inclusion changes how much weight those conditions get, even after per-feature scaling. Compare the all-descriptor result with the core result rather than claiming it is superior because its silhouette is larger.''')
code('''for name in ['acoustic_all16_original_maps','acoustic_all16_cluster_diagnostics','acoustic_all16_acoustic_cluster_profiles','acoustic_all16_hierarchy']:
    figure(name)
''')
md('''## 7. Alternative methods, stability and boundary sensitivity

Average-linkage is evaluated at the KMeans-selected k; it can isolate small branches. HDBSCAN uses fixed min_cluster_size=8 and min_samples=5, may leave points unassigned, and is not tuned to reveal class labels. Its noise label means **no density-cluster assignment**, not noisy/invalid audio. Silhouette on assigned density points must be read with coverage; high score on a small retained subset is not a fair full-sample comparison.

ARI measures agreement between partitions despite arbitrary numeric cluster labels: 1 is identical; near 0 means little agreement relative to chance, and negative is possible. Ten initializations and 20 random 80% identity subsamples quantify stability conditional on the fixed transform. They are not population confidence intervals.

Boundary comparisons report both independently selected k and the same original k. Acoustic changes include refitting the scaler for the trimmed view. Clip-level sensitivity fits on 515 clips at the same k and reports within-ID consistency; these clips remain dependent and do not replace the identity-balanced primary analysis.''')
code('''display(table('method_comparison'))
method=table('all_method_assignments')
sizes=method.loc[method.cluster.ge(0)].groupby(['view','space','method','cluster']).size().rename('identities').reset_index()
display(sizes.groupby(['view','space','method']).identities.agg(['min','max','sum']))
display(table('stability').groupby(['view','space','kind']).ari.agg(['median','min','max']))
figure('stability_overview')
figure('partition_agreement_and_boundaries')
display(table('boundary_sensitivity'))
display(table('within_identity_clip_consistency').query('clips > 1').groupby(['space','label']).agg(identities=('group_id','size'),median_majority=('majority_fraction','median'),split_identities=('distinct_clusters',lambda s:int(s.gt(1).sum()))))
''')
md('''## 8. Inspect clusters and decide next steps

Cluster numbers are arbitrary local identifiers: cluster 0 in one space is not cluster 0 in another. Representative/distant members are based on distance to the full-space KMeans mean; neither is automatically good/bad audio. Linked real clips are unverified extracted intervals, and synthetic clips remain without completed human verification. Listen before semantic labels or exclusions.

Use clusters to audit source conditions and sample coverage. Do not remove groups until real/fake overlap looks better, assign biological sex/age from clusters, or split train/test by cluster alone. Freeze speaker/voice/text partitions, build same-Hebrew-text pairs with symmetric channel/boundary preparation, then repeat these diagnostics.

Source documentation: [clustering overview](https://scikit-learn.org/stable/modules/clustering.html), [silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html). Runtime versions and parameters are saved in provenance. No detector or demographic estimator is trained.''')
code('''display(table('cluster_representatives').query("view == 'original'").head(24))
display(table('cluster_representative_audio_links').query("view == 'original'").head(12))
display(table('posthoc_label_associations'))
display(json.loads((OUT/'provenance.json').read_text()))
assert len(assign)==196*4*2
assert len(table('clip_assignments_sensitivity'))==515*4
assert metrics['api_calls']==0
print('Completed clustering in four spaces and two boundary views. No source audio changes, API calls or human labels.')
''')
notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
for c in cells:
    if c.cell_type=='code':compile(c.source,'<clustering notebook>','exec')
path=ROOT/'voice_clustering.ipynb';nb.write(notebook,path)
if '--execute' in sys.argv:
    from nbclient import NotebookClient
    def progress(cell,cell_index,**kwargs):print('Clustering cell',cell_index+1,'of',len(cells),flush=True)
    try:NotebookClient(notebook,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress).execute()
    finally:nb.write(notebook,path)
    assert not [o for c in notebook.cells for o in c.get('outputs',[]) if o.output_type=='error']
    print('Clustering notebook executed with zero errors')
