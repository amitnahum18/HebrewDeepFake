# Multi-space clustering plan — 2026-11-17

Offline extension of existing features and cached ECAPA vectors. No new generation, inference download, human labels or changes to source audio.

## Scope and units

- Same 515 clips: 231 real / 284 synthetic, grouped into 80 real IDs / 116 nominal provider:voice groups. Primary clustering uses 196 identity representatives so repeated versions are not independent speakers. Clip-level clustering is a separate sensitivity.
- Four spaces: existing core nine acoustic features; all 16 created acoustic descriptors as a redundancy sensitivity; 192-D L2 ECAPA centroids; core acoustic + ECAPA, with equal expected squared block norms. The hybrid is the requested features plus ECAPA comparison. Original and capped-edge-trimmed views use identical identities.
- Neither real/fake, provider/model, source sex/age, text nor IDs enter clustering features or cluster-number selection. They are joined afterward for interpretation. Source sex/age remain incomparable with synthetic labels.

## Preprocessing fixed before fitting

Acoustic group medians; log1p of nonnegative spectral centroid and energy range; log10 floor 1e-8 for highband fraction and spectral flatness; median imputation if necessary; drop truly constant features; RobustScaler median/IQR; cap scaled values to [-5,5] to limit isolated outliers. Report all transforms and clipping counts. All-16 space includes redundant quiet/edge measures, so it is a sensitivity, not extra independent evidence. A no-capping sensitivity is included at the selected k. ECAPA stays L2-normalized without per-dimension scaling. No dimensionality reduction in the primary clustering.

## Methods and selection

- KMeans, k=2..10, fixed seed, n_init=30. Select maximum full-space Euclidean silhouette among partitions with minimum cluster size >=8 and maximum <=90% of identities. If none meet these prespecified support constraints, flag unsupported rather than claim a natural number of clusters.
- Average-linkage hierarchical clustering at selected k as a method sensitivity, plus dendrogram. HDBSCAN min_cluster_size=8, min_samples=5, Euclidean, no forced single cluster. Report noise even if every identity is noise. Never select parameters to maximize real/fake separation.
- Silhouette measures internal compactness/separation; it is not detection accuracy or a guarantee of natural categories. Scores from different spaces are not directly comparable as model quality rankings.
- Stability: 10 seeded KMeans runs and 20 fixed-transform 80% identity subsamples at selected k; ARI against the reference on shared IDs. ARI is invariant to arbitrary cluster numbering. These describe conditional sample/initialization stability, not population confidence.
- PCA retaining >=90% variance then clustering at the same k, and acoustic no-capping sensitivity. Compare boundaries with both independently selected k and original fixed k. Same-k sensitivity avoids attributing every change to a changed cluster count.
- Clip sensitivity reuses identity-fitted acoustic preprocessing and block scale. Fit same k on 515 clips; report agreement within each real ID/nominal voice. No generalization claim.

## Deliverables

Executed notebook, CSV assignments at identity/clip levels, k-selection curves, PCA explained variance/loadings, shared PCA/t-SNE maps coloured by cluster then class/family, silhouette plots, cluster sizes/composition, native/scaled acoustic profiles, representative and distant members, dendrograms, stability and boundary sensitivity, Hebrew report and verification. Do not overwrite previous notebooks/PDFs.

## Interpretation limits

All synthetic recordings share one sentence, real clips vary. Codec and boundary history differ. Clusters can reflect recording conditions, text or level rather than synthesis/identity/demography. IDs are conservative nominal groups, not biometric proof. All identities come from a selected inventory; this is not population-representative Hebrew speech. Source labels are interpretation-only. No cluster is automatically removed or given a quality/real-fake label.

Primary documentation consulted: https://scikit-learn.org/stable/modules/clustering.html ; https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html ; https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html . Runtime scikit-learn version is recorded separately; documentation may describe a newer release.
