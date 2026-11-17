# HebrewDeepFake: Hebrew real and synthetic speech analysis

Exploratory analysis, quality auditing and unsupervised voice clustering for Hebrew speech. This repository contains reproducible analysis code, executed notebooks, reports, compact feature tables and cached ECAPA representations. It is **not** a trained deepfake detector or a completed 20-hour synthetic corpus.

## Current inventory

| Source | Recordings / excerpts | Identity groups | Scope |
|---|---:|---:|---|
| Real Crowd Recital source | 2,398 sessions, ~50.47 reported hours | 85 recorder IDs | Full source EDA; audio obtained separately from Hugging Face |
| Synthetic pilot | 284 short recordings, 22.11 minutes | 116 nominal provider:voice groups | 10 recorded model variants, four families, one shared Hebrew prompt |
| Multi-utterance voice sample | 231 real + all 284 synthetic clips | 80 real + 116 synthetic groups | ECAPA, acoustic distributions and clustering |

Synthetic stock voices are not clones of the real speakers. Shared provider:voice identifiers across model versions are grouped conservatively, not asserted to be biometric identities. All 284 synthetic samples passed final automated screening, but none has a completed human-verification label; seven are in the listening review queue. Automatic synthetic age estimates were rejected as uninformative.

## Start with these results

- [Professional English PDF summary](output/pdf/HebrewDeepFake_Synthetic_Data_Generation_Report.pdf): models, file inventory, metadata, generation, quality and imbalance handling.
- [Clustering report in Hebrew](CLUSTERING_REPORT_HE.md): acoustic-only, ECAPA and hybrid analysis, limitations and 20 plots.
- [Multi-method voice distribution report](VOICE_DISTRIBUTIONS_REPORT_HE.md).
- [Quality, duplicates and matching specification](QUALITY_AND_MATCHING_SPEC_HE.md).
- [Real/synthetic matching recommendations](DATASET_MATCHING_RECOMMENDATIONS_HE.md).

## Executed notebooks

| Notebook | What it covers |
|---|---|
| [dataset.ipynb](dataset.ipynb) | Full real-source EDA: structure, speaker/time exposure, source metadata, text/alignment, audio quality and leakage |
| [synthetic_eda.ipynb](synthetic_eda.ipynb) | Generated pilot inventory, technical/Hebrew review, demographics limitations and costs |
| [real_synthetic_comparison.ipynb](real_synthetic_comparison.ipynb) | Common audio features, effects, confounds and proposed same-text pairs |
| [quality_matching_audit.ipynb](quality_matching_audit.ipynb) | Exact-text repeats, lexical near-duplicates and selection/representation audits |
| [voice_distributions.ipynb](voice_distributions.ipynb) | ECAPA consistency, distance distributions, coverage, PCA/t-SNE, kernel distances and acoustic sensitivities |
| [voice_clustering.ipynb](voice_clustering.ipynb) | Four feature spaces, KMeans/hierarchical/HDBSCAN, internal selection and stability |

Notebooks retain saved outputs for review. Some historical outputs contain original-machine paths and are archival run records, not portable data locations. Regenerate audio-dependent tables after obtaining the source data on another computer. Do not change a path in a hashed manifest and assume historical integrity checks will remain valid.

## Features and clustering results

- `synthetic_pilot/analysis/sample_features.csv`: 284 synthetic rows with generation, acoustic and automated-review metadata.
- `voice_distribution/sample_acoustic_features.csv`: all 515 sampled clips; filter `label == "synthetic"` for the synthetic subset.
- `voice_distribution/embeddings.npz`: 1,030 unit-normalized 192-D vectors, keyed by sample ID, view and WAV hash; contains original and capped-edge-trimmed views.
- `voice_distribution/centroids_*.npz`: one vector per nominal identity, in a recorded order.
- `clustering_outputs/identity_assignments.csv`: assignments in four spaces and two boundary views.
- `clustering_outputs/clip_assignments_sensitivity.csv`: separate clip-level sensitivity, not independent-speaker observations.
- `clustering_outputs/figures/`: 20 detailed plots. [Acoustic maps](clustering_outputs/figures/acoustic_core_original_maps.png), [ECAPA maps](clustering_outputs/figures/ecapa_original_maps.png), [hybrid maps](clustering_outputs/figures/hybrid_original_maps.png), [selection curves](clustering_outputs/figures/k_selection_overview.png).

## Run clustering from the included cached features

Use Python 3.12. An isolated environment is recommended; the analysis runs on CPU and needs no API key, audio download or model weights.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python cluster_voice_features.py
python build_clustering_notebook.py --execute
python verify_clustering.py
```

For a dependency-free check of the published snapshot itself: `python verify_repo_snapshot.py`. The repository preserves historical artifact bytes across platforms so recorded hashes remain meaningful.

The notebook reads existing results by default; set `RECOMPUTE=True` in its first code cell or run the analysis script to fit again. Class/provider/demographic labels are interpretation-only. Primary clusters use 196 identity representatives, not 515 dependent clips. Never treat cluster numbers as true real/fake or demographic labels.

## Reproduce audio-dependent analysis

Obtain `ivrit-ai/crowd-recital` separately. The analyzed revision is `34d93a876ae075a1a7904e7ec2c54d2af1c9d4d7`.

```powershell
hf download ivrit-ai/crowd-recital --repo-type dataset --revision 34d93a876ae075a1a7904e7ec2c54d2af1c9d4d7
```

Set `CROWD_RECITAL_PATH` to the downloaded snapshot if it differs from the usual Hugging Face cache. Follow [README_EDA.md](README_EDA.md), [README_VOICE_DISTRIBUTIONS.md](README_VOICE_DISTRIBUTIONS.md) and the plan/progress documents for each stage. Full raw-derived transcripts, word/segment tables and WAVs are not all included in Git; rebuild them before running audio-dependent notebooks or full integrity checks. See [DATA_AND_GIT_POLICY.md](DATA_AND_GIT_POLICY.md).

New TTS generation and automated audio review are optional, paid operations. See [README_OPENROUTER.md](README_OPENROUTER.md). Read `OPENROUTER_API_KEY` from the local environment; never commit it. Running cached clustering does not generate speech or spend API credits.

## Main limitations and next step

All synthetic clips read one sentence while real clips contain varied text. Codec history, boundaries and source conditions confound comparisons. PCA/t-SNE are visualizations, not proof of natural groups; silhouette is internal separation, not detector accuracy. Human listening and common perceptual labels remain incomplete.

Next: validate real source cuts and create same-Hebrew-text real/synthetic pairs with symmetric preprocessing inside frozen speaker/voice/text partitions. The 308-pair manifest is a proposal only; no audio was generated from it. Keep calibration, controlled evaluation and deployment-oriented evaluation distinct.

## Sources and reproducibility

Primary methodology references are linked in the reports and notebooks: ECAPA-TDNN (Interspeech 2020), ASVspoof 2021, shortcut research (CVPR 2025), and official scikit-learn documentation. All project result numbers come from local manifests and analyses, not paper benchmark scores. Versions, seeds, parameter settings and input hashes are recorded beside the outputs.

The existing repository [LICENSE](LICENSE) is retained. External datasets, model weights and provider materials have their own source terms; this repository does not include the raw audio or downloaded model weights.
