# Multi-method voice distribution analysis progress

## 2026-11-17 — local execution

- Read the official ECAPA paper, SpeechBrain model card and model hyperparameters. Fixed the analysis scope, units, preprocessing sensitivity and interpretation limits in `VOICE_DISTRIBUTION_PLAN.md`.
- Found incompatible global Torch 2.13 / Torchaudio 2.8. Created `.venv-ecapa` with inherited scientific packages and installed CPU Torch 2.8.0 plus SpeechBrain 1.1.2 locally. Global package installations were not changed. Inherited torchvision is incompatible with local Torch but is not imported/used by this workflow.
- Downloaded only official embedding weights and hyperparameters at revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`; saved hashes in `voice_distribution/model_provenance.json`. No recordings were uploaded, and no TTS/review API calls were made.
- Selected up to three real utterances per qualified source ID, preferring distinct sessions/texts and duration near 4.5 seconds, with source/alignment >=0.8, 3–8 seconds and 6–14 words.
- Initial PTS-based cutting exposed browser-MKA packet timestamp jitter: 90 of 231 trial slices would fail 98% timestamp coverage if every small gap were treated as silence. Preserved these attempts in `real_pts_extraction_attempts.csv`. This was an extraction-method issue, not 90 proven bad recordings. Switched to the continuous decoded sample clock consistent with previous EDA; no inserted gap zeros. All 231 final slices decoded successfully, from 80 source IDs. Five source IDs had no eligible candidate and remain in coverage audit. Final boundaries/transcripts are not human-verified.
- Added all 284 existing synthetic clips / 116 nominal provider:voice groups. Extracted 1,030 finite L2-normalized 192-D ECAPA vectors for 515 samples in original and capped-edge-trimmed views. Official Fbank/normalization/ECAPA layers, strict weight loading, CPU inference; no classifier or age/sex model. Inference took approximately 424 seconds.
- Real-speaker sanity check: same-ID versus different-ID descriptive pair AUC ~0.9967, cross-session ~0.9963 over 200 positive pairs from 69 supported source IDs. These are dependent exploratory speaker-ranking pairs, not deepfake detection or held-out generalization results.
- Compared within/between identity cosine distributions, nearest cross-class neighbours, identity-centroid RBF MMD at three bandwidths, equal-family weighting and repeated equal-size (20 per family) comparisons. Added shared PCA/t-SNE and acoustic-feature views. PCA 2-D retained only ~12.4% variance, so projections are illustrative.
- Added a sensitivity check after identifying unequal clip-averaging as a potential confound: randomly select one clip per identity, 50 repeats per view. It complements centroid analysis and does not sample new speakers or remove text/codec confounds.
- Wrote the Hebrew results report, reproducibility README, executed notebook, detailed CSVs and eight plots. No detector trained, no human labels inferred, no source deletion or original audio modification.
- Final execution and verification passed: eight notebook code cells, zero errors, all vector/audio/model/provenance checks and the 50-repeat-per-view single-clip sensitivity. Visually inspected shared projections, identity ECDFs, acoustic comparison and equal-candidate-count family coverage. Report explicitly distinguishes acoustic-neighbour coverage from provider quality and keeps one-sentence/content/codec limitations visible.

## Reproducibility and QA

Commands and dependencies: `README_VOICE_DISTRIBUTIONS.md`.

`python build_voice_distribution_notebook.py --execute`

`python write_voice_distribution_report.py`

`python verify_voice_distributions.py`

Notebook inference is cached: analysis reruns read saved vectors and stay offline. Verification checks all 515 WAV hashes, 1,030 vector bindings/norms, 196 centroids per view, official weight hashes, source-ID accounting, pair semantics, notebook errors and figure presence. See `voice_distribution/verification.json` for final results.

## Required follow-up before controlled matching

Listen to candidate similar/unstable voices; validate real cut boundaries/transcripts; create same-Hebrew-text pairs with symmetric channel/boundary processing inside frozen identity/text partitions; repeat the analysis. Existing single-text pilot is calibration/coverage data, not population-representative Hebrew or a clean synthesis-effect experiment.
