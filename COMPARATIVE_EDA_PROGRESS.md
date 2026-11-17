# Comparative EDA progress

## 2026-11-16 — completed offline analysis

- Read both executed notebooks, source feature tables, review limitations and saved WAV preparation code. No new generation/API requests or human labels.
- Fixed definitions before computation in `COMPARATIVE_EDA_PLAN.md`. Consulted primary ASVspoof guidance and CVPR shortcut research, cited in the notebook/report.
- Reconciled 2,398 real recordings / 85 recorder IDs and 284 synthetic configurations / 116 nominal provider:voice groups.
- Built a sequential real-utterance QC funnel and threshold sensitivity; 50,103 broad eligible utterances, 38,517 at 0.8/0.8 scores. Scores are not calibrated acoustic quality.
- Audited demographic source/proxy coverage, synthetic exposure by family/voice, codec history, exact text overlap, and seven synthetic review candidates. Kept unknown/conflicting labels visible. Rejected uniform automatic synthetic age output; no biological-age matching.
- Recomputed identical waveform and spectral measurements on 362 existing canonical WAVs: 78 selected real-reference speakers and 284 synthetic samples. Compared group medians; added effect sizes, 1,000-repeat group bootstrap stability intervals and analysis-only trim/level sensitivities.
- Found a major boundary-preparation shortcut: edge-quiet orientation-free descriptive AUC ~0.999 before trimming. This is not detector accuracy. Existing real reference cuts use alignment boundaries while synthetic clips retain full generation output. Source codecs and texts also confound labels.
- Solved equal-family diagnostic duration/word matching: 44 pairs, 11 per family, unique identities, eight words and <=0.5 s caliper. These retain different texts and are not proposed as a controlled benchmark.
- Prepared 308 globally exact-text-unique real utterance candidates from 81 speakers (up to four each), provisionally 77 per family. Four remaining source IDs are retained in coverage audit. Model, voice and split deliberately unset; no synthetic audio generated from these texts. Candidate cuts require listening and transcript/boundary validation.
- Created and executed `real_synthetic_comparison.ipynb`: nine code cells, zero execution errors. Saved CSVs, four plots, provenance/input hashes and detailed Hebrew recommendations in `DATASET_MATCHING_RECOMMENDATIONS_HE.md`.
- Verification: `python verify_comparative_eda.py`; includes all 362 unchanged canonical-audio hashes, notebook execution, input hashes, eligibility reconciliation, proposed constraints and family-balanced diagnostic matching. See `comparison_outputs/verification.json` for results.
- Final QA passed: all automated checks, 17 CSVs and four plots. Visually inspected acoustic ECDF and effect-sensitivity plots; axes, legends and group comparisons are readable. Nine notebook code cells completed without errors. No source audio changed.

## Remaining next-stage work, not part of this EDA

Human listening/ratings; validate proposed source boundaries/transcripts; freeze split and symmetric preprocessing policy; assign stock voices per partition; generate a small true same-text paired batch; repeat confound audits before volume expansion. Existing one-sentence pilot stays calibration-only. No 20-hour generation was started.
