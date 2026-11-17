# Real/synthetic Hebrew comparative EDA plan

Date: 2026-11-16. This work is offline: no new synthetic generation, paid review calls, source exclusions or frozen evaluation splits.

## Questions and units

1. Which quality issues, exposure imbalances and metadata gaps remain in each source?
2. Which observed class differences could be explained by text, utterance length, codec history, sample rate, level or edge padding?
3. What matching is feasible with the existing pilot, and what additional data would support a credible detection experiment?

The real corpus has session/speaker/segment units. The synthetic pilot has model/voice configurations and repeated nominal provider/voice identity groups. These units must not be conflated. Perceived synthetic age is not comparable with real birth-year metadata; source biological sex is not interchangeable with a short-clip auditory impression. The 14 human-reviewed real excerpts remain excerpt-scoped annotations and will not be propagated to other recordings.

## Planned analyses

- Reconcile the existing executed EDA tables, synthetic manifest/reviews, source voice labels, billing and recording inventory.
- Build a sequential real-segment eligibility funnel: successful source decoding and nonzero audio; valid positive segment timing within decoded duration; nonempty text/word count; source session score >=0.3; segment alignment score >=0.5. Treat acoustic outliers as review flags, not automatic exclusions. Report sensitivity at session/segment score thresholds 0.3/0.5 and 0.8/0.8 and prototype utterance lengths 2-10 seconds, 6-14 aligned words.
- Inspect retained speakers, exposure concentration, source-demographic coverage and exclusion counts at each step. Invalid birth years must not exclude otherwise valid audio.
- Recompute acoustic features from the exact saved 16 kHz mono PCM16 WAVs for all synthetic samples and the existing 78-person human reference: duration, RMS, sample peak, DC, near-full-scale fraction, zero crossings, quiet fractions at -50/-45/-40 dBFS, leading/trailing quiet-frame estimates, spectral centroid, spectral flatness and energy above 4 kHz. Energy masks are not VAD or calibrated noise/SNR.
- Include two analysis-only sensitivities: cap removal of quiet edge frames at 0.25 seconds per side; rescale waveform RMS toward -23 dBFS subject to sample-peak headroom. Do not write these transformed views over source or analysis audio. Matching uses original-duration/word-count covariates, not these potentially discriminatory sound features.
- Show group-median comparisons so repeated synthetic voice IDs are not treated as independent speakers. Use standardized mean differences, median differences, Cliff's delta and Wasserstein distances. Bootstrap mean differences over the observed speaker/nominal-voice groups with a fixed seed, 1,000 repetitions. Intervals describe sensitivity to observed group composition, not a representative population sample.
- Calculate univariate, orientation-free in-sample label separability only as a shortcut diagnostic. It is not trained detector performance, generalization, EER or evidence of a synthesis-specific cue. No classifier will be trained.
- Audit exact normalized text overlap, lexical coverage, native format confounding, class/unit ratios, nominal voice/family balance, duplicate text/audio and source-label availability.
- Construct duration/word-count matched **diagnostic** pairs: real reference utterances with exactly eight aligned words; one synthetic sample and at most one instance of each provider/voice group; absolute duration caliper <=0.5 seconds. Require equal synthetic-family counts, cap each family at floor(eligible real utterances / 4) and at 20. Maximize matches before minimizing absolute duration difference. Existing texts are different; these are never described as true text-matched real/fake pairs.
- Build a separate proposal for the next text-matched experiment, without generating it: select up to four globally exact-text-unique, qualified source utterances per eligible real speaker, round-robin. Require 2-10 seconds, 6-14 aligned words, session and segment scores >=0.8, and at least 80% Hebrew letters among letters. Propose one family per utterance, with capacity-balanced family assignment. Actual voice assignments and train/dev/test allocations remain unset until protocol approval and QC. Include all 85 source IDs in the coverage table, even when a speaker has no eligible prototype utterance.
- Deliver and execute a new comparison notebook, CSVs, plots, a detailed Hebrew recommendation report, and a concise data-handling/split policy. Verify counts, hashes, no notebook errors and matching constraints; visually inspect representative plots.

## Primary guidance

- ASVspoof 2021 evaluation plan (coding/transmission and compressed-audio conditions): https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf
- Smeu et al., CVPR 2025 (dataset-specific leading-silence shortcuts): https://openaccess.thecvf.com/content/CVPR2025/papers/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.pdf
- Dataset schema and alignment-score semantics: https://huggingface.co/datasets/ivrit-ai/crowd-recital

## Progress

- Inspected both executed EDA outputs and implementation. Confirmed zero exact real-segment matches to the pilot sentence, disjoint native codec sets, and 59 eight-word utterances among 78 human reference excerpts. Analysis definitions and matching calipers were fixed before computation.
