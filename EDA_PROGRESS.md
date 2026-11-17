# Crowd-recital EDA: progress and findings

## Priority findings

- **Contribution and demographic imbalance:** ten of 85 recorder IDs contribute 47.7% of hours. Reported sex is male for 46 speakers, female for 25, and unknown/unspecified for 14; these labels do not measure gender identity. No usable age proxy exists for 568 sessions (23.7%).
- **Technical review:** 22 recordings decode entirely to zero-valued audio. 121 have more than half of 20 ms frames below -45 dBFS; this can reflect pauses or low recording level and does not confirm background noise. 35 recordings exceed the declared duration-discrepancy rule; investigate container timestamps and source timing before cutting audio.
- **Alignment review:** 1,391 segments have zero duration, spanning 152 sessions. None of the aligned segments exceed decoded duration by more than 0.1 s. Thus the earlier decoded-bound invalidity count refers to zero-length segments, not audio overruns. The 37 additional reported-duration overruns disappear when decoded length is used. Session alignment score <0.3 affects 74 recordings, and normalized VTT/aligned text differs in 214 recordings.
- **Split feasibility:** the prespecified strict speaker/exact-segment-text graph puts 99.2% of hours in one component. Positive-length sensitivity checks at minimum segment lengths 1/5/10/20 words produce largest-component hour shares of 98.6%, 51.4%, 18.9%, 6.1%. A meaningful 70/15/15 split under the original grouping rule requires changing corpus selection or explicitly defining a different overlap protocol; the sensitivity results quantify alternatives rather than approving them. Do not silently split a connected group.
- **Review flags are not rejection decisions:** 1214 recordings have at least one broad screening flag, including demographics, statistical tails and alignment rules. No files were removed. Exact byte and canonical PCM duplicate searches found no duplicate recordings. Perceived noise and transcript correctness still require listening.


## Completed work
- Inspected the workspace, supplied research-context PDF, dataset documentation, manifest and JSON structure.
- Ran the user-requested Hugging Face download command before EDA execution and documented the analysis design in EDA_PLAN.md.
- Analyzed revision `34d93a876ae075a1a7904e7ec2c54d2af1c9d4d7` locally; credentials are never included in outputs.
- Inventoried and SHA-256 hashed 9,597 files; checked companion presence, schema, IDs, metadata agreement, transcript and word timing.
- Built recording, speaker, segment, word, alignment, acoustic, review, and split-connectivity tables.
- Decoded 2,398/2,398 recordings successfully; 0 failures retained explicitly.
- Generated feature distributions, demographic weighting, speaker concentration, transcript/alignment, acoustic, association, and outlier analyses.
- Saved tables, plots and provenance under `eda_outputs/`. The executed notebook is `dataset.ipynb`.

## Corpus size and concentration
2,398 sessions, 85 speakers, 50.47 reported recording hours, 325,152 manifest words and 54,122 manifest segments. Recomputed aligned counts: 325,152 word objects and 54,122 segments; these are separate counting conventions.

The top speaker contributes 6.1% of hours; the top five 26.9%; the top ten 47.7%. Hour-based Gini is 0.667, and the effective speaker count is 29.2. These quantify contribution imbalance without assuming a population demographic target.

## Demographic distributions
The dataset labels biological sex, not gender identity. Age at recording cannot be established because a reliable recording date is unavailable. The age proxy is 2026 minus birth year and must not be used as observed age. Unknown sex: 14 speakers. Missing birth year: 349 sessions. Conflicting speaker metadata is retained as a category.

| category | sessions | hours | speakers | field | sessions_pct | hours_pct | speakers_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| female | 481 | 18.074 | 25 | sex_label | 20.058 | 35.809 | 29.412 |
| male | 1739 | 27.896 | 46 | sex_label | 72.519 | 55.268 | 54.118 |
| unknown | 178 | 4.504 | 14 | sex_label | 7.423 | 8.922 | 16.471 |
| 0-17 | 22 | 1.361 | 1 | age_proxy_bin | 0.917 | 2.696 | 1.176 |
| 18-29 | 120 | 4.756 | 7 | age_proxy_bin | 5.004 | 9.422 | 8.235 |
| 30-39 | 831 | 14.417 | 24 | age_proxy_bin | 34.654 | 28.563 | 28.235 |
| 40-49 | 634 | 8.849 | 13 | age_proxy_bin | 26.439 | 17.531 | 15.294 |
| 50-59 | 53 | 2.186 | 10 | age_proxy_bin | 2.21 | 4.332 | 11.765 |
| 60-69 | 170 | 6.276 | 6 | age_proxy_bin | 7.089 | 12.435 | 7.059 |
| unknown | 568 | 12.629 | 24 | age_proxy_bin | 23.686 | 25.021 | 28.235 |

## Alignment and acoustic screening
| metric | flagged | denominator | pct |
| --- | --- | --- | --- |
| session_score_lt_0.3 | 74 | 2398 | 3.086 |
| segment_score_lt_0.5 | 3853 | 54122 | 7.119 |
| word_probability_lt_0.5 | 38881 | 325152 | 11.958 |
| vtt_normalized_mismatch | 214 | 2398 | 8.924 |
| manifest_word_count_mismatch | 0 | 2398 | 0.0 |

| codec | sample_rate | channels | sessions | decoded_hours |
| --- | --- | --- | --- | --- |
| aac | 16000 | 1 | 39 | 1.033 |
| aac | 24000 | 1 | 2 | 0.003 |
| aac | 44100 | 1 | 18 | 1.215 |
| aac | 48000 | 1 | 65 | 1.999 |
| opus | 48000 | 1 | 2274 | 46.143 |

| flag | flagged_sessions | all_sessions | pct |
| --- | --- | --- | --- |
| low_session_alignment | 74 | 2398 | 3.086 |
| low_segment_alignment | 787 | 2398 | 32.819 |
| duration_mismatch | 35 | 2398 | 1.46 |
| decode_failure | 0 | 2398 | 0.0 |
| quiet_majority | 121 | 2398 | 5.046 |
| low_rms | 41 | 2398 | 1.71 |
| near_fullscale_candidate | 0 | 2398 | 0.0 |
| dc_offset_candidate | 0 | 2398 | 0.0 |
| nonfinite_audio | 0 | 2398 | 0.0 |
| invalid_birth_year | 219 | 2398 | 9.133 |
| nonpositive_duration | 0 | 2398 | 0.0 |
| empty_transcript | 0 | 2398 | 0.0 |
| structural_issue | 171 | 2398 | 7.131 |

There are 1,214 sessions with at least one screening flag. Flags overlap and include provider segment-confidence guidance, so their sum is not a rejection count. See `outlier_rules.csv` for robust numerical thresholds and `review_candidates.csv` for all reasons. Quiet fractions depend on the documented energy threshold. Energy dynamic range is not SNR. Near-full-scale values may reflect lossy overshoot, so clipping candidates need listening and waveform review.

## Integrity, repetitions, and split risk
2,856 integrity issue occurrences and 0 manifest/metadata disagreement entries were retained for review. There are 0 recordings participating in exact file duplicates and 0 in canonical PCM duplicates; these are recording counts, not duplicate-group counts. There are 659 exact normalized segment texts shared across speakers. The largest strict speaker/text connected component accounts for 99.2% of recording hours, which constrains a split requiring both speaker and exact text independence. No split was assigned.

## Recommended actions
1. Review decode, timestamp and metadata integrity issues before selecting training data; retain documented reasons and original files.
2. Listen to the flagged queue and `human_listening_sample.csv`, which selects one seeded baseline session per speaker. Human ratings are pending; no perceptual-noise claim is established.
3. Select demographic and speaker-duration quotas with unknowns visible. Evaluate retention rates by speakers and hours, not just session counts, before setting exclusions.
4. Confirm alignment/text mismatches manually. Provider scores measure alignment, and are not measured WER or transcript ground truth.
5. Establish speaker/text grouping and verify largest-component feasibility before freezing evaluation splits. Browser codec, sample rate, channel and source signatures can confound a future real/synthetic experiment.
6. Obtain reliable demographic provenance or recording dates if gender, accent or age-at-recording analyses are required. Do not infer them from voice.

## Limits and remaining human work
This is a descriptive audit of read Hebrew, not a representative sample of Hebrew speakers or spontaneous speech. Sessions repeat speakers and documents; no independent-session significance tests or causal claims were made. Vocabulary is whitespace-based rather than morphological/phonemic coverage. Exact normalization can merge distinctions; perceptual audio duplicates, paraphrases, stable cross-ID speaker identity, speech/noise separation, calibrated SNR, ASR WER, single-speaker verification and human listening are not established. No acoustic denoising, automatic exclusion, synthetic generation or classifier training was performed.

## Reproduction and references
Open `dataset.ipynb` in this project and run all cells using the recorded package versions in `eda_outputs/run_provenance.json`. Set `CROWD_RECITAL_PATH` to move the same input revision. The notebook requires no network after acquisition. Original bytes are hashed in `file_inventory.csv`; current source hashes do not independently authenticate upstream provenance.

- Dataset schema and provider screening guidance: https://huggingface.co/datasets/ivrit-ai/crowd-recital
- Robust outlier screening and cautions: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
- Comparative demographic/speaker reporting methodology: https://github.com/common-voice/cv-dataset/blob/main/datasets/scripted-speech/README.md

## Analyst validation

Birth year `0` occurs in 219 sessions. Missing birth years affect 349 sessions, and invalid/placeholder years affect 219. Together, 568 sessions (23.7%) have no usable age proxy. Original years are retained; the zero placeholder is not interpreted as an age.

Using a 0.1-second tolerance, 1428 aligned segments in 171 sessions violate bounds relative to reported recording duration. Against actual decoded duration, 1391 segments in 152 sessions still violate bounds. There are 0 segments without a decoded duration available. See `timestamp_bounds_validation.csv`; initial reported-duration flags remain intact, and overlapping metadata/alignment issue rows should not be counted as independent failures.

## Final verification and sensitivity interpretation

All 12 code cells executed without errors. Final reconciliation passed for 2,398 recordings, 85 recorder IDs, 9,597 source hashes, 40 CSV tables, eight figures and the 85-speaker baseline listening sample. See eda_outputs/verification.json. Representative figures were visually inspected.

The 20-word leakage sensitivity condition has only seven eligible segments, so its low component share does not establish general absence of leakage. The five-word rule retains 35,889 positive-length segments and yields a 51.4% largest-component share; it changes the prohibited-overlap definition and requires an explicit protocol decision. No split was assigned.

## Supplementary human voice review

2026-11-16: Completed the user-led listening review of one excerpt for each of 14 recorder IDs with missing/unspecified source sex. The user assigned 11 perceived-masculine and 3 perceived-feminine labels. These excerpt-level auditory-impression labels are stored separately under voice_review/ with provenance and an event history. Original demographic metadata and EDA tables are unchanged; no labels were propagated to unheard recordings.
