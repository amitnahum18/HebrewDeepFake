# Synthetic Hebrew voice pilot: findings and progress

## What was completed
284 of 284 planned configurations decoded successfully across 10 models, 4 provider families and 116 nominal provider/voice groups. Generated audio totals 22.11 minutes. Every request used the same Hebrew text: היום ניפגש בשעה שלוש ליד תחנת הרכבת בירושלים.

Original audio and derived analysis WAV have separate hashes. No reference-audio cloning or corpus-volume expansion occurred. The recorded stock-voice catalog, exact requests, generation IDs, billing records and failed review attempts are retained. Initial audio-review JSON-mode failures were diagnosed; after removing the incompatible parameter, explicit repair attempts were logged rather than silently replacing history.

## Coverage and Hebrew screening
| family | model | planned | generated | decoded | technical_pass | automatic_hebrew_pass | combined_pass | seconds | minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elevenlabs | elevenlabs/eleven-v3 | 21 | 21 | 21 | 21 | 21 | 21 | 85.44 | 1.424 |
| elevenlabs | elevenlabs/eleven-v3-conversational | 21 | 21 | 21 | 21 | 21 | 21 | 91.68 | 1.528 |
| elevenlabs | elevenlabs/eleven-v4 | 21 | 21 | 21 | 21 | 21 | 21 | 85.92 | 1.432 |
| elevenlabs | elevenlabs/eleven-v4-turbo | 21 | 21 | 21 | 21 | 21 | 21 | 85.92 | 1.432 |
| google | google/gemini-3.1-flash-tts-preview | 30 | 30 | 30 | 30 | 30 | 30 | 142.24 | 2.371 |
| google | google/gemini-3.8-flash-lite-tts | 30 | 30 | 30 | 30 | 30 | 30 | 176.08 | 2.935 |
| google | google/gemini-3.8-flash-tts | 30 | 30 | 30 | 30 | 30 | 30 | 140.36 | 2.339 |
| minimax | minimax/speech-2.8-hd | 45 | 45 | 45 | 45 | 45 | 45 | 221.413 | 3.69 |
| minimax | minimax/speech-2.8-turbo | 45 | 45 | 45 | 45 | 45 | 45 | 213.278 | 3.555 |
| speechify | speechify/simba-3.5 | 20 | 20 | 20 | 20 | 20 | 20 | 84.48 | 1.408 |

284 configurations passed technical screening and 284 passed automatic Hebrew/text screening. The combined passed count is 284. Language passage is an audio-model transcription/recognition check, not human verification; corrections remain possible in review.html. Failed or unverified clips remain in the manifest and are not counted as automatically accepted Hebrew samples.

## Perceptual voice groups — unvalidated estimates
| unit | field | category | count | denominator | pct |
| --- | --- | --- | --- | --- | --- |
| provider_voice_groups | perceived_presentation | masculine | 67 | 116 | 57.759 |
| provider_voice_groups | perceived_presentation | conflicting_across_models | 35 | 116 | 30.172 |
| provider_voice_groups | perceived_presentation | feminine | 14 | 116 | 12.069 |
| provider_voice_groups | perceived_age_band | uniform_estimator_not_validated | 116 | 116 | 100.0 |

These are casting impressions from very short synthetic clips, not real ages, biological sex or verified gender identities. Below-0.6 self-confidence is mapped to uncertain. The model's confidence is not calibrated. 35 nominal voice groups have conflicting presentation estimates across model versions. Do not hide those conflicts with majority voting. Source-provided voice labels are reported separately in provider_labels.png and voice_summary.csv.

Age estimator uninformative: **True**. The audio model returned the same age category for all clips, including voices with different source age descriptions. This failed informativeness check prevents presenting those outputs as a usable age distribution. All raw estimates remain in raw_age_model_outputs.csv; perceived age requires human listening or a separately validated estimator.

## Acoustic patterns and outliers
The notebook includes numerical percentiles, model comparisons, quiet-frame fractions, waveform levels, format history, duplicate screening and robust outlier flags. There are 7 configurations with one or more broad review flags. No recordings were deleted. Silence thresholds are not VAD, dynamic range is not SNR, and near-full-scale samples do not establish clipping. Repeated text is deliberate and lexical coverage is therefore fixed, unlike the real corpus.

## Real-speech reference
78 utterances from 78 real recorder IDs were processed at the same analysis sample rate. Each was selected to be 3-8 seconds and 6-10 aligned words, with source alignment score >=0.8. They are not the same sentence and have different recording/channel histories. Comparisons must not be presented as proof of a synthesis effect or used to claim classifier accuracy. This subset is not a replacement for the full 85-recorder dataset.

## Costs
{
  "checked_utc": "2026-11-16T19:50:06.800093+00:00",
  "speech_requests_with_cost": 284,
  "speech_requests_total": 284,
  "reported_speech_cost_usd": 0.695038,
  "audio_reviews_with_cost": 284,
  "reported_review_cost_usd": 0.1301988,
  "latest_key_usage_delta_usd": 0.8170324,
  "billing_complete": true,
  "key_usage_note": "Key usage may lag and fluctuate between cached snapshots; per-request records are preferred.",
  "reported_request_cost_sum_usd": 0.8252368000000001
}

Missing or incomplete billing is explicitly marked and must not be interpreted as free generation. API usage snapshots can be eventually consistent. The project limit remains $20; this pilot does not authorize spending the remaining balance on a 20-hour dataset.

## Next steps
Listen to review candidates and the seeded one-example-per-voice baseline. Rate perceived presentation, casting-age category, Hebrew correctness and intelligibility; export human ratings from the workbench. Double-rate a subset before treating perceptual labels as reliable. Choose text and voice/engine split rules before any training experiment, and retain the repeated-text pilot as a clearly tagged calibration/EDA subset. The four provider families are not proof of architectural independence, and models sharing voice identifiers are dependent.
