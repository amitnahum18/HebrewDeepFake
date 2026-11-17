# Quality and matching audit progress

## 2026-11-17

- Extended prior EDA to whole-corpus exact normalized text duplication, lexical near-session candidate search, source-demographic/time representation and concrete quality review queues.
- Preserved full recordings, eligible-segment pool, 78-speaker short reference and 308 proposed pair subsets as distinct scopes. Measured speaker-share and time-share selection shifts instead of calling the selected reference population-representative.
- Counted exact duplicate PCM fingerprints from previous full-real decoding; reopened all 362 saved canonical WAVs for an additional common sample-payload fingerprint check. No exact duplicate sample payloads found in that saved set. Full perceptual near-audio search was not performed.
- Located 1,151 repeated segment-text groups / 3,657 rows; 259 groups / 615 rows among eligible segments of at least six normalized words. 145 of the latter groups span speakers. Exported every occurrence with session, speaker and time coordinates.
- Located 12 repeated whole-session transcript groups / 45 sessions, including short headers. Screened all session pairs of >=20 source words for unigram/bigram TF-IDF cosine >=0.92, excluding exact normalized matches: 12 near-text candidate pairs. These are review candidates, not confirmed audio duplicates.
- Exported real priority review flags and seven synthetic outlier candidates. Added identity-coverage listening manifests accounting for all 85 real IDs and 116 nominal synthetic voice groups, with no new extraction, generation or human labels.
- Wrote the detailed Hebrew specification `QUALITY_AND_MATCHING_SPEC_HE.md`: actual duplicate examples, scope limitations, QC handling, pair construction, controlled vs deployment-oriented sampling, split leakage policy and staged acceptance criteria.
- Built `quality_matching_audit.ipynb`. Local notebook launch initially could not read the installed user kernel directory under the filesystem sandbox. Re-ran the same offline execution through approved escalation; project outputs stay in the workspace.
- Completed execution: six code cells, zero errors. `verify_quality_audit.py` passed all checks, including count reconciliation, 362 unchanged WAV file hashes, sample-payload uniqueness, 85/116 identity coverage, near-text thresholds and demographic share normalization. Saved 18 CSV tables and two figures. Visually inspected the source-sex representation figure; legends and both weighting views are readable.

## Reproduce

`python build_quality_audit_notebook.py --execute`

`python verify_quality_audit.py`

Execution and reconciliation results are recorded in `quality_audit/verification.json`. No API spend, source deletion, new synthetic audio or human judgment was performed in this audit.
