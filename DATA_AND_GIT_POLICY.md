# What is versioned and what stays local

## Included

- Analysis/generation/review scripts and pinned analysis dependencies.
- Executed notebooks, methods/progress documents, Hebrew reports and the English PDF summary.
- Compact derived feature tables, cached ECAPA vectors/identity centroids, clustering matrices and assignments, figures and provenance/verification snapshots.
- Synthetic planned/generated manifests, final review CSV, review-attempt history, reported costs, public catalog/voice snapshots and proposal manifests.
- Selected real-source summaries and audit outputs. `repo_snapshot_manifest.json` records the exact inclusion list and file hashes.
- At the user's explicit request, `similar_voice_pairs/listen_portable.html` embeds selected real and synthetic listening excerpts: six pairs per group A/B/C and consensus. This is an exception for the playable listening artifact, not publication of all audio. Raw/listening-level representations are copies, not new independent recordings. The snapshot manifest records the number of selected underlying recordings.

## Excluded

- Standalone original real/synthetic audio and derived real-reference WAVs, including per-recording audio-folder JSON sidecars. Selected excerpt bytes are embedded only in the explicitly requested portable listening HTML described above.
- Model weights, local Python environments, dependency installation folders, cache/temp files and local browser/server state.
- API keys, environment files, raw account/auth responses and the local budget-state guard.
- Bulky real-source transcript/word/segment exports and raw full-recording feature tables containing full text. Rebuild from the pinned source snapshot instead of treating Git as a dataset store.

The publication snapshot preserves numerical values and saved run records rather than rewriting historic hashes/paths. Original-machine absolute paths in metadata and outputs do not work on a fresh clone. Cached-feature clustering is portable because it uses project-relative table locations and the stored vectors; listening/extraction/review UI requires local audio restored or regenerated. Snapshot verification checks committed-file integrity; historic full analysis verifiers may require excluded audio/source exports and run-specific original paths.

## Units and label policy

284 generated samples represent 116 nominal voice groups across ten model variants. Analysis variants and copies do not create additional independent speakers. Keep provider casting labels, source demographics, model estimates and human listening judgments separate. Automatic synthetic age was uninformative and must not be used as measured age. Seven synthetic flags are review candidates; zero completed human-verification records is the current status.

## Credentials

Use environment variables or the existing secure local prompt path; no key belongs in code, a notebook, an artifact, a commit message or this repository. Publication uses an allowlist and scans the exact candidate files for credential patterns before staging. This check complements review; it is not a reason to upload account/auth responses.

## Reproduction boundaries

Read README.md for offline cached-feature clustering. Follow stage-specific READMEs to download source data and model weights and rebuild larger exports. New synthesis/review requests can incur charges and are not executed merely by cloning this repository or running clustering. The existing $20 budget is not an instruction to generate 20 hours automatically.
