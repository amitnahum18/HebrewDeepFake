# Human listening review

This supplementary review is independent of the source `biological_sex` field. Labels describe the user's auditory impression of the reviewed excerpt: masculine, feminine, uncertain, or multiple speakers. They do not verify biological sex or gender identity and are never substituted for the original metadata.

The queue selects one representative recording for each of the 14 recorder IDs with missing/unspecified source sex. Initial selection prefers decodable, nonzero audio of at least eight seconds, then higher alignment confidence and fewer quiet frames. Selection is a convenience for listening, not a claim of representativeness.

`review_queue.csv` retains recording and speaker IDs, source metadata, annotation provenance and scope, and pending/presented/reviewed status. `current_excerpt.json` records the presented excerpt's source, timing and SHA-256. No label is assigned before the user answers. A label applies only to the reviewed excerpt; other recordings require verification before any separate speaker-level annotation is proposed.

Listening files are derived 16 kHz mono excerpts, normally seconds 1-13. WAV is available without additional lossy encoding; MP3 is a playback convenience. Original recordings and all EDA outputs remain unchanged.
