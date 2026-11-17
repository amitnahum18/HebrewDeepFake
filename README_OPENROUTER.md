# OpenRouter integration and Hebrew synthetic pilot

This Python project uses the official OpenAI Python client with `base_url=https://openrouter.ai/api/v1`, following [OpenRouter's quickstart](https://openrouter.ai/docs/quickstart) and [speech API documentation](https://openrouter.ai/docs/guides/overview/multimodal/tts).

## Install and authenticate

```powershell
python -m pip install -r requirements-openrouter.txt
```

The client reads only `OPENROUTER_API_KEY` for authentication. Create a key at https://openrouter.ai/keys if needed. Never paste a key into code, notebooks, output metadata or chat. `.env` and local credential files are ignored by Git, and this integration does not create them.

To set the variable in the current PowerShell session without typing the key into shell history:

```powershell
$routerSecureKey = Read-Host 'OpenRouter API key' -AsSecureString
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new('', $routerSecureKey).Password
python openrouter_example.py --json
Remove-Item Env:\OPENROUTER_API_KEY
Remove-Variable routerSecureKey
```

The default text model is `openai/gpt-4o-mini`. Override it with `--model provider/model-slug` or `OPENROUTER_MODEL`. The SDK has no automatic retries; authentication, credit, rate-limit, timeout, connection, empty-response and non-audio-response failures are handled with safe messages rather than raw provider bodies.

## One speech request

With `OPENROUTER_API_KEY` set:

```powershell
python openrouter_example.py --speech --model elevenlabs/eleven-v3 --voice adam --prompt 'היום ניפגש בשעה שלוש ליד תחנת הרכבת בירושלים.' --format mp3 --output openrouter_outputs/example.mp3
```

The output path must be new. Audio is saved alongside JSON provenance containing model, voice, exact text, generation ID, content type, request options and SHA-256. Credentials are never included. The speech endpoint takes TTS model slugs; text-only models such as Claude Opus are not speech engines.

## Completed pilot

`synthetic_pilot/review.html` is an offline Hebrew listening workbench with filters, audio playback, summaries, and separate human ratings for perceived presentation, casting-age group, Hebrew text correctness and intelligibility. Ratings remain in the browser until exported as JSON. They do not overwrite provider labels or automatic estimates.

`synthetic_eda.ipynb` contains the executed offline EDA; `SYNTHETIC_EDA_PROGRESS.md` explains findings and limitations. Original audio, 16 kHz WAVs, frozen public catalogs, provider voice metadata, generation attempts, blind audio-model reviews and per-request billing records are under `synthetic_pilot/`.

The first completed pilot generated 284 short recordings across 10 model variants and 116 nominal provider/voice groups, using one fixed Hebrew sentence. A blind transcription-based check screened all samples for Hebrew/text agreement. This is automated screening, not human verification. The audio-model age estimator returned one category for all recordings and was rejected as uninformative; use supplied source labels where available and human listening instead. Presentation estimates also have cross-version conflicts and are not demographic truth.

## Reproduction

Existing samples are not automatically generated again, and failed attempts are not automatically retried. Public catalog preparation does not require a key. Generation and audio review are paid API operations; the notebook, reference extraction and workbench build are offline.

```powershell
python -m pip install -r requirements-synthetic.txt
python synthetic_pilot.py prepare
python synthetic_pilot.py generate --hidden-key
python synthetic_review.py --hidden-key
python synthetic_billing.py --hidden-key
python synthetic_reference.py
python build_synthetic_notebook.py
python execute_synthetic_eda.py
python build_synthetic_review.py
python verify_synthetic_pilot.py
```

`--hidden-key` reads through a no-echo prompt and places the key only in the running process environment. No key file is created. The existing frozen plan is reused, so new catalog releases do not silently change a resumed run. No expansion toward 20 hours is performed by these commands.

The project budget is $20; local guards normally stop new batches before $18. OpenRouter key-usage telemetry can be delayed or nonmonotonic, so per-request reported costs are preferred for reconciliation. A strict account-side spending cap should also be set on the key for future large runs; the provided key's account-side limit was not changed by this integration.

## Verification

```powershell
python -m unittest test_openrouter -v
python verify_synthetic_pilot.py
```

Offline API-boundary tests use the real SDK with mocked HTTP responses and no paid requests. Real pilot generation has separately exercised the speech endpoint end to end. Verification reconciles plan, generation, review, hashes, costs, notebook execution and local audio links.
