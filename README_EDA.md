# Hebrew crowd-recital EDA

Start with `EDA_PLAN.md` for the analysis design, `dataset.ipynb` for the executed tables, plots, and listening examples, and `EDA_PROGRESS.md` for findings and limitations after execution.

## Input
The requested acquisition command is:

```powershell
hf download ivrit-ai/crowd-recital --repo-type=dataset
```

The analysis uses the original session repository, pinned to revision `34d93a876ae075a1a7904e7ec2c54d2af1c9d4d7`. The default local path is the Hugging Face snapshot cache under the current Windows user's home directory. For another location, set `CROWD_RECITAL_PATH` to the directory containing `manifest.csv` and session folders. Existing dataset access is needed only to acquire the files; credentials are not saved in the analysis.

## Run

Use the package versions recorded in `requirements-eda.txt` and the Python version in `eda_outputs/run_provenance.json`. Open the notebook from this project directory and run all cells, or execute:

```powershell
python execute_eda.py
python verify_eda.py
```

`build_eda_notebook.py` is the source used to author the notebook. Running it replaces the notebook and clears its execution outputs; it is unnecessary for ordinary re-execution. Keep `audio_eda.py` beside the notebook: it provides the acoustic measurements in two Windows-compatible worker processes.

## Outputs

`eda_outputs/` contains CSV tables, figure PNGs, machine-readable key metrics and run provenance. Important tables include recording and speaker summaries, raw-aligned segments and words, demographic balance, integrity issues, all acoustic features, explicit outlier rules, per-recording screening flags, human-review candidates, a seeded listening baseline, and speaker/text connected components. All source files are SHA-256 hashed in `file_inventory.csv`.

Audio originals are unchanged. Energy measures and alignment confidence are screening signals, not human-confirmed noise labels or measured transcript accuracy. Review candidates are not an exclusion list. Demographic labels are not inferred from voice; biological sex is not gender identity, and the age proxy is not age at recording. The report describes the checks that remain for humans.
