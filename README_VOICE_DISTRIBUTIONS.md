# Local ECAPA + acoustic distribution analysis

The inference environment is isolated in `.venv-ecapa` and inherits existing scientific Python packages. The global Python environment was not changed. Its existing Torch/Torchaudio versions were incompatible, so Torch 2.8.0 CPU was installed locally to match Torchaudio 2.8.0 CPU. SpeechBrain 1.1.2 was installed locally. No torchvision operation is used; inherited torchvision expects a different Torch and is outside this workflow.

## Reproduce

```powershell
python -m venv --system-site-packages .venv-ecapa
.\.venv-ecapa\Scripts\python.exe -m pip install torch==2.8.0+cpu torchaudio==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv-ecapa\Scripts\python.exe -m pip install speechbrain==1.1.2
python prepare_voice_distribution.py
.\.venv-ecapa\Scripts\python.exe ecapa_embeddings.py --download
.\.venv-ecapa\Scripts\python.exe ecapa_embeddings.py
python build_voice_distribution_notebook.py --execute
python verify_voice_distributions.py
```

Downloads need network access; Jupyter needs access to the installed user kernel configuration. All analysis after downloading/extracting vectors runs offline. No OpenRouter key or paid API is used. Recordings stay local. Model weights and new extracted WAVs are ignored by Git.

`voice_distribution/model_provenance.json` pins the actual checkpoint revision and hashes. Model loading uses official SpeechBrain feature/ECAPA layers, strict pretrained state loading and L2-normalized 192-D output, without speaker-class predictions or remote custom code. The cached `embeddings.npz` binds each vector to its sample ID, view and WAV hash. Preserve the model revision with the cache; delete/rebuild the cache if intentionally changing the checkpoint or extraction code.

The notebook compares two boundary views, same/different identity distributions, cross-session real-speaker sanity checks, full-space kernel distances, nearest-neighbour coverage, shared PCA/t-SNE and an independent acoustic-feature view. It trains no detector and infers no age/sex labels.

Read `VOICE_DISTRIBUTIONS_REPORT_HE.md` for findings, limits and recommendations. This selected identity sample and the one-sentence TTS pilot are not population-representative Hebrew speech.
