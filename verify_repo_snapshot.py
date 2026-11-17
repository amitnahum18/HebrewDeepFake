"""Portable committed-snapshot verification; no audio, model or API required."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'repo_snapshot_manifest.json').read_text(encoding='utf-8'))
failures=[]
for item in manifest['files']:
    path=ROOT/item['path']
    if not path.is_file():failures.append(item['path']+':missing');continue
    content=path.read_bytes()
    if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:failures.append(item['path']+':hash_or_size')
    if path.suffix.lower() in {'.wav','.mp3','.pcm','.mka','.flac','.ckpt','.pth','.pt'}:failures.append(item['path']+':excluded_media')
notebooks=[]
for path in ROOT.glob('*.ipynb'):
    nb=json.loads(path.read_text(encoding='utf-8'));code=[c for c in nb['cells'] if c['cell_type']=='code']
    if not all(c.get('execution_count') is not None for c in code):failures.append(path.name+':unexecuted')
    if any(o.get('output_type')=='error' for c in code for o in c.get('outputs',[])):failures.append(path.name+':cell_error')
    notebooks.append({'name':path.name,'executed_code_cells':len(code)})
assert not failures,failures
print(json.dumps({'passed':True,'manifest_files':len(manifest['files']),'notebooks':notebooks,'standalone_audio_or_model_files':0,'embedded_listening_audio':manifest.get('included_embedded_audio'),'note':'Snapshot integrity only. Full audio-dependent runs require excluded inputs.'},indent=2))
