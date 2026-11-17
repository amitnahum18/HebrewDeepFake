"""Reconcile executed notebook outputs and exported corpus-level tables."""
from pathlib import Path
import hashlib, json
import nbformat
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parent; out=root/'eda_outputs'
nb=nbformat.read(root/'dataset.ipynb',as_version=4); nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code), 'Unexecuted notebook cells'
errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
assert not errors, 'Notebook cell errors'
metrics=json.loads((out/'key_metrics.json').read_text(encoding='utf-8'))
recordings=pd.read_csv(out/'recording_features.csv')
speakers=pd.read_csv(out/'speaker_summary.csv')
audio=pd.read_csv(out/'audio_features.csv')
inventory=pd.read_csv(out/'file_inventory.csv')
flags=pd.read_csv(out/'session_flags.csv')
presence=pd.read_csv(out/'companion_file_presence.csv')
sample=pd.read_csv(out/'human_listening_sample.csv')
assert len(inventory)==9597==metrics['inventory_files']
assert len(recordings)==len(audio)==len(flags)==2398==metrics['sessions']
assert recordings.session_id.is_unique and audio.session_id.is_unique
assert set(recordings.session_id)==set(audio.session_id)==set(flags.session_id)
assert len(speakers)==85==metrics['speakers']
assert presence.drop(columns='session_id').all().all()
assert len(sample)==85 and sample.user_id.nunique()==85
assert inventory.sha256.str.fullmatch('[0-9a-f]{64}').all()
assert np.isclose(recordings.session_duration.sum()/3600,metrics['hours'])
assert np.isclose(speakers.hours.sum(),metrics['hours'])
assert int(audio.decode_error.notna().sum())==metrics['decode_failure']
assert len(pd.read_csv(out/'segments.csv'))==metrics['aligned_segments']
assert len(pd.read_csv(out/'words.csv'))==metrics['aligned_words']
assert (out/'timestamp_bounds_validation.csv').is_file()
figures=list((out/'figures').glob('*.png')); assert len(figures)==8
verification={'executed_code_cells':len(code),'notebook_errors':len(errors),'recordings':len(recordings),'speakers':len(speakers),'source_files_hashed':len(inventory),'csv_tables':len(list(out.glob('*.csv'))),'figures':len(figures),'status':'passed',
 'implementation_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['audio_eda.py','build_eda_notebook.py','execute_eda.py','verify_eda.py']}}
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
