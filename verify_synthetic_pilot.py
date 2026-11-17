"""Validate generated audio, language screening, billing, notebook and listening links."""
from pathlib import Path
import hashlib
import json
import re
import wave

import nbformat
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'synthetic_pilot'; ANALYSIS=OUT/'analysis'
plan=pd.read_csv(OUT/'planned_samples.csv',keep_default_na=False)
generated=pd.read_csv(OUT/'manifest.csv',keep_default_na=False)
review=pd.read_csv(OUT/'automated_review.csv',keep_default_na=False)
assert len(plan)==len(generated)==len(review)==284
assert plan.sample_id.is_unique and generated.sample_id.is_unique and review.sample_id.is_unique
assert set(plan.sample_id)==set(generated.sample_id)==set(review.sample_id)
assert generated.generation_status.eq('decoded').all()
assert plan.text.nunique()==1 and plan.requested_language.eq('he').all()
assert review.language.eq('Hebrew').all() and review.language_qc_pass.all()
assert plan.model.nunique()==10 and plan.voice_identity_group.nunique()==116
assert not review.human_verified.any()
for r in generated.itertuples():
    original=OUT/r.original_path; wav=OUT/r.wav_path
    assert hashlib.sha256(original.read_bytes()).hexdigest()==r.original_sha256
    assert hashlib.sha256(wav.read_bytes()).hexdigest()==r.wav_sha256
    with wave.open(str(wav),'rb') as f:
        assert f.getframerate()==16000 and f.getnchannels()==1
        assert abs(f.getnframes()/16000-r.duration_s)<.001
notebook=nbformat.read(ROOT/'synthetic_eda.ipynb',as_version=4); nbformat.validate(notebook)
code=[c for c in notebook.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not [o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
metrics=json.loads((ANALYSIS/'key_metrics.json').read_text(encoding='utf-8'))
assert metrics['age_estimator_uninformative'] is True
assert metrics['billing']['billing_complete'] is True
assert metrics['billing']['reported_request_cost_sum_usd']<20
page=(OUT/'review.html').read_text(encoding='utf-8')
match=re.search('<script id="data" type="application/json">(.*?)</script>',page,re.S)
data=json.loads(match.group(1)); assert len(data['samples'])==284
assert all((OUT/s['wav_path']).exists() for s in data['samples'])
# Match actual credential prefixes only; never print file contents or secret values.
secret_files=[]
credential_prefix=b'sk-or-'+b'v1-'
for file in list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))+list(ROOT.glob('*.ipynb'))+list(OUT.rglob('*.json'))+list(OUT.rglob('*.jsonl'))+list(OUT.rglob('*.html'))+list(OUT.rglob('*.csv')):
    if credential_prefix in file.read_bytes(): secret_files.append(str(file.relative_to(ROOT)))
assert not secret_files, 'Credential-like material found in project artifacts; filenames suppressed.'
result={'status':'passed','samples':284,'models':10,'nominal_voice_groups':116,'automated_hebrew_checks_passed':284,
        'executed_notebook_code_cells':len(code),'notebook_errors':0,'csv_tables':len(list(ANALYSIS.glob('*.csv'))),
        'figures':len(list((ANALYSIS/'figures').glob('*.png'))),'reported_cost_usd':metrics['billing']['reported_request_cost_sum_usd'],
        'age_estimates_rejected_as_uninformative':True,'human_demographic_verification':'pending','credential_scan':'passed'}
(ANALYSIS/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
