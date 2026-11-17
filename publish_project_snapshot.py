"""Prepare an allowlisted analysis snapshot in an existing separate Git clone.

Does not commit/push. Credentials/account state, raw audio and weights are never
included. A snapshot manifest preserves original project-relative file hashes.
"""
from pathlib import Path
import hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parent; TARGET=ROOT/'.repo_publish'
if not (TARGET/'.git').is_dir():raise RuntimeError('Expected pre-existing isolated Git clone')
if TARGET.resolve().parent!=ROOT.resolve():raise RuntimeError('Publication target outside project')
files=[];excluded=[]
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ['.py','.md','.ipynb'] or p.name.startswith('requirements') and p.suffix=='.txt' or p.name in {'.gitignore','.gitattributes'}):
        files.append(p)

bulk_real={'words.csv','segments.csv','segment_quality.csv','recording_features.csv','transcript_features.csv','review_candidates.csv'}
for directory in ['eda_outputs','comparison_outputs','quality_audit','voice_distribution','clustering_outputs']:
    for p in (ROOT/directory).rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if any(part in {'model','real_audio','__pycache__','.cache'} for part in rel.parts):continue
        if directory=='eda_outputs' and (p.name in bulk_real or p.stat().st_size>=2_000_000):
            excluded.append({'path':rel.as_posix(),'reason':'large real-source export; regenerate from pinned dataset'});continue
        if p.suffix.lower() in {'.csv','.json','.png','.npz','.md'}:files.append(p)

pilot_files={'planned_samples.csv','manifest.csv','automated_review.csv','billing_summary.json','generation_costs.csv','catalog_snapshot.json','provider_voice_metadata.json','review_attempt_history.jsonl','review_instructions.txt','review_model_snapshot.json','run_config.json','human_reference.csv','review.html'}
for p in (ROOT/'synthetic_pilot').iterdir():
    if p.is_file() and p.name in pilot_files:files.append(p)
for p in (ROOT/'synthetic_pilot/analysis').rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.csv','.json','.png'}:files.append(p)
for p in (ROOT/'voice_review').iterdir():
    if p.is_file() and p.suffix.lower() in {'.csv','.json','.jsonl','.md'} and p.name not in {'current_excerpt.json'}:files.append(p)
for p in (ROOT/'output/pdf').iterdir():
    if p.is_file() and p.name in {'HebrewDeepFake_Synthetic_Data_Generation_Report.pdf','report_verification.json'}:files.append(p)

files=sorted(set(files))
patterns=[re.compile(rb'sk-or-v1-[A-Za-z0-9]{20,}'),re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{30,}'),re.compile(rb'hf_[A-Za-z0-9]{25,}'),re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]
bad=[];manifest=[]
for p in files:
    rel=p.relative_to(ROOT);content=p.read_bytes()
    if p.stat().st_size>25_000_000:raise RuntimeError('Oversized publication candidate: '+str(rel))
    if any(pattern.search(content) for pattern in patterns):bad.append(rel.as_posix())
    manifest.append({'path':rel.as_posix(),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
if bad:raise RuntimeError('Credential-pattern candidates (values not printed): '+', '.join(bad))

for p in files:
    dest=TARGET/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
snapshot={'date':'2026-11-17','source':'local HebrewDeepFake analysis workspace','files':manifest,'file_count':len(manifest),'total_bytes':sum(r['bytes'] for r in manifest),'excluded_real_exports':excluded,'excluded_categories':['raw/derived audio','downloaded model weights','local environments/dependencies','API keys and account/auth responses','local budget guard','temporary/browser state'],'credential_pattern_scan':'passed; no candidate matches; secret values never printed','note':'Historical run paths/hashes retained. Rebuild audio-dependent exports for a new machine. LICENSE belongs to existing remote and is unchanged.'}
(TARGET/'repo_snapshot_manifest.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'copied_files':snapshot['file_count'],'megabytes':round(snapshot['total_bytes']/1e6,2),'excluded_real_exports':len(excluded),'credential_scan':'passed','target':str(TARGET)},indent=2))
