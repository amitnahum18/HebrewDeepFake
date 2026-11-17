"""Prepare an allowlisted analysis snapshot in an existing separate Git clone.

Does not commit/push. Credentials/account state, standalone audio and weights
are excluded. Explicitly requested listening excerpts are embedded in one HTML.
"""
from pathlib import Path
import hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parent; TARGET=ROOT/'.repo_publish'
if not (TARGET/'.git').is_dir():raise RuntimeError('Expected pre-existing isolated Git clone')
if TARGET.resolve().parent!=ROOT.resolve():raise RuntimeError('Publication target outside project')
files=[];excluded=[]
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ['.py','.md','.ipynb'] or p.name.startswith('requirements') and p.suffix=='.txt' or p.name in {'.gitignore','.gitattributes'}):
        if p.name in {'prepare_drive_audio_bundle.py','prepare_remaining_drive_uploads.py','diagnose_voice_pair_playback.py'}:continue
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

# The user specifically requested the playable six-pair-per-group update in Git.
# One self-contained page embeds only selected excerpts; do not duplicate it or
# upload the entire WAV folder/full pairwise matrix/temporary extracted JS.
pair_root=ROOT/'similar_voice_pairs'
pair_names={'listen_portable.html','README_HE.md','selected_pairs.csv','controlled_candidate_pairs.csv','uncontrolled_selection_reference.csv','audio_assets.json','gallery_data.json','key_metrics.json','provenance.json','verification.json','browser_verification.json','browser_desktop.png','browser_mobile.png','human_pair_feedback.json','matching_audit.csv','MATCHING_AUDIT_HE.md','selection_before_metadata_filter.csv'}
for p in pair_root.rglob('*'):
    if not p.is_file():continue
    rel=p.relative_to(pair_root)
    if p.name in pair_names and len(rel.parts)==1 or rel.parts[0] in {'metadata','figures'} and p.suffix.lower() in {'.json','.png'}:files.append(p)

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
pair_metrics=json.loads((pair_root/'key_metrics.json').read_text())
snapshot={'date':'2026-11-17','source':'local HebrewDeepFake analysis workspace','files':manifest,'file_count':len(manifest),'total_bytes':sum(r['bytes'] for r in manifest),'excluded_real_exports':excluded,'excluded_categories':['standalone audio files (selected listening audio is explicitly embedded in the HTML)','downloaded model weights','local environments/dependencies','API keys and account/auth responses','local budget guard','temporary/browser state','pending Drive upload bundles/logs'],'included_embedded_audio':{'path':'similar_voice_pairs/listen_portable.html','original_recordings':pair_metrics['distinct_selected_original_recordings'],'selection_cards':24,'pairs_per_group':6,'scope':'raw and listening-level representations of selected real/synthetic excerpts; explicitly requested playable examples'},'credential_pattern_scan':'passed; no candidate matches; secret values never printed','note':'Historical run paths/hashes retained. Rebuild audio-dependent exports for a new machine. LICENSE belongs to existing remote and is unchanged.'}
(TARGET/'repo_snapshot_manifest.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps({'copied_files':snapshot['file_count'],'megabytes':round(snapshot['total_bytes']/1e6,2),'excluded_real_exports':len(excluded),'credential_scan':'passed','target':str(TARGET)},indent=2))
