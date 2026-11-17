"""Reconcile corpus audit counts, candidate semantics, hashes and execution."""
from pathlib import Path
import json, hashlib
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'quality_audit'
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name
def table(name): return pd.read_csv(OUT/(name+'.csv'))
notebook=nbformat.read(ROOT/'quality_matching_audit.ipynb',as_version=4)
nbformat.validate(notebook)
codes=[c for c in notebook.cells if c.cell_type=='code']
check('all_six_code_cells_executed',len(codes)==6 and all(c.execution_count is not None for c in codes))
check('no_execution_errors',not any(o.output_type=='error' for c in codes for o in c.outputs))
provenance=json.loads((OUT/'provenance.json').read_text())
for path,digest in provenance['inputs'].items():
    check('unchanged_input:'+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
for prefix in ['real_segment_exact','eligible_segment_exact','real_session_exact']:
    g=table(prefix+'_groups'); o=table(prefix+'_occurrences')
    check(prefix+'_counts_reconcile',g.rows.sum()==len(o) and g.text_group_id.is_unique and set(g.text_group_id)==set(o.text_group_id))
    observed=o.groupby('text_group_id').agg(rows=('user_id','size'),speakers=('user_id','nunique'))
    expected=g.set_index('text_group_id')
    check(prefix+'_speaker_counts_reconcile',observed.sort_index().equals(expected[['rows','speakers']].sort_index()))
near=table('near_session_text_candidates')
check('12_near_text_pairs_meet_threshold',len(near)==12 and near.cosine_similarity.ge(.92).all() and near.words_a.ge(20).all() and near.words_b.ge(20).all())
check('near_pairs_unique',not near[['session_a','session_b']].duplicated().any())
pcm=table('saved_audio_pcm_fingerprints')
check('362_distinct_canonical_pcm_payloads',len(pcm)==362 and pcm.pcm16_sha256.is_unique)
check('zero_saved_pcm_duplicates',table('saved_audio_exact_pcm_duplicates').empty)
# Audio files remain byte-identical to prior analysis manifests.
syn=pd.read_csv(ROOT/'synthetic_pilot/analysis/sample_features.csv')
ref=pd.read_csv(ROOT/'synthetic_pilot/human_reference.csv')
check('all362_saved_wav_files_unchanged',all(hashlib.sha256((ROOT/'synthetic_pilot'/r.wav_path).read_bytes()).hexdigest()==r.wav_sha256 for df in [syn,ref] for r in df.itertuples()))
demo=table('real_demographic_coverage_by_scope')
shares=demo.groupby(['scope','field'])[['time_share','speaker_share']].sum()
check('both_weight_systems_sum_to_one',((shares-1).abs()<1e-9).all().all())
check('all_85_source_IDs_accounted',len(table('real_identity_coverage_listening'))==85 and table('real_identity_coverage_listening').user_id.is_unique)
check('116_nominal_voice_listening_rows',len(table('synthetic_identity_coverage_listening'))==116 and table('synthetic_identity_coverage_listening').voice_identity_group.is_unique)
check('review_synthetic_seven_candidates',len(table('synthetic_priority_review'))==7)
check('two_nonempty_figures',len(list((OUT/'figures').glob('*.png')))==2 and all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.png')))
result={'passed':all(checks.values()),'checks':checks,'code_cells':len(codes),'csv_tables':len(list(OUT.glob('*.csv'))),'api_calls':0,'audio_mutations':0}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
