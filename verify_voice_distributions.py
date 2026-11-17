"""Verify actual vectors, source bindings, grouping, outputs and notebook."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'voice_distribution'
checks={}
def check(name,condition):
    checks[name]=bool(condition)
    assert condition,name
def table(name):return pd.read_csv(OUT/(name+'.csv'))
samples=table('samples'); data=np.load(OUT/'embeddings.npz',allow_pickle=False)
check('515_unique_samples',len(samples)==515 and samples.sample_id.is_unique)
check('231_real_284_synthetic',samples.groupby('label').size().to_dict()=={'real':231,'synthetic':284})
check('80_real_116_synthetic_groups',samples.groupby('label').group_id.nunique().to_dict()=={'real':80,'synthetic':116})
check('1030_finite_192d_vectors',data['embeddings'].shape==(1030,192) and np.isfinite(data['embeddings']).all())
check('unit_normalized_vectors',np.allclose(np.linalg.norm(data['embeddings'],axis=1),1,atol=1e-5))
binding=samples.set_index('sample_id').wav_sha256.to_dict()
check('every_vector_bound_to_input_hash',all(binding[k]==h for k,h in zip(data['sample_ids'],data['wav_hashes'])))
check('two_views_each_sample',pd.DataFrame({'sample_id':data['sample_ids'],'view':data['views']}).groupby('sample_id').view.agg(set).map(lambda s:s=={'original','edge_trimmed'}).all())
check('all515_wavs_unchanged',all(hashlib.sha256(Path(r.absolute_wav_path).read_bytes()).hexdigest()==r.wav_sha256 for r in samples.itertuples()))
coverage=table('real_speaker_coverage')
check('85_source_IDs_accounted',len(coverage)==85 and coverage.user_id.is_unique and coverage.extracted.sum()==231 and coverage.extracted.eq(0).sum()==5)
check('maximum_three_real_clips',samples.loc[samples.label.eq('real')].groupby('group_id').size().le(3).all())
real=table('real_extraction_manifest')
check('231_successful_real_cuts_no_inserted_pts_gaps',len(real)==231 and real.status.eq('extracted_not_human_verified').all() and real.decoded_sample_coverage_fraction.ge(.98).all() and real.extraction_clock.eq('concatenated_decoded_samples_no_PTS_gap_insertion').all())
for name,filekey in [('model_provenance','files')]:
    info=json.loads((OUT/(name+'.json')).read_text())
    for filename,expected in info[filekey].items():check('model_hash:'+filename,hashlib.sha256((OUT/'model'/filename).read_bytes()).hexdigest()==expected)
provenance=json.loads((OUT/'analysis_provenance.json').read_text())
for path,digest in provenance['inputs'].items():check('input_hash:'+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
for view in ['original','edge_trimmed']:
    centroid=np.load(OUT/('centroids_'+view+'.npz'),allow_pickle=False)
    check('196_unit_centroids_'+view,centroid['embeddings'].shape==(196,192) and np.allclose(np.linalg.norm(centroid['embeddings'],axis=1),1,atol=1e-5))
pairs=table('within_identity_pairs'); sf=samples.set_index('sample_id')
check('within_pairs_really_share_identity',all(sf.loc[r.sample_a,'group_id']==sf.loc[r.sample_b,'group_id']==r.group_id and r.sample_a!=r.sample_b for r in pairs.itertuples()))
rp=pairs.loc[pairs.label.eq('real')]
check('cross_session_flags_accurate',all((sf.loc[r.sample_a,'session_id']!=sf.loc[r.sample_b,'session_id'])==r.cross_session for r in rp.itertuples()))
near=table('cross_class_nearest_neighbours')
check('392_cross_class_neighbours',len(near)==392 and near.cosine_distance.between(-1e-5,2+1e-5).all())
kernel=table('kernel_distribution_distances')
check('36_nonnegative_kernel_results',len(kernel)==36 and kernel.biased_weighted_mmd_squared.ge(-1e-6).all())
stability=table('equal_size_family_sensitivity')
check('equal20_family_sensitivity',len(stability)==8 and stability.groups_each.eq(20).all() and stability.subsamples.eq(200).all())
single=table('one_clip_per_identity_sensitivity')
check('50_single_clip_repeats_per_view',len(single)==100 and single.groupby('view').size().eq(50).all() and single.mmd_squared_one_clip_per_identity.ge(-1e-6).all())
nb=nbformat.read(ROOT/'voice_distributions.ipynb',as_version=4);nbformat.validate(nb)
code=[c for c in nb.cells if c.cell_type=='code']
check('all_eight_notebook_code_cells_executed',len(code)==8 and all(c.execution_count is not None for c in code))
check('zero_notebook_errors',not any(o.output_type=='error' for c in code for o in c.outputs))
check('eight_nonempty_figures',len(list((OUT/'figures').glob('*.png')))==8 and all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.png')))
result={'passed':all(checks.values()),'checks':checks,'samples':515,'vectors':1030,'dimensions':192,'figures':8,'executed_notebook_code_cells':len(code),'paid_api_calls':0,'human_labels_added':0}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
