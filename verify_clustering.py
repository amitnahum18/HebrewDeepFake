"""Reconcile cached clustering assignments, feature geometry and notebook."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import nbformat
from sklearn.metrics import silhouette_score,adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'clustering_outputs'
checks={}
def check(name,value):
    checks[name]=bool(value);assert value,name
def table(name):return pd.read_csv(OUT/(name+'.csv'))
p=json.loads((OUT/'provenance.json').read_text())
for name,digest in p['input_hashes'].items():check('input_hash:'+name,hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest)
a=table('identity_assignments');selected=table('selected_models');grid=table('k_selection');mat=np.load(OUT/'clustering_matrices.npz',allow_pickle=False)
check('1568_unique_identity_assignments',len(a)==196*8 and not a[['view','space','group_id']].duplicated().any())
check('same196_IDs_every_partition',all(set(g.group_id)==set(mat['group_ids']) for _,g in a.groupby(['view','space'])))
check('label_counts_80real_116synthetic',all(g.groupby('label').size().to_dict()=={'real':80,'synthetic':116} for _,g in a.groupby(['view','space'])))
check('72_k_candidates',len(grid)==8*9 and grid.groupby(['view','space']).size().eq(9).all())
with threadpool_limits(limits=2):
    for r in selected.itertuples():
        x=mat[r.view+'__'+r.space];g=a.loc[a.view.eq(r.view)&a.space.eq(r.space)].set_index('group_id').loc[mat['group_ids']]
        check('finite_matrix:'+r.view+':'+r.space,x.shape==(196,r.dimensions) and np.isfinite(x).all())
        check('clusters_match_k:'+r.view+':'+r.space,g.cluster.nunique()==r.selected_k)
        check('silhouette_recomputed:'+r.view+':'+r.space,np.isclose(silhouette_score(x,g.cluster),r.silhouette,atol=1e-7))
        options=grid.loc[grid.view.eq(r.view)&grid.space.eq(r.space)&grid.eligible]
        check('k_selected_without_class_score:'+r.view+':'+r.space,len(options)>0 and np.isclose(options.silhouette.max(),r.silhouette))
        check('support_constraints:'+r.view+':'+r.space,g.cluster.value_counts().min()>=8 and g.cluster.value_counts().max()<=.9*196)
    for view in ['original','edge_trimmed']:
        e=mat[view+'__ecapa'];h=mat[view+'__hybrid']
        check('unit_ECAPA:'+view,np.allclose(np.linalg.norm(e,axis=1),1,atol=1e-5))
        check('balanced_hybrid_blocks:'+view,np.isclose(np.mean((h[:,:9]**2).sum(axis=1)),.5) and np.isclose(np.mean((h[:,9:]**2).sum(axis=1)),.5))
        check('hybrid_ecapa_binding:'+view,np.allclose(h[:,9:],e/np.sqrt(2)))
        check('capped_acoustic_features:'+view,np.abs(mat[view+'__acoustic_core']).max()<=5 and np.abs(mat[view+'__acoustic_all16']).max()<=5)
methods=table('all_method_assignments')
check('4704_method_assignments',len(methods)==196*8*3 and not methods[['view','space','method','group_id']].duplicated().any())
check('HDBSCAN_hybrid_unassigned_preserved',methods.loc[methods.space.eq('hybrid')&methods.method.eq('hdbscan')].cluster.eq(-1).all())
stable=table('stability');check('240_stability_runs',len(stable)==8*30 and stable.ari.between(-1,1).all())
clips=table('clip_assignments_sensitivity');check('2060_unique_clip_assignments',len(clips)==515*4 and not clips[['space','sample_id']].duplicated().any())
check('posthoc_features_excluded',p['selection_uses_class_labels'] is False and set(['label','family','model','text','sex_label','age_proxy_bin']).issubset(p['feature_metadata_excluded']))
nb=nbformat.read(ROOT/'voice_clustering.ipynb',as_version=4);nbformat.validate(nb);codes=[c for c in nb.cells if c.cell_type=='code']
check('nine_code_cells_executed',len(codes)==9 and all(c.execution_count is not None for c in codes))
check('zero_notebook_errors',not any(o.output_type=='error' for c in codes for o in c.outputs))
check('20_nonempty_figures',len(list((OUT/'figures').glob('*.png')))==20 and all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.png')))
result={'passed':all(checks.values()),'checks':checks,'identity_partitions':8,'identities_per_partition':196,'clip_assignments':2060,'figures':20,'executed_code_cells':len(codes),'api_calls':0}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
