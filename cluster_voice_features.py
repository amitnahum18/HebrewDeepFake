"""Unsupervised, identity-balanced acoustic / ECAPA / hybrid clustering."""
from pathlib import Path
import hashlib,json,importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import KMeans,AgglomerativeClustering,HDBSCAN
from sklearn.metrics import silhouette_score,silhouette_samples,adjusted_rand_score,normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent; BASE=ROOT/'voice_distribution'; OUT=ROOT/'clustering_outputs'; FIG=OUT/'figures'
SEED=20261117
CORE=['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction','energy_dynamic_range_db','spectral_centroid_hz','highband_energy_fraction','spectral_flatness','zero_crossing_fraction']
ALL=['duration_s','rms_dbfs','peak','dc_offset','near_fullscale_fraction','quiet_fraction_50dbfs','quiet_fraction_45dbfs','quiet_fraction_40dbfs','leading_quiet_s','trailing_quiet_s','edge_quiet_fraction','energy_dynamic_range_db','zero_crossing_fraction','spectral_centroid_hz','highband_energy_fraction','spectral_flatness']
SPACES=['acoustic_core','acoustic_all16','ecapa','hybrid']
NAMES={'acoustic_core':'Acoustic core features','acoustic_all16':'All acoustic descriptors','ecapa':'ECAPA embeddings','hybrid':'Acoustic core + ECAPA'}
def export(df,name):
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig');return df
def finish(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=155,bbox_inches='tight');plt.close(fig)
def relabel(y):
    order=sorted(set(y)-{-1},key=lambda k:(-int((y==k).sum()),int(k)))
    mapping={int(k):i for i,k in enumerate(order)};mapping[-1]=-1
    return np.array([mapping[int(k)] for k in y])
def transform(df,features,fit=None,cap=True):
    a=df[features].astype(float).to_numpy().copy(); log=[]
    for i,key in enumerate(features):
        if key in ['spectral_centroid_hz','energy_dynamic_range_db']:
            a[:,i]=np.log1p(np.maximum(a[:,i],0));log.append(key+':log1p')
        elif key in ['highband_energy_fraction','spectral_flatness']:
            a[:,i]=np.log10(np.maximum(a[:,i],1e-8));log.append(key+':log10_floor_1e-8')
    missing=~np.isfinite(a)
    if fit is None:
        med=np.nanmedian(np.where(missing,np.nan,a),axis=0)
        med=np.nan_to_num(med)
        a=np.where(missing,med,a)
        keep=np.ptp(a,axis=0)>1e-12
        scaler=RobustScaler().fit(a[:,keep])
        fit={'medians':med,'keep':keep,'scaler':scaler,'features':features,'log':log}
    else:a=np.where(missing,fit['medians'],a)
    z=fit['scaler'].transform(a[:,fit['keep']]);clipped=int((np.abs(z)>5).sum())
    return (np.clip(z,-5,5) if cap else z),fit,clipped,int(missing.sum())
def km(x,k,seed=SEED,n_init=30):return KMeans(n_clusters=int(k),n_init=n_init,random_state=seed).fit(x)
def safe_silhouette(x,y):
    unique=set(y)-{-1};mask=y!=-1
    return float(silhouette_score(x[mask],y[mask])) if 1<len(unique)<int(mask.sum()) else np.nan
def native_metadata():
    s=pd.read_csv(BASE/'samples.csv',keep_default_na=False)
    meta=s.groupby(['label','group_id'],sort=True).agg(family=('family','first'),clips=('sample_id','size'),sex_label=('sex_label','first'),age_proxy_bin=('age_proxy_bin','first')).reset_index()
    return s,meta
def projection_plots(meta,x,y,space,view):
    pca=PCA().fit(x);p=pca.transform(x)[:,:2]
    tsne=TSNE(n_components=2,random_state=SEED,perplexity=30,init='pca',learning_rate='auto',max_iter=1000).fit_transform(x)
    coord=meta.copy();coord['cluster']=y;coord['pca1']=p[:,0];coord['pca2']=p[:,1];coord['tsne1']=tsne[:,0];coord['tsne2']=tsne[:,1]
    export(coord,space+'_'+view+'_coordinates')
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for row,xy in enumerate([p,tsne]):
        for col,key in enumerate(['cluster','label','family']):
            ax=axes[row,col]
            for value in sorted(coord[key].unique(),key=str):
                mask=coord[key].eq(value).to_numpy(); ax.scatter(xy[mask,0],xy[mask,1],s=24,alpha=.8,label=str(value))
            ax.set_title(('PCA' if row==0 else 't-SNE')+' / '+key);ax.legend(fontsize=8)
            ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)' if row==0 else 't-SNE coordinate 1')
            ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)' if row==0 else 't-SNE coordinate 2')
    fig.suptitle(NAMES[space]+' | '+view+' | clusters fitted in full space; projections for display',fontsize=14)
    finish(fig,space+'_'+view+'_maps')
    variance=pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),'variance_fraction':pca.explained_variance_ratio_,'cumulative_variance':np.cumsum(pca.explained_variance_ratio_)})
    export(variance,space+'_'+view+'_pca_variance')
    return pca

def run():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    samples,meta=native_metadata();acoustics=pd.read_csv(BASE/'sample_acoustic_features.csv')
    assert len(samples)==515 and len(meta)==196 and samples.sample_id.is_unique
    embedding=np.load(BASE/'embeddings.npz',allow_pickle=False)
    selections=[];sweeps=[];assignments=[];methods=[];stabilities=[];sensitivities=[];profiles=[];representatives=[];clip_assignments=[];clip_consistency=[];transforms=[];partitions={};matrices={};saved_models={};pcas={};method_partitions={}
    for view in ['original','edge_trimmed']:
        prefix='' if view=='original' else 'edge_trimmed_'
        group=acoustics.groupby(['label','group_id'],sort=True)[[prefix+k for k in ALL]].median().reset_index().rename(columns={prefix+k:k for k in ALL})
        assert list(zip(group.label,group.group_id))==list(zip(meta.label,meta.group_id))
        centers=np.load(BASE/('centroids_'+view+'.npz'),allow_pickle=False)
        assert np.array_equal(centers['group_ids'],meta.group_id.to_numpy())
        e=centers['embeddings']
        core,core_fit,core_clips,core_missing=transform(group,CORE)
        full,full_fit,full_clips,full_missing=transform(group,ALL)
        block_scale=float(np.sqrt(np.mean(np.sum(core*core,axis=1))))
        spaces={'acoustic_core':core,'acoustic_all16':full,'ecapa':e,'hybrid':np.column_stack([core/block_scale,e])/np.sqrt(2)}
        for name,fit,clipped,missing in [('acoustic_core',core_fit,core_clips,core_missing),('acoustic_all16',full_fit,full_clips,full_missing)]:
            for i,key in enumerate(fit['features']):
                kept=bool(fit['keep'][i]);j=int(fit['keep'][:i].sum())
                transforms.append(dict(view=view,space=name,feature=key,retained=kept,imputation_transformed_median=fit['medians'][i],robust_center=fit['scaler'].center_[j] if kept else np.nan,robust_scale=fit['scaler'].scale_[j] if kept else np.nan,transformation=next((t for t in fit['log'] if t.startswith(key+':')),'identity'),total_clipped_cells=clipped,total_missing_cells=missing,hybrid_acoustic_global_rms_norm=block_scale))
        for space,x in spaces.items():
            print('Clustering',view,space,x.shape,flush=True)
            fits={}
            for k in range(2,11):
                model=km(x,k);y=relabel(model.labels_);counts=np.bincount(y)
                score=float(silhouette_score(x,y));eligible=bool(counts.min()>=8 and counts.max()<=.9*len(x))
                sweeps.append(dict(view=view,space=space,k=k,silhouette=score,inertia=float(model.inertia_),smallest_cluster=int(counts.min()),largest_cluster=int(counts.max()),eligible=eligible,dimensions=x.shape[1]))
                fits[k]=(model,y,score,eligible)
            admissible=[k for k,v in fits.items() if v[3]]
            supported=bool(admissible)
            selected=max(admissible or list(fits),key=lambda k:(fits[k][2],-k))
            model,y,score,_=fits[selected]
            partitions[(view,space)]=y;matrices[(view,space)]=x;saved_models[(view,space)]=model
            selections.append(dict(view=view,space=space,selected_k=selected,silhouette=score,dimensions=x.shape[1],support_constraints_met=supported,selection='maximum eligible full-space silhouette' if supported else 'unsupported fallback: best silhouette with support warning'))
            sil=silhouette_samples(x,y)
            for i,r in meta.iterrows():assignments.append(dict(view=view,space=space,**r.to_dict(),cluster=int(y[i]),silhouette=float(sil[i])))
            agg=relabel(AgglomerativeClustering(n_clusters=selected,linkage='average',metric='euclidean').fit_predict(x))
            density=HDBSCAN(min_cluster_size=8,min_samples=5,metric='euclidean',allow_single_cluster=False).fit_predict(x)
            density=relabel(density)
            for method,lab in [('kmeans',y),('average_linkage',agg),('hdbscan',density)]:
                mask=lab!=-1;common=int(mask.sum())
                methods.append(dict(view=view,space=space,method=method,clusters=len(set(lab)-{-1}),noise_fraction=float((lab==-1).mean()),assigned=common,silhouette_assigned=safe_silhouette(x,lab),ari_vs_kmeans_on_assigned=float(adjusted_rand_score(y[mask],lab[mask])) if common>1 else np.nan))
                method_partitions[(view,space,method)]=lab
            rng=np.random.default_rng(SEED)
            for repeat in range(10):
                alt=km(x,selected,SEED+repeat+1,n_init=20).labels_
                stabilities.append(dict(view=view,space=space,kind='initialization',repeat=repeat+1,shared_identities=len(x),ari=float(adjusted_rand_score(y,alt))))
            for repeat in range(20):
                ids=np.sort(rng.choice(len(x),round(.8*len(x)),replace=False))
                alt=km(x[ids],selected,SEED+repeat,n_init=20).labels_
                stabilities.append(dict(view=view,space=space,kind='80pct_subsample_fixed_transform',repeat=repeat+1,shared_identities=len(ids),ari=float(adjusted_rand_score(y[ids],alt))))
            pca90=PCA(n_components=.9,svd_solver='full').fit(x);z=pca90.transform(x)
            alt=km(z,selected).labels_
            sensitivities.append(dict(view=view,space=space,kind='PCA90_same_k',dimensions=z.shape[1],ari=float(adjusted_rand_score(y,alt))))
            if space!='ecapa':
                uncap_core,_,_,_=transform(group,CORE,core_fit,cap=False)
                uncap=transform(group,ALL,full_fit,cap=False)[0] if space=='acoustic_all16' else (uncap_core if space=='acoustic_core' else np.column_stack([uncap_core/block_scale,e])/np.sqrt(2))
                alt=km(uncap,selected).labels_
                sensitivities.append(dict(view=view,space=space,kind='no_cap_same_k',dimensions=uncap.shape[1],ari=float(adjusted_rand_score(y,alt))))
            for cluster in sorted(set(y)):
                ids=np.flatnonzero(y==cluster);center=x[ids].mean(axis=0);dist=np.linalg.norm(x[ids]-center,axis=1)
                for role,ordered in [('representative',np.argsort(dist)[:3]),('distant_member',np.argsort(dist)[-2:])]:
                    for j in ordered:
                        idx=ids[j];representatives.append(dict(view=view,space=space,cluster=int(cluster),role=role,distance_to_kmeans_mean=float(dist[j]),**meta.iloc[idx].to_dict()))
                for key in ALL:profiles.append(dict(view=view,space=space,cluster=int(cluster),feature=key,identity_median=float(group.iloc[ids][key].median()),identities=len(ids)))
            if view=='original':
                pca=projection_plots(meta,x,y,space,view);pcas[space]=pca
                if space.startswith('acoustic'):
                    keys=np.array(CORE if space=='acoustic_core' else ALL)[core_fit['keep'] if space=='acoustic_core' else full_fit['keep']]
                    export(pd.DataFrame(pca.components_[:2].T,index=keys,columns=['PC1_coefficient','PC2_coefficient']).rename_axis('transformed_feature').reset_index(),space+'_pca_axis_coefficients')
                # Detailed cluster geometry and post-hoc label composition.
                fig,axes=plt.subplots(2,2,figsize=(14,10));offset=0
                for c in sorted(set(y)):
                    values=np.sort(sil[y==c]);axes[0,0].fill_betweenx(np.arange(offset,offset+len(values)),0,values,alpha=.7,label='C'+str(c));offset+=len(values)+5
                axes[0,0].axvline(score,color='black',ls='--');axes[0,0].set(title='Full-space silhouette by cluster',xlabel='Silhouette (not accuracy)',ylabel='Identities');axes[0,0].legend(fontsize=8)
                pd.Series(y).value_counts().sort_index().plot.bar(ax=axes[0,1],color='teal');axes[0,1].set(title='Cluster sizes',xlabel='Cluster',ylabel='Identities')
                comp=meta.assign(cluster=y)
                pd.crosstab(comp.cluster,comp.label).plot.bar(stacked=True,ax=axes[1,0]);axes[1,0].set(title='Real/synthetic: interpretation only',ylabel='Identities')
                pd.crosstab(comp.cluster,comp.family).plot.bar(stacked=True,ax=axes[1,1]);axes[1,1].set(title='Source/family composition',ylabel='Identities');axes[1,1].legend(fontsize=8)
                fig.suptitle(NAMES[space]+' | selected k='+str(selected));finish(fig,space+'_cluster_diagnostics')
                prof=group[CORE].copy();scaled=RobustScaler().fit_transform(prof)
                prof=pd.DataFrame(scaled,columns=CORE).assign(cluster=y).groupby('cluster').median()
                fig,ax=plt.subplots(figsize=(14,max(3,selected*.6+2)))
                im=ax.imshow(prof.to_numpy(),aspect='auto',cmap='RdBu_r',vmin=-3,vmax=3)
                ax.set_xticks(np.arange(len(CORE)),CORE,rotation=35,ha='right');ax.set_yticks(np.arange(selected),['C'+str(c) for c in prof.index]);ax.set_title(NAMES[space]+' | post-hoc acoustic profile (pooled robust units, color capped +/-3)')
                for row in range(selected):
                    for col in range(len(CORE)):ax.text(col,row,f'{prof.iloc[row,col]:.1f}',ha='center',va='center',fontsize=8)
                fig.colorbar(im,ax=ax,label='Median difference / IQR');finish(fig,space+'_acoustic_cluster_profiles')
                fig,ax=plt.subplots(figsize=(14,5));dendrogram(linkage(x,method='average',metric='euclidean'),truncate_mode='lastp',p=24,show_leaf_counts=True,ax=ax)
                ax.set(title=NAMES[space]+' | average-linkage dendrogram, last 24 branches',xlabel='Leaf/merged branch (parentheses = identities)',ylabel='Euclidean linkage distance');finish(fig,space+'_hierarchy')
                # Clip-level sensitivity: reuse group-fitted transformations.
                clip=acoustics.copy()
                cp=clip.rename(columns={k:'unused_'+k for k in ALL}) if prefix else clip
                if prefix:cp=cp.rename(columns={prefix+k:k for k in ALL})
                ci,_,_,_=transform(cp,CORE,core_fit)
                cfull,_,_,_=transform(cp,ALL,full_fit)
                emb_index={str(k):j for j,(k,v) in enumerate(zip(embedding['sample_ids'],embedding['views'])) if v==view}
                ce=np.array([embedding['embeddings'][emb_index[k]] for k in clip.sample_id])
                cx={'acoustic_core':ci,'acoustic_all16':cfull,'ecapa':ce,'hybrid':np.column_stack([ci/block_scale,ce])/np.sqrt(2)}[space]
                cy=relabel(km(cx,selected).labels_)
                for idx,r in clip.iterrows():clip_assignments.append(dict(space=space,sample_id=r.sample_id,label=r.label,group_id=r.group_id,family=r.family,cluster=int(cy[idx])))
                for (label,group_id),g in clip.assign(cluster=cy).groupby(['label','group_id']):
                    clip_consistency.append(dict(space=space,label=label,group_id=group_id,clips=len(g),distinct_clusters=g.cluster.nunique(),majority_fraction=float(g.cluster.value_counts().max()/len(g))))
    assignment=export(pd.DataFrame(assignments),'identity_assignments')
    selection=export(pd.DataFrame(selections),'selected_models');sweep=export(pd.DataFrame(sweeps),'k_selection')
    export(pd.DataFrame(methods),'method_comparison');export(pd.DataFrame(stabilities),'stability')
    export(pd.DataFrame(sensitivities),'preprocessing_sensitivity');export(pd.DataFrame(profiles),'cluster_native_feature_profiles')
    reps=export(pd.DataFrame(representatives),'cluster_representatives');export(pd.DataFrame(clip_assignments),'clip_assignments_sensitivity');export(pd.DataFrame(clip_consistency),'within_identity_clip_consistency');export(pd.DataFrame(transforms),'acoustic_transform_parameters')
    # Attach real listening references and synthetic files to cluster examples.
    evidence=[]
    for r in reps.to_dict('records'):
        g=samples.loc[samples.group_id.eq(r['group_id'])]
        for s in g.to_dict('records'):
            evidence.append({**r,'sample_id':s['sample_id'],'audio_path':s['absolute_wav_path'],'text':s['text'],'model':s['model']})
    export(pd.DataFrame(evidence),'cluster_representative_audio_links')
    composition=[];associations=[]
    for (view,space),g in assignment.groupby(['view','space']):
        for field in ['label','family']:
            tab=pd.crosstab(g.cluster,g[field])
            for cluster,row in tab.iterrows():
                for value,count in row.items():composition.append(dict(view=view,space=space,field=field,cluster=cluster,value=value,identities=int(count),within_cluster_share=count/max(1,row.sum())))
            associations.append(dict(view=view,space=space,field=field,nmi=float(normalized_mutual_info_score(g[field],g.cluster)),interpretation='posthoc association not accuracy; labels not used for fitting or k'))
    export(pd.DataFrame(composition),'cluster_composition');export(pd.DataFrame(associations),'posthoc_label_associations')
    boundaries=[]
    for space in SPACES:
        a=partitions[('original',space)];b=partitions[('edge_trimmed',space)];k=int(selection.loc[selection.view.eq('original')&selection.space.eq(space),'selected_k'].iloc[0])
        fixed=km(matrices[('edge_trimmed',space)],k).labels_
        boundaries.append(dict(space=space,original_k=k,trimmed_selected_k=int(selection.loc[selection.view.eq('edge_trimmed')&selection.space.eq(space),'selected_k'].iloc[0]),ari_independent_k=float(adjusted_rand_score(a,b)),ari_same_original_k=float(adjusted_rand_score(a,fixed))))
    export(pd.DataFrame(boundaries),'boundary_sensitivity')
    agreement=pd.DataFrame(index=SPACES,columns=SPACES,dtype=float)
    for a in SPACES:
        for b in SPACES:agreement.loc[a,b]=adjusted_rand_score(partitions[('original',a)],partitions[('original',b)])
    export(agreement.rename_axis('space').reset_index(),'between_space_ari')
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for space,ax in zip(SPACES,axes.ravel()):
        for view in ['original','edge_trimmed']:
            g=sweep.loc[sweep.space.eq(space)&sweep.view.eq(view)];ax.plot(g.k,g.silhouette,marker='o',label=view)
            bad=g.loc[~g.eligible];ax.scatter(bad.k,bad.silhouette,marker='x',s=90,color='red')
        ax.set(title=NAMES[space],xlabel='k',ylabel='Full-space Euclidean silhouette');ax.legend(fontsize=8)
    fig.suptitle('Internal k selection; red x = insufficient cluster support; scores not cross-space quality ranks');finish(fig,'k_selection_overview')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    stable=pd.DataFrame(stabilities)
    for ax,kind in zip(axes,['initialization','80pct_subsample_fixed_transform']):
        values=[stable.loc[stable.view.eq('original')&stable.space.eq(s)&stable.kind.eq(kind),'ari'] for s in SPACES]
        ax.boxplot(values,tick_labels=SPACES);ax.tick_params(axis='x',rotation=20);ax.set(title=kind,ylabel='ARI vs fixed reference (1 = same partition)')
    finish(fig,'stability_overview')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    im=axes[0].imshow(agreement.astype(float),vmin=0,vmax=1,cmap='viridis');axes[0].set_xticks(range(4),SPACES,rotation=30,ha='right');axes[0].set_yticks(range(4),SPACES);axes[0].set_title('Agreement between feature spaces (ARI)')
    for i in range(4):
        for j in range(4):axes[0].text(j,i,f'{agreement.iloc[i,j]:.2f}',ha='center',va='center',color='white')
    pd.DataFrame(boundaries).set_index('space')[['ari_independent_k','ari_same_original_k']].plot.bar(ax=axes[1]);axes[1].set(title='Boundary sensitivity',ylabel='Original vs capped-edge-trimmed ARI');axes[1].tick_params(axis='x',rotation=25);axes[1].legend(fontsize=8)
    fig.colorbar(im,ax=axes[0]);finish(fig,'partition_agreement_and_boundaries')
    fig,ax=plt.subplots(figsize=(11,5))
    for space,pca in pcas.items():ax.plot(np.arange(1,len(pca.explained_variance_ratio_)+1),np.cumsum(pca.explained_variance_ratio_),label=space)
    ax.axhline(.9,ls='--',color='grey');ax.set(xlabel='PCA components',ylabel='Cumulative variance fraction',title='PCA is displayed separately; primary clustering uses full feature spaces');ax.legend();finish(fig,'pca_variance_overview')
    all_methods=[]
    for (view,space,method),labels in method_partitions.items():
        for i,label in enumerate(labels):all_methods.append(dict(view=view,space=space,method=method,group_id=meta.iloc[i].group_id,cluster=int(label)))
    export(pd.DataFrame(all_methods),'all_method_assignments')
    np.savez_compressed(OUT/'clustering_matrices.npz',**{v+'__'+s:x for (v,s),x in matrices.items()},group_ids=meta.group_id.to_numpy(dtype=str))
    inputs=[BASE/'samples.csv',BASE/'sample_acoustic_features.csv',BASE/'embeddings.npz',BASE/'centroids_original.npz',BASE/'centroids_edge_trimmed.npz',Path(__file__),ROOT/'CLUSTERING_PLAN.md']
    provenance={'seed':SEED,'runtime_versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','matplotlib']},'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'primary_unit':'one nominal identity representative','identity_order':meta.group_id.tolist(),'k_grid':list(range(2,11)),'selection_uses_class_labels':False,'feature_metadata_excluded':['label','family','model','voice','group_id','text','sex_label','age_proxy_bin'],'k_support_min_size':8,'k_support_max_share':.9,'kmeans_n_init':30,'hdbscan':{'min_cluster_size':8,'min_samples':5},'stability_repeats':{'initialization':10,'80pct_fixed_transform':20},'note':'Selected k maximizes eligible internal silhouette, not true class separation. All methods exploratory; no ground-truth cluster labels.'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    return {'identity_count':196,'clip_count':515,'spaces':SPACES,'boundary_views':2,'selected_models':selections,'figures':len(list(FIG.glob('*.png'))),'api_calls':0}

if __name__=='__main__':
    with threadpool_limits(limits=2):
        result=run()
    (OUT/'key_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
