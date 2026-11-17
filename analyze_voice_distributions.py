"""Offline multi-view ECAPA and acoustic distribution analysis from cached vectors."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from comparative_audio import analyze_one
from comparative_analysis import effects

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'voice_distribution'; FIG=OUT/'figures'
SEED=20261117
def export(df,name):
    df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig'); return df
def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def summary(x):
    x=np.asarray(x)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95))) if len(x) else dict(n=0,mean=np.nan,median=np.nan,p05=np.nan,p95=np.nan)
def ecdf(ax,x,label):
    x=np.sort(x); ax.step(x,np.arange(1,len(x)+1)/len(x),label=label)
def finish(fig,name):
    fig.tight_layout(); fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight'); plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True)
    samples=pd.read_csv(OUT/'samples.csv',keep_default_na=False)
    data=np.load(OUT/'embeddings.npz',allow_pickle=False)
    lookup=samples.set_index('sample_id'); metrics={}; distributions=[]; positive_rows=[]; neighbours=[]; mmd=[]; stability=[]; center_rows=[]; family_coverage=[]; single_clip=[]
    rng=np.random.default_rng(SEED)
    for view in ['original','edge_trimmed']:
        select=data['views']==view
        frame=lookup.loc[data['sample_ids'][select]].reset_index()
        vectors=data['embeddings'][select]
        centers=[]; metadata=[]
        for group,g in frame.groupby(['label','group_id'],sort=True):
            ids=g.index.to_numpy()
            centers.append(unit(vectors[ids].mean(axis=0)))
            metadata.append(dict(label=group[0],group_id=group[1],family=g.family.iloc[0],clips=len(g),sex_label=g.sex_label.iloc[0],age_proxy_bin=g.age_proxy_bin.iloc[0]))
        centers=np.array(centers); meta=pd.DataFrame(metadata)
        for k,r in meta.iterrows(): center_rows.append(dict(view=view,**r.to_dict(),vector_index=k))
        np.savez_compressed(OUT/('centroids_'+view+'.npz'),embeddings=centers,group_ids=meta.group_id.to_numpy(dtype=str),labels=meta.label.to_numpy(dtype=str),families=meta.family.to_numpy(dtype=str))
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for label,ax in zip(['real','synthetic'],axes):
            ids=np.flatnonzero(frame.label.eq(label)); part=frame.iloc[ids].reset_index(drop=True); e=vectors[ids]
            i,j=np.triu_indices(len(part),1); cos=(e@e.T)[i,j]
            same=part.group_id.to_numpy()[i]==part.group_id.to_numpy()[j]
            cross=part.session_id.to_numpy()[i]!=part.session_id.to_numpy()[j] if label=='real' else np.ones(len(i),dtype=bool)
            for kind,mask in [('same_group',same),('different_group',~same),('same_group_cross_session',same&cross)]:
                if label=='synthetic' and kind=='same_group_cross_session': continue
                distributions.append(dict(view=view,label=label,comparison=kind,**summary(cos[mask])))
                if np.any(mask): ecdf(ax,cos[mask],kind)
            auc=roc_auc_score(same.astype(int),cos) if same.any() else np.nan
            metrics[view+'_'+label+'_same_identity_pair_auc']=float(auc)
            if label=='real':
                mask=(~same)|(same&cross)
                metrics[view+'_real_cross_session_pair_auc']=float(roc_auc_score(same[mask].astype(int),cos[mask]))
                metrics['real_cross_session_positive_pairs']=int((same&cross).sum())
                metrics['real_cross_session_supported_speakers']=int(part.loc[np.unique(np.r_[i[same&cross],j[same&cross]])].group_id.nunique())
            for pos in np.flatnonzero(same):
                a=part.iloc[i[pos]]; b=part.iloc[j[pos]]
                positive_rows.append(dict(view=view,label=label,group_id=a.group_id,sample_a=a.sample_id,sample_b=b.sample_id,model_a=a.model,model_b=b.model,cross_session=bool(cross[pos]),cosine_similarity=float(cos[pos])))
            ax.set(xlabel='Cosine similarity (original 192-D vectors)',ylabel='Pair empirical CDF',title=label+'; dependent pairs, descriptive only'); ax.legend()
        finish(fig,'identity_consistency_'+view)
        real_mask=meta.label.eq('real').to_numpy(); syn_mask=~real_mask
        a=centers[real_mask]; b=centers[syn_mask]; ra=meta.loc[real_mask].reset_index(drop=True); sb=meta.loc[syn_mask].reset_index(drop=True)
        cross_cos=a@b.T; aa=a@a.T; bb=b@b.T
        np.fill_diagonal(aa,-np.inf); np.fill_diagonal(bb,-np.inf)
        for label,ma,similarity,target in [('real',ra,cross_cos,sb),('synthetic',sb,cross_cos.T,ra)]:
            for k,r in ma.iterrows():
                best=int(similarity[k].argmax())
                neighbours.append(dict(view=view,direction=label+'_to_other_class',group_id=r.group_id,family=r.family,nearest_group=target.iloc[best].group_id,nearest_family=target.iloc[best].family,cosine_similarity=float(similarity[k,best]),cosine_distance=float(1-similarity[k,best]),meaning='candidate_acoustic_neighbour_not_identity_or_duplicate'))
        for label,ma,cosine in [('real',ra,aa),('synthetic',sb,bb)]:
            i,j=np.triu_indices(len(ma),1); values=cosine[i,j]
            distributions.append(dict(view=view,label=label,comparison='different_group_centroids',**summary(values)))
            order=np.argsort(values)[::-1][:20]
            export(pd.DataFrame([dict(group_a=ma.iloc[i[k]].group_id,group_b=ma.iloc[j[k]].group_id,family_a=ma.iloc[i[k]].family,family_b=ma.iloc[j[k]].family,cosine_similarity=values[k],status='listen_candidate_not_duplicate_proof') for k in order]),'closest_'+label+'_groups_'+view)
        fig,ax=plt.subplots(figsize=(10,5))
        ecdf(ax,1-cross_cos.max(axis=1),'real -> nearest synthetic (116 candidates)')
        ecdf(ax,1-aa.max(axis=1),'real -> nearest other real (n-1 candidates)')
        ecdf(ax,1-cross_cos.max(axis=0),'synthetic -> nearest real')
        ax.set(xlabel='Nearest cosine distance',ylabel='Identity empirical CDF',title='Coverage diagnostic; unequal candidate-set sizes, not calibrated matching'); ax.legend()
        finish(fig,'nearest_coverage_'+view)
        distances=cdist(centers,centers,'sqeuclidean'); bandwidth=float(np.median(pdist(centers,'sqeuclidean')))
        # Check whether unequal clips/centroid averaging drives the result.
        group_indices=[g.index.to_numpy() for _,g in frame.groupby(['label','group_id'],sort=True)]
        real_ids=np.flatnonzero(real_mask); synth_ids=np.flatnonzero(syn_mask)
        for repeat in range(50):
            one=vectors[np.array([rng.choice(ids) for ids in group_indices])]
            kernel_one=np.exp(-cdist(one,one,'sqeuclidean')/(2*bandwidth))
            score=kernel_one[np.ix_(real_ids,real_ids)].mean()+kernel_one[np.ix_(synth_ids,synth_ids)].mean()-2*kernel_one[np.ix_(real_ids,synth_ids)].mean()
            single_clip.append(dict(view=view,repeat=repeat+1,mmd_squared_one_clip_per_identity=score,median_between_real_cosine=float(np.median(1-pdist(one[real_mask],'cosine'))),median_between_synthetic_cosine=float(np.median(1-pdist(one[syn_mask],'cosine'))),interpretation='one_clip_per_ID_sampling_sensitivity_not_population_CI'))
        for multiplier in [.5,1,2]:
            kernel=np.exp(-distances/(2*bandwidth*multiplier))
            for target in ['all_identity_equal','all_family_equal']+sorted(sb.family.unique().tolist()):
                ia=np.flatnonzero(real_mask)
                ib=np.flatnonzero(syn_mask & (meta.family.eq(target).to_numpy() if target not in ['all_identity_equal','all_family_equal'] else np.ones(len(meta),dtype=bool)))
                wa=np.full(len(ia),1/len(ia)); wb=np.full(len(ib),1/len(ib))
                if target=='all_family_equal': wb=np.array([1/(4*(meta.iloc[ib].family==meta.iloc[k].family).sum()) for k in ib])
                score=wa@kernel[np.ix_(ia,ia)]@wa+wb@kernel[np.ix_(ib,ib)]@wb-2*wa@kernel[np.ix_(ia,ib)]@wb
                mmd.append(dict(view=view,target=target,bandwidth_multiplier=multiplier,pooled_median_squared_distance=bandwidth,biased_weighted_mmd_squared=score,real_groups=len(ia),synthetic_groups=len(ib)))
        # Same-size family comparison avoids differing family inventory sizes.
        kernel=np.exp(-distances/(2*bandwidth)); size=min(20,len(a),int(sb.groupby('family').size().min()))
        for family in sorted(sb.family.unique()):
            ia=np.flatnonzero(real_mask); ib=np.flatnonzero(meta.family.eq(family).to_numpy())
            scores=[]
            for _ in range(200):
                x=rng.choice(ia,size,replace=False); y=rng.choice(ib,size,replace=False)
                scores.append(kernel[np.ix_(x,x)].mean()+kernel[np.ix_(y,y)].mean()-2*kernel[np.ix_(x,y)].mean())
            stability.append(dict(view=view,family=family,groups_each=size,subsamples=200,median_mmd_squared=np.median(scores),p025=np.quantile(scores,.025),p975=np.quantile(scores,.975),interpretation='observed_subset_composition_range_not_population_CI'))
            candidates=np.flatnonzero(sb.family.eq(family).to_numpy())
            coverage_scores=[]
            for _ in range(200):
                y=rng.choice(candidates,size,replace=False)
                coverage_scores.append(float(np.median(1-cross_cos[:,y].max(axis=1))))
            family_coverage.append(dict(view=view,family=family,synthetic_candidates_each=size,real_groups=len(a),subsamples=200,median_real_nearest_distance=np.median(coverage_scores),p025=np.quantile(coverage_scores,.025),p975=np.quantile(coverage_scores,.975),interpretation='same_candidate_count_descriptive_coverage_not_quality_rank'))
        if view=='original':
            pca=PCA(n_components=2,random_state=SEED); p=pca.fit_transform(centers)
            tsne=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=SEED,max_iter=1000).fit_transform(centers)
            coord=meta.copy(); coord['pca1']=p[:,0]; coord['pca2']=p[:,1]; coord['tsne1']=tsne[:,0]; coord['tsne2']=tsne[:,1]
            export(coord,'projection_coordinates'); metrics['pca_2d_variance_fraction']=float(pca.explained_variance_ratio_.sum())
            fig,axes=plt.subplots(1,2,figsize=(14,6))
            for ax,xy,title in [(axes[0],p,'Shared PCA'),(axes[1],tsne,'Shared t-SNE')]:
                for family in ['real']+sorted(sb.family.unique()):
                    mask=meta.family.eq(family).to_numpy(); ax.scatter(xy[mask,0],xy[mask,1],s=23,alpha=.8,label=family)
                ax.set_title(title+'; one centroid per nominal identity'); ax.legend()
            finish(fig,'speaker_projections')
    export(pd.DataFrame(distributions),'cosine_distribution_summary')
    export(pd.DataFrame(positive_rows),'within_identity_pairs')
    export(pd.DataFrame(neighbours),'cross_class_nearest_neighbours')
    export(pd.DataFrame(mmd),'kernel_distribution_distances')
    export(pd.DataFrame(stability),'equal_size_family_sensitivity')
    export(pd.DataFrame(family_coverage),'equal_size_family_coverage')
    export(pd.DataFrame(single_clip),'one_clip_per_identity_sensitivity')
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for ax,view in zip(axes,['original','edge_trimmed']):
        part=pd.DataFrame(mmd); part=part.loc[part.view.eq(view)]
        for target,g in part.groupby('target'):
            ax.plot(g.bandwidth_multiplier,g.biased_weighted_mmd_squared,marker='o',label=target)
        ax.set(title=view,xlabel='RBF bandwidth multiplier',ylabel='Biased weighted MMD squared; not a quality score');ax.legend(fontsize=8)
    finish(fig,'kernel_sensitivity')
    fig,ax=plt.subplots(figsize=(11,5))
    part=pd.DataFrame(family_coverage)
    for offset,view in [(-.08,'original'),(.08,'edge_trimmed')]:
        g=part.loc[part.view.eq(view)].sort_values('family'); y=g.median_real_nearest_distance.to_numpy()
        ax.errorbar(np.arange(len(g))+offset,y,yerr=[y-g.p025.to_numpy(),g.p975.to_numpy()-y],fmt='o',capsize=4,label=view)
    ax.set_xticks(np.arange(len(g)),g.family);ax.set(ylabel='Median real -> nearest synthetic cosine distance',title='20 candidate voices per family; observed subset ranges, not quality rank');ax.legend()
    finish(fig,'family_coverage_controlled_count')
    export(pd.DataFrame(center_rows),'centroid_metadata')
    # Independent acoustic view, identical functions and weights to ECAPA sample.
    acoustics=pd.DataFrame([analyze_one(r) for r in samples.to_dict('records')])
    export(acoustics,'sample_acoustic_features')
    export(pd.concat([effects(acoustics,'selected_original'),effects(acoustics,'selected_edge_trimmed','edge_trimmed_')],ignore_index=True),'selected_acoustic_effect_sizes')
    original={str(k):e for k,v,e in zip(data['sample_ids'],data['views'],data['embeddings']) if v=='original'}
    trimmed={str(k):e for k,v,e in zip(data['sample_ids'],data['views'],data['embeddings']) if v=='edge_trimmed'}
    sensitivity=samples[['sample_id','label','group_id','family']].copy()
    sensitivity['original_trimmed_cosine']=sensitivity.sample_id.map(lambda k:float(original[k]@trimmed[k]))
    export(sensitivity,'embedding_boundary_sensitivity')
    keys=['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction','energy_dynamic_range_db','spectral_centroid_hz','highband_energy_fraction','spectral_flatness','zero_crossing_fraction']
    g=acoustics.groupby(['label','group_id','family'])[keys].median().reset_index()
    export(g,'identity_acoustic_features')
    z=RobustScaler().fit_transform(g[keys]); z=np.nan_to_num(z)
    fig,ax=plt.subplots(figsize=(12,6))
    means=pd.DataFrame(z,columns=keys).assign(label=g.label.to_numpy()).groupby('label').mean().T
    means.plot.barh(ax=ax); ax.set_xlabel('Mean of pooled median/IQR-scaled identity features; not SNR or quality rank')
    finish(fig,'acoustic_multivariate_view')
    metrics.update(real_clips=int(samples.label.eq('real').sum()),synthetic_clips=int(samples.label.eq('synthetic').sum()),real_speakers=int(samples.loc[samples.label.eq('real'),'group_id'].nunique()),synthetic_nominal_voices=116,embedding_dimensions=192,views=2,api_calls=0)
    (OUT/'key_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    inputs=[OUT/'samples.csv',OUT/'embeddings.npz',OUT/'model_provenance.json',ROOT/'prepare_voice_distribution.py',ROOT/'ecapa_embeddings.py',Path(__file__),ROOT/'VOICE_DISTRIBUTION_PLAN.md']
    (OUT/'analysis_provenance.json').write_text(json.dumps({'seed':SEED,'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'mmd':'biased weighted RBF MMD squared; no significance or causal claims','embedding_centroids':'L2-normalized mean of L2 sample vectors; equal nominal identity weighting','text_caution':'all synthetic same sentence; real varied content; codec and duration still uncontrolled','projection_caution':'PCA/t-SNE are visualization only; metrics use original 192 dimensions','acoustic_scaling':'pooled observed identity median/IQR; not fitted detector or normalization policy'},indent=2),encoding='utf-8')
    return metrics

if __name__=='__main__': print(json.dumps(main(),indent=2))
