"""Source-only matching; broad age categories are an experimental policy.

Provider age words have no numeric definition. The mapping is not a verified
age or a claim about the perceived age of any recording. No audio inference.
"""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
AGE_POLICY={'18-29':'young','30-39':'young','40-49':'middle_aged',
            '50-59':'middle_aged','60-69':'old','70-79':'old','80-89':'old','90+':'old'}


def eligible_pool(pool,policy):
    if policy not in {'broad','strict'}:raise ValueError('Unknown metadata age policy')
    samples=pd.read_csv(ROOT/'voice_distribution/samples.csv',keep_default_na=False).set_index('sample_id')
    manifest=pd.read_csv(ROOT/'synthetic_pilot/manifest.csv',keep_default_na=False).set_index('sample_id')
    data=pool.copy()
    data['real_source_sex_label']=data.real_sample.map(samples.sex_label)
    data['synthetic_provider_gender_label']=data.synthetic_sample.map(manifest.provider_gender_label)
    data['real_source_age_proxy_bin']=data.real_sample.map(samples.age_proxy_bin)
    data['synthetic_provider_age_label']=data.synthetic_sample.map(manifest.provider_age_label)
    data['synthetic_label_source']=data.synthetic_sample.map(manifest.provider_label_source)
    data['same_source_gender_label']=(data.real_source_sex_label.isin(['male','female']) &
        data.real_source_sex_label.eq(data.synthetic_provider_gender_label))
    data['experimental_real_age_category']=data.real_source_age_proxy_bin.map(AGE_POLICY).fillna('unknown')
    data['same_experimental_age_category']=(data.experimental_real_age_category.ne('unknown') &
        data.experimental_real_age_category.eq(data.synthetic_provider_age_label))
    data['numerically_verified_age_match']=False
    if policy=='strict':return data.iloc[:0].copy()
    human=ROOT/'similar_voice_pairs/human_pair_feedback.json'
    rejected={r['pair_id'] for r in json.loads(human.read_text(encoding='utf-8'))} if human.exists() else set()
    return data.loc[data.same_source_gender_label & data.same_experimental_age_category & ~data.pair_id.isin(rejected)].copy()
