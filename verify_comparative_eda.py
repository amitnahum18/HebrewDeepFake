"""Verify executed outputs, untouched audio, and prespecified matching constraints."""
from pathlib import Path
import hashlib, json
import pandas as pd
import nbformat

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'comparison_outputs'; BASE=ROOT/'synthetic_pilot'
checks={}
def check(name,condition):
    checks[name]=bool(condition)
    if not condition: raise AssertionError(name)

nb=nbformat.read(ROOT/'real_synthetic_comparison.ipynb',as_version=4)
codes=[c for c in nb.cells if c.cell_type=='code']
check('every_code_cell_executed',all(c.execution_count is not None for c in codes))
check('zero_notebook_errors',not any(o.output_type=='error' for c in codes for o in c.outputs))
provenance=json.loads((OUT/'provenance.json').read_text())
for path,expected in provenance['input_hashes'].items():
    check('input_hash:'+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected)
common=pd.read_csv(OUT/'common_audio_features.csv')
synth=pd.read_csv(BASE/'analysis/sample_features.csv')
ref=pd.read_csv(BASE/'human_reference.csv')
expected=dict(zip(synth.sample_id,synth.wav_sha256))
expected.update({'real_'+r.session_id+'_'+str(r.segment_index):r.wav_sha256 for r in ref.itertuples()})
check('all_saved_audio_hashes_unchanged',all(hashlib.sha256(Path(r.absolute_wav_path).read_bytes()).hexdigest()==r.wav_sha256==expected[r.sample_id] for r in common.itertuples()))
check('362_unique_samples',len(common)==362 and common.sample_id.is_unique)
check('78_real_116_nominal_voice_groups',common.groupby('label').group_id.nunique().to_dict()=={'real':78,'synthetic':116})
check('finite_primary_features',common[['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction']].notna().all().all())
pairs=pd.read_csv(OUT/'diagnostic_duration_word_pairs.csv')
check('44_diagnostic_pairs',len(pairs)==44)
check('unique_diagnostic_identities',pairs.real_group_id.is_unique and pairs.synthetic_group_id.is_unique)
check('equal_11_per_family',pairs.groupby('family').size().eq(11).all() and pairs.family.nunique()==4)
check('eight_words_and_half_second_caliper',pairs.real_words.eq(8).all() and pairs.synthetic_words.eq(8).all() and pairs.duration_difference_s.le(.5+1e-9).all())
check('diagnostic_text_not_claimed_matched',pairs.text_matched.eq(False).all())
proposal=pd.read_csv(OUT/'proposed_same_text_pairs.csv')
coverage=pd.read_csv(OUT/'proposed_speaker_coverage.csv')
check('308_unique_proposals_81_speakers',len(proposal)==308 and proposal.exact_text_group.is_unique and proposal.user_id.nunique()==81)
check('maximum_four_per_speaker',proposal.groupby('user_id').size().le(4).all())
check('proposed_duration_words_scores',proposal.duration_s.between(2,10).all() and proposal.word_count.between(6,14).all() and proposal.quality_score.ge(.8).all() and proposal.segment_alignment_score.ge(.8).all() and proposal.hebrew_letter_fraction.ge(.8).all())
check('77_proposals_per_family',proposal.groupby('desired_family').size().eq(77).all())
check('all85_speakers_accounted',len(coverage)==85 and coverage.user_id.is_unique and coverage.proposed_pairs.sum()==308 and coverage.proposed_pairs.eq(0).sum()==4)
check('future_assignments_not_fabricated',proposal[['assigned_model','assigned_voice','split']].isna().all().all())
funnel=pd.read_csv(OUT/'real_eligibility_funnel.csv')
check('sequential_funnel_reconciles',funnel.segments.iloc[0]-funnel.removed_at_step.sum()==funnel.segments.iloc[-1]==50103)
effect=pd.read_csv(OUT/'group_effect_sizes.csv')
check('four_views_nine_features',len(effect)==36 and effect.groupby('view').size().eq(9).all())
check('no_invalid_auc_claims',effect.in_sample_orientation_free_auc.between(.5,1).all())
check('four_nonempty_figures',len(list((OUT/'figures').glob('*.png')))==4 and all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.png')))
result={'passed':all(checks.values()),'checks':checks,'executed_code_cells':len(codes),'output_csvs':len(list(OUT.glob('*.csv'))),'figures':4,'human_labels_added':0,'api_calls':0}
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
