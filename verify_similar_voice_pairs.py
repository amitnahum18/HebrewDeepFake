"""Verify controlled selections, audio provenance and offline gallery assets."""
from pathlib import Path
import hashlib,json,re,subprocess,wave,base64
import numpy as np
import pandas as pd
from comparative_audio import read_wav

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'similar_voice_pairs'
checks={}
def check(name,value):checks[name]=bool(value);assert value,name
s=pd.read_csv(OUT/'selected_pairs.csv');pool=pd.read_csv(OUT/'controlled_candidate_pairs.csv')
assets=json.loads((OUT/'audio_assets.json').read_text(encoding='utf-8'))
sources=pd.read_csv(ROOT/'voice_distribution/samples.csv').set_index('sample_id')
check('24_selections_six_per_group',len(s)==24 and s.groupby('selection_method').size().eq(6).all())
check('all_pairs_controlled_eight_words_half_second',s.real_words.eq(8).all() and s.synthetic_words.eq(8).all() and s.duration_gap_s.le(.5+1e-8).all())
check('not_false_same_text_claim',s.text_matched.eq(False).all())
for method,g in s.groupby('selection_method'):
    check('unique_real_synthetic_family:'+method,g.real_group.nunique()==6 and g.synthetic_group.nunique()==6 and g.family.nunique()>=2 and g.family.value_counts().max()<=3)
    key=method+'_percentile' if method!='consensus' else 'consensus_max_percentile'
    check('relative_scores_correct_sorted:'+method,np.allclose(g.relative_match_score,100*(1-g[key])) and g.relative_match_score.is_monotonic_decreasing)
check('selected_pairs_from_shared_pool',set(s.pair_id).issubset(set(pool.pair_id)))
check('relative_scores_bounded',s.relative_match_score.between(0,100).all())
for sid,a in assets.items():
    raw=OUT/a['raw_audio'];level=OUT/a['level_audio']
    source=sources.loc[sid]
    check('source_id_to_audio:'+sid,hashlib.sha256(Path(source.absolute_wav_path).read_bytes()).hexdigest()==a['raw_sha256']==source.wav_sha256)
    check('raw_provenance:'+sid,hashlib.sha256(raw.read_bytes()).hexdigest()==a['raw_sha256']==a['source_sha256'])
    check('listening_hash:'+sid,hashlib.sha256(level.read_bytes()).hexdigest()==a['level_sha256'])
    x=read_wav(raw);y=read_wav(level)
    check('listening_gain_only:'+sid,len(x)==len(y) and np.max(np.abs(y-x*10**(a['listening_gain_db']/20)))<=1/32768+1e-8)
    check('finite_nonzero_headroom:'+sid,np.isfinite(y).all() and np.any(y) and np.max(np.abs(y))<.991)
page=(OUT/'listen.html').read_text(encoding='utf-8')
embedded=json.loads(re.search(r'const AUDIO_SOURCES=(.*?);const METHOD_NAMES=',page).group(1))
for sid,a in assets.items():
    for kind in ['raw','level']:
        payload=embedded[a[kind+'_audio']].split(',',1)[1]
        check('embedded_audio_mapping:'+sid+':'+kind,hashlib.sha256(base64.b64decode(payload)).hexdigest()==a[kind+'_sha256'])
check('no_remote_dependencies','https://' not in page and '<script src=' not in page)
check('default_consensus_blind_local_ratings','value="consensus"' in page and 'let revealed=false' in page and 'localStorage' in page)
cards=json.loads((OUT/'gallery_data.json').read_text(encoding='utf-8'))
feedback=json.loads((OUT/'human_pair_feedback.json').read_text(encoding='utf-8'))
check('two_human_rejected_pairs_recorded',len(feedback)==2 and {f['reason'] for f in feedback}=={'perceived_age_difference','perceived_voice_presentation_difference'})
by_feedback={f['pair_id']:f for f in feedback}
for card in cards:
    if card['pair_id'] in by_feedback:
        f=by_feedback[card['pair_id']]
        check('feedback_side_mapping:'+card['selection_method']+str(card['within_method_rank']),card['sides']['A']['sample_id']==f['side_A'] and card['sides']['B']['sample_id']==f['side_B'])
        check('human_rejection_propagates:'+card['selection_method']+str(card['within_method_rank']),card['matching_audit']['human_feedback_status']=='human_reported_mismatch')
    check('one_real_one_synthetic:'+card['selection_method']+str(card['within_method_rank']),{card['sides']['A']['label'],card['sides']['B']['label']}=={'real','synthetic'})
script=re.search(r'<script>([\s\S]*)</script>',page).group(1)
js=OUT/'gallery_script_check.js';js.write_text(script,encoding='utf-8')
result=subprocess.run(['node','--check',str(js)],capture_output=True,text=True)
check('gallery_javascript_compiles',result.returncode==0)
check('two_nonempty_plots',len(list((OUT/'figures').glob('*.png')))==2 and all(p.stat().st_size>1000 for p in (OUT/'figures').glob('*.png')))
p=json.loads((OUT/'provenance.json').read_text())
for path,digest in p['inputs'].items():check('input_hash:'+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
browser_result=json.loads((OUT/'browser_verification.json').read_text()) if (OUT/'browser_verification.json').exists() else None
current_browser=browser_result and browser_result.get('passed') and browser_result.get('gallery_sha256')==hashlib.sha256((OUT/'listen_portable.html').read_bytes()).hexdigest()
report={'passed':all(checks.values()),'checks':checks,'unique_audio_recordings':len(assets),'selected_cards':len(s),'unique_pairs':s.pair_id.nunique(),'audio_copies':len(assets)*2,'gallery_js_checked':True,'browser_playback_test':'passed in isolated test browser' if current_browser else 'not yet performed for this gallery version','browser_playback_cases':browser_result.get('playback_cases',0) if current_browser else 0,'new_speech_generated':False,'api_calls':0,'human_similarity_ratings':0,'human_pair_feedback_records':len(feedback)}
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
