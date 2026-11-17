"""Exercise portable HTML playback from a folder containing no audio assets."""
from pathlib import Path
import json,shutil,hashlib
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'similar_voice_pairs';TEST=ROOT/'tmp/voice_pair_portable_test'
metadata_match=bool(json.loads((OUT/'key_metrics.json').read_text()).get('metadata_matching_policy'))
TEST.mkdir(parents=True,exist_ok=True)
standalone=TEST/'listen_only.html';shutil.copy2(OUT/'listen_portable.html',standalone)
with sync_playwright() as p:
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    browser=p.chromium.launch(headless=True,executable_path=str(edge),args=['--autoplay-policy=no-user-gesture-required'])
    context=browser.new_context(viewport={'width':1280,'height':1000},accept_downloads=True)
    page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(standalone.as_uri());page.wait_for_selector('.card')
    assert page.locator('.card').count()==6
    assert page.locator('audio').count()==12
    assert page.locator('#cards .score-box').count()==18
    assert page.locator('#group-summary .score-box').count()==3
    assert page.locator('audio').first.get_attribute('src').startswith('data:audio/wav;base64,')
    playback=[]
    for level in ['level_audio','raw_audio']:
        page.locator('#level').select_option(level)
        for method in ['consensus','acoustic','ecapa','hybrid']:
            page.locator('#method').select_option(method)
            assert page.locator('.card').count()==6
            assert page.locator('#cards .score-box').count()==18
            assert page.locator('#group-summary .score-box').count()==3
            if metadata_match:
                assert page.locator('[data-metadata-eligible="true"]').count()==6
                assert page.locator('[data-review-status="human_reported_mismatch"]').count()==0
                assert 'לא גיל מאומת' in page.locator('.card').first.inner_text()
            elif method=='hybrid':
                assert page.locator('[data-review-status="human_reported_mismatch"]').count()==2
                assert 'ילדה' in page.locator('.card').nth(3).inner_text()
                assert 'גברית' in page.locator('.card').nth(4).inner_text()
            assert all(code in page.locator('.card').first.inner_text() for code in ['A —','B —','C —'])
            for index in range(12):
                page.locator('.play-audio').nth(index).click()
                page.wait_for_function('(index)=>{const a=document.querySelectorAll("audio")[index];return a.currentTime>0.1&&!a.paused}',arg=index,timeout=15000)
                state=page.locator('audio').nth(index).evaluate('(a)=>({duration:a.duration,time:a.currentTime,readyState:a.readyState,error:a.error?.message||null})')
                assert state['duration']>0 and state['time']>.1 and state['error'] is None
                assert page.locator('audio').evaluate_all('(all)=>all.filter(a=>!a.paused).length')==1
                playback.append({'level':level,'method':method,'side_index':index,**state})
    page.locator('#method').select_option('consensus');page.locator('#level').select_option('level_audio')
    page.locator('[data-key="voice_similarity"]').first.select_option('4')
    assert len(page.evaluate("JSON.parse(localStorage.getItem('hebrew_similar_voice_pairs_v1'))"))==1
    with page.expect_download() as download:page.locator('#export').click()
    assert download.value.suggested_filename=='voice_pair_listening_ratings.json'
    page.locator('#reveal').click();assert 'מסונתזת' in page.locator('#cards').inner_text()
    page.evaluate('localStorage.clear()');page.reload();page.locator('audio').first.evaluate('a=>a.pause()')
    page.screenshot(path=str(OUT/'browser_desktop.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844});page.screenshot(path=str(OUT/'browser_mobile.png'),full_page=False)
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    result={'passed':True,'gallery_sha256':hashlib.sha256(standalone.read_bytes()).hexdigest(),'cards_per_group':6,'standalone_html_without_audio_folder':True,'playback_cases':len(playback),'tests':playback,'one_playing_at_a_time':True,'rating_save_export':True,'mobile_no_overflow':True,'page_errors':errors,'test_ratings_added_to_dataset':False}
    (OUT/'browser_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='tests'},indent=2))
    context.close();browser.close()
