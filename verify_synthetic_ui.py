"""Check the offline listening workbench in an isolated headless browser."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root=Path(__file__).resolve().parent
out=root/'synthetic_pilot'/'analysis'
with sync_playwright() as p:
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    browser=p.chromium.launch(headless=True,executable_path=str(edge)) if edge.exists() else p.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1100},accept_downloads=True)
    page=context.new_page(); errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((root/'synthetic_pilot'/'review.html').as_uri())
    assert page.locator('#gallery .card').count()==24
    assert page.locator('#model option').count()==11
    duration=page.locator('#gallery audio').first.evaluate('''async audio => {
      await audio.play(); await new Promise(resolve=>setTimeout(resolve,200)); audio.pause(); return audio.duration;
    }''')
    assert duration>0
    page.locator('#gallery .card select').nth(1).select_option('uncertain')
    saved=page.evaluate("JSON.parse(localStorage.getItem('HebrewSyntheticPilotRatingsV1'))")
    assert len(saved)==1
    with page.expect_download() as download:
        page.locator('#export').click()
    assert download.value.suggested_filename=='synthetic_human_ratings.json'
    page.evaluate('localStorage.clear()'); page.reload()
    page.locator('#model').select_option('speechify/simba-3.5')
    assert page.locator('#gallery .card').count()==20
    page.locator('#model').select_option('')
    page.screenshot(path=str(out/'ui_desktop.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(out/'ui_mobile.png'),full_page=False)
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    assert not errors, errors
    result={'status':'passed','initial_audio_cards':24,'model_filters':10,'audio_playback_duration_s':duration,
            'human_rating_save':'passed_in_isolated_ephemeral_browser','rating_export':'passed',
            'mobile_no_horizontal_overflow':True,'page_errors':errors,
            'test_human_ratings_included_in_dataset':False}
    (out/'ui_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    context.close(); browser.close()
