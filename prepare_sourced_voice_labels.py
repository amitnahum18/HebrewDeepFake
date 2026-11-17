"""Consolidate supplied labels without inferring demographics from audio.

Real-source demographic metadata and synthetic-provider voice descriptions are
different fields. Existing user impressions stay excerpt-scoped and separate.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sourced_voice_labels'


def main():
    OUT.mkdir(exist_ok=True)
    real_path = ROOT / 'eda_outputs/recording_features.csv'
    synthetic_path = ROOT / 'synthetic_pilot/manifest.csv'
    user_path = ROOT / 'voice_review/review_queue.csv'
    real = pd.read_csv(real_path, keep_default_na=False)
    synthetic = pd.read_csv(synthetic_path, keep_default_na=False)
    user = pd.read_csv(user_path, keep_default_na=False)
    records = []
    for r in real.itertuples():
        records.append(dict(
            recording_id=str(r.session_id), dataset='real', group_id=str(r.user_id),
            duration_s=float(r.session_duration),
            source_sex_label=str(r.sex_label), provider_voice_gender_label=None,
            label_source='crowd-recital source metadata; normalized existing sex_label',
            source_file='eda_outputs/recording_features.csv',
            source_raw_biological_sex=str(r.biological_sex),
            annotation_scope='source session metadata',
            label_availability='available_in_source' if r.sex_label in {'male','female'} else 'unknown',
            new_audio_inference=False, model='', voice=''))
    for r in synthetic.itertuples():
        label=str(r.provider_gender_label)
        records.append(dict(
            recording_id=str(r.sample_id), dataset='synthetic', group_id=str(r.voice_identity_group),
            duration_s=float(r.duration_s), source_sex_label=None,
            provider_voice_gender_label=label,
            label_source=str(r.provider_label_source),
            source_file='synthetic_pilot/manifest.csv', source_raw_biological_sex=None,
            annotation_scope='provider description of nominal synthetic voice; not a person demographic',
            label_availability='available_in_source' if label in {'male','female','neutral'} else 'unknown',
            new_audio_inference=False, model=str(r.model), voice=str(r.voice)))
    # Preserve exactly what the user supplied; do not propagate auditory
    # impressions to source demographics or other recordings of a source ID.
    impressions = user.to_dict('records')
    payload = dict(schema_version='1.0', created_date='2026-11-17',
                   method='existing source/provider labels only; no new demographic inference from audio',
                   recordings=records, existing_user_excerpt_annotations=impressions,
                   source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [real_path,synthetic_path,user_path]})
    assert len(records)==len(real)+len(synthetic)
    assert len({(r['dataset'],r['recording_id']) for r in records})==len(records)
    assert all(not r['new_audio_inference'] for r in records)
    assert all(r['provider_voice_gender_label'] is None for r in records if r['dataset']=='real')
    assert all(r['source_sex_label'] is None for r in records if r['dataset']=='synthetic')
    counts={
        'real_source_labels':dict(Counter(r['source_sex_label'] for r in records if r['dataset']=='real')),
        'synthetic_provider_voice_labels':dict(Counter(r['provider_voice_gender_label'] for r in records if r['dataset']=='synthetic')),
        'existing_user_excerpt_annotations':len(impressions),
        'recordings':len(records), 'new_audio_inferences':0,
        'automatic_auditory_demographic_predictions_used':False,
    }
    (OUT/'recording_labels.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    text='''# תיוגים קיימים להקלטות אמת וסינתזה

זהו איחוד תיוגים ממקורות קיימים, ולא תיוג חדש של בן/בת מתוך קול. אין להסיק מגדר או מין של אדם מקולו בלבד.

| מקור | זכר / קול שמתואר כגברי | נקבה / קול שמתואר כנשי | ניטרלי | לא ידוע | סך הקלטות |
|---|---:|---:|---:|---:|---:|
'''
    a=counts['real_source_labels'];b=counts['synthetic_provider_voice_labels']
    text+=f"| אמת — תווית מקור | {a.get('male',0)} | {a.get('female',0)} | 0 | {a.get('unknown',0)} | {len(real)} |\n"
    text+=f"| סינתזה — תיאור קול של הספק | {b.get('male',0)} | {b.get('female',0)} | {b.get('neutral',0)} | {b.get('unknown',0)} | {len(synthetic)} |\n"
    text+='''
הספירות הן של קבצים/סשנים, לא של דוברים ייחודיים. יחידת האמת היא סשן המקור; יחידת הסינתזה היא קובץ שנוצר. השוואת התפלגויות דורשת גם שקלול לפי משך וקבוצת דובר.

`recording_labels.json` מכיל מזהה, קבוצת דובר/קול, משך, תווית קיימת, מקור התווית, תחולת התיוג ומודל/קול בסינתזה. השדות לתווית המקור האמיתי ולתיאור קול מסונתז נפרדים במפורש. לא הומר תיאור קול למין של אדם.

14 התרשמויות קודמות של המשתמש נשמרו בנפרד כפי שתועדו, בתחולת הקטע שנשמע בלבד. הן לא הוחלו על סשנים שלמים או על קטעים אחרים. המשוב על זוגות A4/A5 נשאר ב־`similar_voice_pairs/human_pair_feedback.json` ולא שימש לקביעת תווית דמוגרפית.

178 סשנים אמיתיים ו־200 קבצים מסונתזים נשארו ללא תיוג מקור. כדי להשלים תוויות דמוגרפיות יש להשתמש במידע מפורש של הדוברים או של הספק. סיווגים אוטומטיים קודמים לפי האודיו לא שימשו למילוי החסר. שם קול לבדו לא שימש לתיוג.

לצורך התאמת קולות אפשר למדוד בנפרד גובה קול, קצב וגוון, ולאסוף דירוגי דמיון אנושיים. מאפיינים אלה אינם תחליף למידע דמוגרפי.

שחזור: `python prepare_sourced_voice_labels.py`. לא בוצעו קריאות API, לא נוצר אודיו, וקובצי המקור נשארו ללא שינוי.
'''
    (OUT/'README_HE.md').write_text(text,encoding='utf-8')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    main()
