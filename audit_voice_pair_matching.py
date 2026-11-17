"""Audit existing selections; keep human feedback separate from weak metadata.

This does not infer age/sex from audio, change source labels, or rerank results.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'similar_voice_pairs'


def audit():
    selected = pd.read_csv(OUT / 'selected_pairs.csv')
    samples = pd.read_csv(ROOT / 'voice_distribution/samples.csv').set_index('sample_id')
    reviews = pd.read_csv(ROOT / 'synthetic_pilot/automated_review.csv').set_index('sample_id')
    # Exact IDs preserve feedback even if the ordering changes later.
    feedback = [
        {'pair_id': 'real_hhimhVLGDTDBxuoMlA5t3_65__e1a3a90aa0aee4824351',
         'original_group': 'A', 'original_rank': 4,
         'status': 'human_reported_mismatch', 'reason': 'perceived_age_difference',
         'note_he': 'לפי האזנת המשתמש: הקלטה א׳ נשמעת ילדה והקלטה ב׳ נשמעת בוגרת יותר. זהו תיאור תפיסתי, לא גיל מאומת.',
         'side_A': 'real_hhimhVLGDTDBxuoMlA5t3_65', 'side_B': 'e1a3a90aa0aee4824351'},
        {'pair_id': 'real_7gkGX0sVCA46_OiQfobgP_0__3bd9461132b8ffcc2960',
         'original_group': 'A', 'original_rank': 5,
         'status': 'human_reported_mismatch', 'reason': 'perceived_voice_presentation_difference',
         'note_he': 'לפי האזנת המשתמש: הקלטה אחת נשמעת גברית והשנייה נשית. לא נמסר שיוך אנושי לכל צד.',
         'side_A': '3bd9461132b8ffcc2960', 'side_B': 'real_7gkGX0sVCA46_OiQfobgP_0'},
    ]
    for item in feedback:
        item.update(reviewer='user', reported_date='2026-11-17',
                    scope='pair similarity only; no source demographic label overwritten',
                    numeric_similarity_rating=None)
    (OUT / 'human_pair_feedback.json').write_text(json.dumps(feedback, ensure_ascii=False, indent=2), encoding='utf-8')
    by_id = {f['pair_id']: f for f in feedback}
    rows = []
    for r in selected.to_dict('records'):
        real = samples.loc[r['real_sample']]
        synthetic = reviews.loc[r['synthetic_sample']]
        source = str(real.sex_label)
        presentation = str(synthetic.perceived_voice_presentation)
        proxy = {'male': 'masculine', 'female': 'feminine'}.get(source)
        # Source sex and model-perceived presentation are different constructs.
        # Flag disagreement for listening, never treat agreement as validation.
        status = ('unknown' if proxy is None or presentation not in {'masculine', 'feminine'}
                  else 'proxy_disagreement' if proxy != presentation else 'proxy_agreement_unverified')
        human = by_id.get(r['pair_id'])
        rows.append({
            'selection_method': r['selection_method'], 'within_method_rank': r['within_method_rank'],
            'pair_id': r['pair_id'], 'real_source_sex_label': source,
            'synthetic_model_perceived_presentation': presentation,
            'synthetic_presentation_confidence': synthetic.presentation_confidence,
            'synthetic_human_verified': bool(synthetic.human_verified),
            'metadata_screen_status': status,
            'human_feedback_status': human['status'] if human else 'not_reviewed',
            'human_feedback_reason': human['reason'] if human else '',
            'human_feedback_note_he': human['note_he'] if human else '',
            'age_matching_status': 'human_reported_difference' if human and human['reason']=='perceived_age_difference' else 'not_validated',
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'matching_audit.csv', index=False, encoding='utf-8-sig')
    summary = frame.groupby('selection_method').agg(
        cards=('pair_id', 'size'),
        metadata_disagreements=('metadata_screen_status', lambda x: x.eq('proxy_disagreement').sum()),
        unknown_metadata=('metadata_screen_status', lambda x: x.eq('unknown').sum()),
        human_reported_mismatches=('human_feedback_status', lambda x: x.eq('human_reported_mismatch').sum()))
    text = '''# בדיקת התאמת הזוגות בעקבות משוב אנושי

## הממצא

זוגות A4 ו־A5 דווחו כלא מתאימים. ב־A4 המשתמש שמע הבדל בגיל הנתפס: א׳ נשמעת ילדה וב׳ בוגרת יותר. ב־A5 דווח על הבדל בין קול גברי לנשי. אין כאן אימות של גיל או מין בפועל.

הבחירה המקורית הגבילה מספר מילים, משך וגיוון דוברים וספקים; היא לא הגבילה התאמה בגיל נתפס או בהצגה קולית. לכן מרחק מספרי קטן לא הבטיח התאמה שמיעתית. דירוג יחסי של 99.77 אינו 99.77% דמיון ואינו הסתברות לאותה זהות.

בדיקת hashes ושיוך מזהי המקור לנגנים נעשית בנפרד מבדיקת הדמיון. מעבר של בדיקת ניגון אינו אישור לאיכות הזוג.

## סריקה של כל הקבוצות

הטבלה משווה תיוג מין קיים במקור האמיתי לסיווג אוטומטי של הצגת הקול המסונתז. אלה משתנים שונים, ולכן אי־הסכמה היא דגל לביקורת ולא הוכחה לטעות. גם הסכמה אינה אישור אנושי. הסיווגים המסונתזים לא אומתו אנושית; גילם הנתפס אינו מאומת.

| שיטה | זוגות | אי־הסכמה בין תיוגי העזר | תיוג לא ידוע | זוגות שנפסלו במשוב המשתמש |
|---|---:|---:|---:|---:|
'''
    names = {'hybrid': 'A — שילוב', 'acoustic': 'B — אקוסטיקה', 'ecapa': 'C — ECAPA', 'consensus': 'הסכמה'}
    for method, row in summary.iterrows():
        text += f'| {names[method]} | {row.cards} | {row.metadata_disagreements} | {row.unknown_metadata} | {row.human_reported_mismatches} |\n'
    text += '''
## תיקון התהליך

- המשוב נשמר לפי מזהי הקבצים והצדדים ב־`human_pair_feedback.json`, ומופיע בכל מקום שבו אותו זוג מוצג. לא הומצאו ציוני דמיון מספריים מהמשוב.
- הבחירות, הציונים והסדר המקוריים נשמרו כתיעוד של ביצועי השיטות. זוגות שנפסלו אינם מוצגים כמתאימים מאומתים. החלפה שקטה שלהם הייתה מסתירה את הכשל.
- יש ליצור בהמשך קבוצת התאמה נפרדת: אימות אנושי של ההצגה הקולית וטווח הגיל הנתפס בכל קטע, טיפול מפורש בלא ידוע, ואז דירוג מרחקים בתוך קבוצות מתאימות. אין לקבוע גיל מדויק מהקול.
- לבדיקת השיטות עצמן יש להשאיר גם כשלים, לאסוף דירוגי דמיון בהאזנה עיוורת ולהפריד קבוצת כיול מקבוצת הערכה. יש לדווח על שיעור כשלים ועל הסכמה בין מעריכים, ולא להסתפק בששת הזוגות הטובים ביותר.
- רצוי להוסיף מאפייני גובה קול וקצב כמשתנים אקוסטיים לבדיקה, לאחר בדיקת איכות חילוצם; אין להשתמש בהם לבדם כהוכחה למין או לגיל.

לא נוצר אודיו חדש, לא שונו תוויות המקור, והסיווג האוטומטי לא שימש כמסנן דמוגרפי מחייב.
'''
    (OUT / 'MATCHING_AUDIT_HE.md').write_text(text, encoding='utf-8')
    # Keep each published per-selection record consistent with the gallery.
    lookup = {}
    for row in rows:
        lookup[(row['selection_method'], row['within_method_rank'])] = row
        path = OUT / 'metadata' / f"{row['selection_method']}_{row['within_method_rank']}.json"
        data = json.loads(path.read_text(encoding='utf-8'))
        data['matching_audit'] = row
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return lookup


if __name__ == '__main__':
    audit()
