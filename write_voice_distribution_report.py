"""Write a Hebrew results report using the completed local analysis outputs."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'voice_distribution'
def read(name):return pd.read_csv(OUT/(name+'.csv'))
def table(df):
    rows=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x).replace('|','/') for x in row)+' |')
    return '\n'.join(rows)

def main():
    m=json.loads((OUT/'key_metrics.json').read_text())
    dist=read('cosine_distribution_summary')
    original=dist.loc[dist.view.eq('original'),['label','comparison','n','median','p05','p95']]
    kernel=read('kernel_distribution_distances')
    kernel=kernel.loc[kernel.bandwidth_multiplier.eq(1),['view','target','biased_weighted_mmd_squared','real_groups','synthetic_groups']]
    nearest=read('cross_class_nearest_neighbours')
    near=nearest.groupby(['view','direction']).cosine_distance.agg(['count','median','min','max']).reset_index()
    acoustic=read('identity_acoustic_features').groupby('label').median(numeric_only=True)[['duration_s','rms_dbfs','quiet_fraction_45dbfs','edge_quiet_fraction','spectral_centroid_hz']].reset_index()
    boundary=read('embedding_boundary_sensitivity').groupby('label').original_trimmed_cosine.agg(['median','min']).reset_index()
    same=read('within_identity_pairs')
    low=same.loc[same.view.eq('original')&same.label.eq('synthetic')].sort_values('cosine_similarity').head(6)[['group_id','model_a','model_b','cosine_similarity']]
    close=read('closest_synthetic_groups_original').head(6)[['group_a','group_b','cosine_similarity']]
    sample=read('samples')
    real_labels=sample.loc[sample.label.eq('real')].drop_duplicates('group_id')
    sex=real_labels.sex_label.value_counts().rename_axis('source_sex').reset_index(name='speakers')
    age=real_labels.age_proxy_bin.value_counts().rename_axis('source_age_proxy_2026').reset_index(name='speakers')
    single=read('one_clip_per_identity_sensitivity').groupby('view').agg(median_mmd_squared=('mmd_squared_one_clip_per_identity','median'),min_mmd_squared=('mmd_squared_one_clip_per_identity','min'),max_mmd_squared=('mmd_squared_one_clip_per_identity','max'),median_real_cosine=('median_between_real_cosine','median'),median_synthetic_cosine=('median_between_synthetic_cosine','median')).reset_index()
    text=f'''# התפלגויות קול בכמה שיטות — תוצאות ECAPA ואקוסטיקה

תאריך: 17 בנובמבר 2026. הניתוח בוצע בפועל, כולל חילוץ embeddings ממודל ECAPA מקומי. לא בוצעו הפקת דיבור, קריאות API בתשלום או תיוג אנושי חדש.

המחברת: [voice_distributions.ipynb](voice_distributions.ipynb). תוכנית: [VOICE_DISTRIBUTION_PLAN.md](VOICE_DISTRIBUTION_PLAN.md). שחזור: [README_VOICE_DISTRIBUTIONS.md](README_VOICE_DISTRIBUTIONS.md).

## מה נמדד

נבחרו {m['real_clips']} קטעים מ־{m['real_speakers']} דוברים אמיתיים, עד שלושה לדובר ובמידת האפשר מסשנים וטקסטים שונים, באורך 3–8 שניות ובציוני מקור/יישור של לפחות 0.8. לצדם כל {m['synthetic_clips']} המסונתזים מ־{m['synthetic_nominal_voices']} קבוצות ספק:קול. חמשת מזהי המקור ללא מועמד מתאים מתועדים, ולא הוסרו מהקורפוס.

לכל קטע חולץ וקטור בן {m['embedding_dimensions']} ממדים בשתי תצוגות: הקטע המקורי וגרסה עם הסרת עד 0.25 שנייה שקטה מכל צד לפי כלל אנרגיה זהה. הקבצים המקוריים נשמרו. כל זהות נומינלית מקבלת מרכז אחד, בלי לתת משקל יתר לקול שמופיע בכמה גרסאות מודל.

זה מדגם לכיסוי זהויות, לא דגימת שעות ולא ייצוג של כל הדיבור בעברית. כל המסונתזים אומרים אותו משפט והאמיתיים טקסטים מגוונים. דמיון/הבדל בייצוג עלול לשקף גם תוכן, ערוץ, משך ומקור ההקלטה. אין כאן אימות גיל, מין או זהות ביומטרית.

## הממצאים המרכזיים לפעולה

- הייצוג מבחין היטב יחסית בין אותו דובר לדוברים אחרים במדגם האמיתי, גם בסשנים שונים. לכן הוא שימושי לביקורת גיוון מקומית, עם המגבלות בהמשך.
- מרכזי הקולות הסינתטיים קרובים זה לזה יותר בממוצע/חציון מאשר מרכזי הדוברים האמיתיים במדגם. זו אינה הוכחה לחוסר גיוון ביומטרי: כל הסינתטיים חולקים משפט וקיימים הבדלי ערוץ. נדרשת בדיקה חוזרת בטקסטים משותפים ומגוונים.
- אותו שם קול אינו מבטיח עקביות בין גרסאות מודל. אותרו זוגות גרסאות בעלי דמיון נמוך, ורשימות ההאזנה להלן מאפשרות לבדוק אותם בלי לנחש.
- קיצור שקט בקצוות אינו מקרב מהותית את מדד MMD הכולל בתנאים שנבדקו. זה מצמצם הסבר יחיד של שקט, אך אינו מבודד את הסינתזה מתוכן, פונמות וערוץ.
- מפת PCA הדו־ממדית שומרת רק {100*m['pca_2d_variance_fraction']:.1f}% מהשונות, ולכן אינה בסיס מספיק להחלטת matching לבדה.

## 1. בדיקה ראשונית: האם ECAPA שימושי בעברית שלנו

ההפרדה התיאורית בין זוגות של אותו מזהה דובר אמיתי לזוגות של מזהים שונים נותנת AUC של {m['original_real_same_identity_pair_auc']:.4f}. בהגבלת הזוגות החיוביים לסשנים שונים: {m['original_real_cross_session_pair_auc']:.4f}, עם {m['real_cross_session_positive_pairs']} זוגות חיוביים מ־{m['real_cross_session_supported_speakers']} דוברים נתמכים.

זה סימן שהייצוג מכיל מידע שימושי על דוברים במדגם, **לא** AUC של זיהוי זיוף, בדיקת הכללה עצמאית או סף מאומת לזהות. זוגות משתפים הקלטות ודוברים ולכן אינם עצמאיים. דוברים עם קטע יחיד אינם תורמים זוגות עקביות. תנאי ערוץ ושגיאות תיוג אפשריות לא בודדו לגמרי.

## 2. התפלגויות דמיון בתוך קולות ובין קולות

דמיון קוסינוס גבוה יותר משמעו קרבה לפי הייצוג. הטבלה מציגה את התצוגה המקורית; החישוב נעשה במרחב 192 הממדים ולא במפה דו־ממדית.

{table(original)}

[תרשימי עקביות מקוריים](voice_distribution/figures/identity_consistency_original.png) ו־[לאחר קיצור קצוות](voice_distribution/figures/identity_consistency_edge_trimmed.png).

זוגות אותו קול סינתטי משווים גרסאות מודל בעלות אותו מזהה ספק:קול; הם לא משפטים עצמאיים. גם כל הזוגות הסינתטיים האחרים חולקים משפט, ולכן אין לפרש הבדל בינם לבין האמיתיים כהבדל בזהות בלבד.

קבוצות קול עם דמיון נמוך יחסית בין גרסאות, להאזנה:

{table(low)}

קולות בעלי מזהים שונים עם דמיון גבוה יחסית, להאזנה:

{table(close)}

אלה מועמדים לבדיקת גיוון או עקביות. אין כאן הוכחה שאחד הקולות הוא כפיל, טעות או זיוף של אדם אחר. אין למזג או למחוק קולות על בסיס המספר לבדו. רשימות מלאות: [זוגות באותה זהות](voice_distribution/within_identity_pairs.csv), [קולות מסונתזים קרובים](voice_distribution/closest_synthetic_groups_original.csv).

## 3. מפות PCA ו־t-SNE משותפות

[מפת הקולות לפי משפחה](voice_distribution/figures/speaker_projections.png) מציגה מרכז אחד לכל דובר/קול. שני צירי PCA מסבירים {100*m['pca_2d_variance_fraction']:.1f}% מהשונות. המידע שלא מוצג במפה עדיין משתתף במדדי המרחק.

הטלת t-SNE יכולה להבליט שכונות מקומיות ומשתנה עם פרמטרים; מרחק בין איים וצורת האשכולות אינם מדדי התפלגות מכוילים. אין להסיק מחפיפה בתרשים ששתי הקבוצות שקולות או שמודל לא יבחין ביניהן.

## 4. כיסוי: לכל קול, מי השכן מהקבוצה האחרת

{table(near)}

המרחק הוא 1 פחות דמיון קוסינוס. [תרשים הכיסוי](voice_distribution/figures/nearest_coverage_original.png) כולל גם שכן אמיתי אחר כנקודת ייחוס. ל־116 קולות יש יתרון חיפוש אפשרי לעומת מספר קטן יותר של מועמדים; לכן חישבנו גם [כיסוי משפחות עם אותו מספר קולות מועמדים](voice_distribution/equal_size_family_coverage.csv), ב־200 תתי־דגימות.

שכן סינתטי קרוב לאדם הוא דוגמה שנשמעת קרובה למודל הייצוג; הוא אינו חיקוי מאומת של האדם. אנשים שמרחקם לשכן הסינתטי גדול הם מועמדים לבחינת אזורי כיסוי חסרים, ולא בהכרח דוגמאות בעייתיות. [כל השכנים והמרחקים](voice_distribution/cross_class_nearest_neighbours.csv).

## 5. מרחק בין התפלגויות במרחב המלא — MMD

בנוסף למרחקי שכנים חושב MMD בריבוע עם גרעין RBF. בטבלה רוחב הגרעין המרכזי; בקובץ המלא גם 0.5x ו־2x לבדיקת רגישות.

{table(kernel)}

`all_identity_equal` נותן משקל שווה לכל קול; `all_family_equal` נותן משקל כולל שווה לכל אחת מארבע משפחות הסינתזה. כך אפשר לראות אם המסקנה תלויה בהרכב הקטלוג. בנוסף בוצעה [השוואת משפחות באותו גודל מדגם](voice_distribution/equal_size_family_sensitivity.csv), עד 20 זהויות מכל צד, ב־200 חזרות.

זה MMD מוטה בגודל מדגם סופי. ערך נמוך יותר מציין קרבה תחת הייצוג והגרעין האלה; הוא אינו אחוז איכות, טבעיות או ייצוג המציאות. אין כאן מבחן מובהקות, אפקט סיבתי של סינתזה או דירוג ספקים. טווחי תתי־הדגימות מתארים רגישות להרכב הזהויות הנצפות, לא רווח סמך לאוכלוסייה.

מאחר שמספר הקטעים הממוצעים למרכז שונה בין קבוצות, נוספה בדיקת רגישות: קטע אקראי אחד בלבד לכל זהות, 50 פעמים, בלי למצע כמה הקלטות. רוחב הגרעין קבוע לפי הניתוח המרכזי כדי לבודד את שינוי הדגימה. התוצאות:

{table(single)}

זו בדיקה של רגישות לתצפית שנבחרה לכל קול, לא דגימה של דוברים חדשים. גם בה נשארים הבדלי תוכן וערוץ. [כל החזרות](voice_distribution/one_clip_per_identity_sensitivity.csv).

## 6. השוואה אקוסטית נפרדת מ־ECAPA

נמדדו מחדש באותו קוד משך, עוצמה, שקט, טווח אנרגיה ומאפיינים ספקטרליים. חציוני תכונות ברמת זהות:

{table(acoustic)}

[תמונה רב־משתנית של התכונות](voice_distribution/figures/acoustic_multivariate_view.png) משתמשת בסקאלה משותפת של חציון ו־IQR; [טבלת גודלי האפקט ביחידות המקוריות](voice_distribution/selected_acoustic_effect_sizes.csv) כוללת גם מרחק Wasserstein ו־Cliff's delta. המדגם החדש מכוון למשכים קרובים ל־4.5 שניות, ולכן אינו זהה למדגם 78 קטעי הייחוס הקודם.

המדדים אינם SNR או דירוג טבעיות. גם מדד ספקטרלי שמבדיל בין הקבוצות עלול לשקף את הטקסט היחיד בפיילוט, המיקרופון או עקבות הקידוד.

## 7. רגישות הייצוג לשקט בקצוות

דמיון הווקטור של אותו קטע לפני ואחרי הקיצור המוגבל:

{table(boundary)}

דמיון גבוה מציין יציבות לקיצור הספציפי שנבדק, לא חסינות כללית לשקט, רעש או ערוץ. השווינו גם את התפלגויות הקולות וה־MMD בשתי התצוגות. הגרסאות המקורית והמקוצרת של אותו קטע לא נספרו כקולות עצמאיים.

## 8. כיסוי דמוגרפי — מקור בלבד

{table(sex)}

{table(age)}

אלה שדות המקור האמיתי. גיל מחושב ביחס ל־2026 ולא למועד ההקלטה. אין טבלת גיל/מין מקבילה אמינה לסינתטיים, ולא נגזרו תוויות כאלה מ־ECAPA. unknown נשאר גלוי. התפלגויות שעות ומין/גיל בקורפוס המלא מופיעות בביקורת הקודמת; המדגם הנוכחי משווה זהויות שנבחרו בתנאים מפורשים.

## ההמלצה המעשית

1. להשתמש ב־ECAPA לביקורת גיוון ועקביות ולתור האזנה לקולות דומים/לא עקביים, לצד הבדיקות האקוסטיות והמטא־דאטה. לא להשתמש בו כמקור גיל/מין או כהוכחת כפילות.
2. לאזן את החשיפה לפי קבוצת קול ודובר, ולבדוק בנפרד איזון משפחות; גרסאות של אותו קול נשארות תלויות.
3. לשמור את התפלגות הקולות הרחבה בניסוי הראשי. אם רוצים לבחון זוגות קרובים ב־ECAPA, לבנות תת־ניסוי נפרד ולא לבחור את כל המאגר רק לפי השכן הקרוב.
4. לבנות זוגות אמיתי/מסונתז עם אותו טקסט ושיטת גבולות/ערוץ סימטרית, ואחריהם לחזור על אותן בדיקות. אין להסיק מהפיילוט בעל המשפט היחיד על הבדלי סינתזה נקיים מתוכן.
5. לבחור דוברים וקולות לכל חלוקה מראש; מקור, חיתוכים וצאצאי עיבוד נשארים יחד. לבדוק דליפות זהות, טקסט ומסמך.
6. לא להגדיר הצלחה כ״חפיפה מלאה של ההתפלגויות״: רוצים כיסוי הוגן של קולות ותנאי שימוש, תוך שמירת מידע שעשוי להבחין בין סינתזה לדיבור אמיתי.

## שחזור ואימות

מודל: [SpeechBrain ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), לפי [הארכיטקטורה הרשמית](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/blob/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/hyperparams.yaml) וה־[מאמר המקורי](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf). משקולות, גרסה, hash וסביבת הריצה מתועדים בקבצי provenance. מספרי benchmark ממאמרים אינם תוצאות על המאגר שלנו.

תוצאות האימות: [verification.json](voice_distribution/verification.json). זהויות ושמות קבצים נשמרו עם hash; נתוני אודיו לא הועלו לשירות חיצוני. יומן עבודה: [VOICE_DISTRIBUTION_PROGRESS.md](VOICE_DISTRIBUTION_PROGRESS.md).
'''
    (ROOT/'VOICE_DISTRIBUTIONS_REPORT_HE.md').write_text(text,encoding='utf-8')
    print('Wrote Hebrew multi-method results report')

if __name__=='__main__':main()
