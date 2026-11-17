"""Professional, source-grounded project summary; render every page for QA."""
from pathlib import Path
import sys,json,hashlib,html
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'.pdf_deps'))
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT,TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing,Rect,String,Line
from bidi.algorithm import get_display
import pandas as pd

OUT=ROOT/'output/pdf'; TMP=ROOT/'tmp/pdfs'
OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
PATH=OUT/'HebrewDeepFake_Synthetic_Data_Generation_Report.pdf'
pdfmetrics.registerFont(TTFont('Arial','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialBold','C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
NAVY=colors.HexColor('#142B45'); TEAL=colors.HexColor('#007E87'); INK=colors.HexColor('#243448'); MUTED=colors.HexColor('#526278'); LIGHT=colors.HexColor('#EEF4F7'); GOLD=colors.HexColor('#AD6D19')
WIDTH,HEIGHT=A4; W=WIDTH-92
styles={
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.2,leading=14.4,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.6,leading=11.8,textColor=MUTED,spaceAfter=6),
 'h1':ParagraphStyle('h1',fontName='ArialBold',fontSize=23,leading=28,textColor=NAVY,spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='ArialBold',fontSize=13,leading=17,textColor=TEAL,spaceBefore=10,spaceAfter=7),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=9,leading=12,textColor=INK),
 'head':ParagraphStyle('head',fontName='ArialBold',fontSize=8.7,leading=11.5,textColor=colors.white),
 'he':ParagraphStyle('he',fontName='Arial',fontSize=13,leading=20,alignment=TA_RIGHT,textColor=NAVY,spaceAfter=8),
}
story=[]; page_titles=[]
def p(text,style='body'):return Paragraph(text,styles[style])
def add(text,style='body'):story.append(p(text,style))
def title(number,text,subtitle):
    if story:story.append(PageBreak())
    page_titles.append(text)
    add(f'PROJECT REPORT / {number:02d}', 'small'); add(text,'h1');add(subtitle,'body')
def h(text):add(text,'h2')
def bullet(text):
    story.append(Paragraph(text,ParagraphStyle('bullet',parent=styles['body'],leftIndent=10,firstLineIndent=-8),bulletText='-'))
def table(headers,rows,widths):
    rows=[[p(html.escape(str(c)),'head') for c in headers]]+[[p(str(c),'cell') for c in row] for row in rows]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),('LINEBELOW',(0,0),(-1,0),.5,NAVY),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#DCE4EB'))]))
    story.append(t);story.append(Spacer(1,9))
def callout(label,text):
    t=Table([[p(label,'h2')],[p(text)]],colWidths=[W])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('BOX',(0,0),(-1,-1),.5,colors.HexColor('#C9DEE2')),('LEFTPADDING',(0,0),(-1,-1),13),('RIGHTPADDING',(0,0),(-1,-1),13),('TOPPADDING',(0,0),(-1,0),3),('BOTTOMPADDING',(0,-1),(-1,-1),8)]))
    story.append(t);story.append(Spacer(1,10))
def fig(path,height):
    from PIL import Image as PILImage
    iw,ih=PILImage.open(path).size
    width=min(W,height*iw/ih); story.append(Image(str(path),width=width,height=width*ih/iw));story.append(Spacer(1,6))

syn=pd.read_csv(ROOT/'synthetic_pilot/analysis/sample_features.csv')
metrics=json.loads((ROOT/'synthetic_pilot/analysis/key_metrics.json').read_text())
billing=json.loads((ROOT/'synthetic_pilot/billing_summary.json').read_text())
model=syn.groupby(['family','model']).agg(voices=('voice_identity_group','nunique'),recordings=('sample_id','size'),seconds=('duration_s','sum')).reset_index()
family=syn.groupby('family').agg(voices=('voice_identity_group','nunique'),recordings=('sample_id','size'),seconds=('duration_s','sum')).reset_index()
ecapa=json.loads((ROOT/'voice_distribution/key_metrics.json').read_text())

title(1,'Synthetic Hebrew speech\nGeneration & readiness summary','HebrewDeepFake project | Pilot generated 16 November 2026 | Analysis through 17 November 2026')
add('Purpose: document what was generated, how each recording can be traced and checked, and what is needed before a fair real-versus-synthetic experiment.','body')
table(['Recordings','Nominal voice groups','Model variants','Audio duration'],[['<b>284</b>','<b>116</b>','<b>10 across 4 families</b>','<b>22.11 minutes</b>']],[W*.19,W*.25,W*.29,W*.27])
h('The result in plain language')
add('We created one short Hebrew recording for each selected model-and-voice combination. A <b>model</b> is the speech-generation system; a <b>voice</b> is a named stock voice offered by its provider. The same voice identifier can appear in several model versions, so 284 files do not mean 284 independent speakers.')
callout('Current status: a useful pilot, not a finished benchmark','All 284 recordings passed the final automated technical and Hebrew/text checks. None has a completed human verification label in the current synthetic records. Seven recordings are flagged for listening. The pilot is suitable for voice inventory, pipeline checks and calibration; it is not yet a representative 20-hour corpus or a clean real/fake evaluation set.')
h('What is complete')
bullet('Generation, original-file preservation, standardized analysis copies, structured metadata, automated review, executed exploratory notebooks and duplicate/imbalance audits.')
bullet('A local ECAPA voice analysis, comparing the synthetic pilot with multiple excerpts from real speakers; no new speech was generated for that analysis.')
h('What is still required')
bullet('Human listening, reliable perceptual labels, more Hebrew texts, consistent boundary/channel processing and frozen identity/text-aware evaluation splits.')
add('Reported request cost: <b>$0.82524</b> for speech generation plus automated audio review. The authorized project budget was $20; this pilot did not generate the planned maximum of 20 audio hours. Future volume and prices require a new estimate.','small')
add('Report guide: inventory p. 2 | process p. 3 | metadata p. 4 | quality p. 5 | imbalance p. 6 | real/fake comparison p. 7 | voice analysis p. 8 | next steps p. 9 | costs and references p. 10.','small')

title(2,'Inventory: what was generated','The following names are model slugs recorded in the generation manifest, not a claim about current provider availability.')
rows=[]
for r in model.itertuples():
    # Keep provider names separate and wrap slugs only at punctuation.
    slug=r.model.split('/',1)[1].replace('-','-<wbr/>')
    rows.append([r.family,slug,str(r.voices),'1',str(r.recordings),f'{r.seconds/60:.2f}'])
table(['Provider','Recorded model slug suffix','Voices','Clips / voice','Clips','Minutes'],rows,[71,210,53,61,43,W-438])
add('Use the provider prefix plus the suffix above to reconstruct the full slug, for example <b>google/gemini-3.8-flash-tts</b>. Displayed minutes are rounded; totals use unrounded durations.','small')
h('Why there are 116 voice groups, not 284 speakers')
add('Google contributes 30 nominal voices used across three versions; ElevenLabs 21 across four; MiniMax 45 across two; and Speechify 20 in one. The total is 30 + 21 + 45 + 20 = <b>116 provider:voice groups</b>. Shared identifiers are grouped conservatively; they do not prove biometric identity or perfect consistency across versions.')
h('How many physical files?')
table(['Item','Count','Meaning'],[['Generated original audio','284','194 MP3 files + 90 raw PCM files'],['Standardized analysis WAV','284','Copies of the same recordings, not new speech'],['Per-recording JSON sidecars','284','Saved generation/request metadata'],['Core subtotal','852','568 audio files + 284 JSON sidecars']],[142,52,W-194])
add('CSV manifests, shared configuration/catalog snapshots, review logs, notebooks and derived analysis files are additional artifacts. Real-reference excerpts and model weights are not synthetic generation files.','small')

title(3,'Generation process and safeguards','A reproducible pilot: one fixed text, stock voices and explicit request records.')
h('The shared Hebrew prompt')
add(get_display('היום ניפגש בשעה שלוש ליד תחנת הרכבת בירושלים.'),'he')
add('Meaning: "Today we will meet at three o\'clock near the train station in Jerusalem." The Hebrew prompt contains eight whitespace-separated words. Using the same text makes first-pass voice comparisons easier, but does not test general Hebrew pronunciation or content diversity.')
table(['Step','What was done','Why it matters'],[
['1. Define scope','One short sample per selected stock model/voice; target language Hebrew. No voice cloning.','Keeps the pilot bounded and avoids presenting catalog voices as replicas of real participants.'],
['2. Freeze configuration','Saved model catalog, voice metadata, prompt, request settings and planned sample IDs.','Makes later changes and repeated runs traceable.'],
['3. Generate','Used OpenRouter with an OpenAI-compatible client; recorded model, voice, generation ID and status.','Links each result to the request that produced it.'],
['4. Preserve and decode','Retained original MP3/PCM and created mono 16 kHz, 16-bit PCM WAV analysis copies.','Supports common measurements without overwriting originals.'],
['5. Review automatically','A separate audio model, openai/gpt-audio-mini, transcribed the clip without seeing the expected text or voice identity.','Reduces direct prompting toward the expected answer.'],
['6. Audit and document','Measured audio features, checked transcripts and hashes, recorded costs, executed notebooks and built listening queues.','Provides evidence and exposes unresolved quality issues.'],
],[91,229,W-320])
add('Generation automatic retries were disabled. During audio review, four attempts failed because of an upstream response-format incompatibility; the request was repaired and explicitly retried. The history retains 288 review attempts: 284 successful final reviews and four failures. A failed review request is not a failed speech-generation recording.','small')
add('Format terms: mono means one audio channel; 16 kHz means 16,000 sound samples per second; 16-bit PCM stores numerical sample values directly. A codec is the method used to encode or compress audio. The WAV copy provides a common analysis format, while preserving the original reveals its encoding history.','small')

title(4,'Metadata: what accompanies each recording','Metadata is information about a recording. Some fields describe how it was made; others are measurements or unverified estimates.')
table(['Category and example fields','What it tells us','How it helps'],[
['<b>Identity and grouping</b><br/>sample_id, model, canonical_model, family, voice, voice_identity_group','Which configuration produced the clip and which versions share a nominal voice.','Reproducibility, balanced sampling and preventing the same voice from leaking across evaluation splits.'],
['<b>Text and language</b><br/>text_id, text, requested_language','The intended Hebrew content.','Text-matched pairs and comparison of what was requested with what was heard.'],
['<b>Request provenance</b><br/>attempted_utc, generation_id, generation_status, format, pcm_sample_rate','When and how the request was made and whether its output decoded.','Troubleshooting, request reconciliation and explaining missing/failed results.'],
['<b>Files and fingerprints</b><br/>original_path, wav_path, bytes, original_sha256, wav_sha256','Where the files are stored and their digital fingerprints.','Integrity checks and exact-file duplicate screening. A hash does not detect re-encoded or trimmed copies.'],
['<b>Audio measurements</b><br/>duration_s, source_codec, source_sample_rate, channels, rms_dbfs, peak, quiet_fraction_45dbfs','Duration, format, average signal level (RMS), largest sample (peak) and an energy-based quiet fraction.','Find technical differences and review outliers. These are not human quality ratings or reliable noise-to-speech estimates.'],
['<b>Automated review</b><br/>transcript, language, speech_intelligibility, normalized_character_error_rate, review_status','What a separate model heard and how closely its transcript matches the prompt.','Screen Hebrew/content problems. The score is model-generated, not calibrated human accuracy.'],
['<b>Labels and their source</b><br/>provider_gender_label, provider_age_label, provider_label_source, presentation_confidence','Provider casting descriptors or model-estimated vocal presentation, with provenance.','Coverage auditing, while keeping unknown and conflicting labels visible. Not biological sex or actual age.'],
['<b>QC, cost and review history</b><br/>technical_qc_pass, language_qc_pass, human_verified, review_reasons; request-cost ledger','Which checks passed, what remains unverified, and reported charges.','Acceptance decisions, listening queues and budget accountability.']
],[W*.40,W*.29,W*.31])
add('These fields are joined by sample_id across manifests, review tables and analysis tables. They are not all embedded inside the audio or duplicated in every JSON sidecar. ECAPA vectors are separate derived analysis artifacts, also linked by sample ID and WAV hash.','small')

title(5,'Quality evidence and label limitations','A passed automatic check means the configured test passed. It is not a guarantee of naturalness, correct identity or representative data.')
table(['Check','Observed result','Interpretation'],[
['Technical decoding','284 / 284 passed','The saved recordings decode and pass the configured technical checks.'],
['Automated Hebrew/text check','284 / 284 passed','Final review met the screening rules; listening is still necessary.'],
['Human verification','0 / 284 complete','Do not describe the pilot as human-validated.'],
['Outlier listening queue','7 recordings','Flags are review priorities, not automatic rejection decisions.'],
['Exact PCM duplicates','0 found in the saved comparison set','A later check reopened all 284 synthetic WAVs and 78 real-reference WAVs and hashed their sample payloads.'],
['Voice-presentation consistency','35 / 116 nominal groups conflicting','An unvalidated audio reviewer assigned different presentations across model versions.'],
['Automatic age estimates','One category for all 284 clips','Rejected as uninformative; not used for age balancing.'],
],[150,120,W-270])
add('Final language screening required the reviewer to identify Hebrew, at least 80% Hebrew letters among letters, normalized character error at most 0.25, and a model-estimated intelligibility score of at least 0.7. These are operational thresholds, not validated universal standards.','small')
h('What is known about voice labels?')
add('Only the 21 ElevenLabs nominal voices have mapped provider descriptors: 13 male, 7 female and 1 neutral; age descriptors are 13 middle_aged, 7 young and 1 old. The other <b>95 groups are unknown</b>. These source casting labels were linked using provider metadata; their mappings should still be audited.')
callout('Keep facts, measurements and estimates separate','A synthetic voice has no measured biological age. Human birth-year metadata, catalog casting labels and a listener\'s impression are different concepts. None should silently replace another. Rejected raw age outputs remain available for audit, but are excluded from meaningful age summaries.')

title(6,'Imbalance: what we addressed','Data imbalance means that some speakers, voices, models or conditions contribute much more information than others.')
table(['Family','Nominal voices','Recordings','Share of clips','Share of audio time'],[[r.family,str(r.voices),str(r.recordings),f'{100*r.recordings/284:.1f}%',f'{100*r.seconds/family.seconds.sum():.1f}%'] for r in family.itertuples()],[95,91,80,110,W-376])
h('Completed: account for dependent recordings')
add('A Google voice repeated across three model versions is one nominal group with three observations. The analysis uses one representative center or median per voice group rather than treating every file as a new speaker. This reduces over-weighting of voices with more versions.')
h('Completed: test different weighting choices')
add('We compared equal weight per voice with equal total weight per provider family. We also compared families using 20 candidate voices each, repeated over 200 subsets, and checked one clip per identity in 50 repetitions. These are analysis controls; they do not change the generated catalog or make the underlying data balanced.')
h('Completed: expose real-corpus concentration')
add('In the real corpus, the ten most prolific speakers contribute about 47.7% of recorded time. The initial 78-clip reference and later 80-speaker sample target identity coverage, not the original distribution of hours. Both units are reported separately.')
h('Still needed: balance generation and validated labels')
bullet('Speechify contributes only 7.0% of clips and 6.4% of time. A future controlled batch needs explicit family and voice exposure targets, measured both in clips and minutes.')
bullet('Demographic matching cannot be completed with 95 unknown provider labels and an unusable automatic age estimate. Human vocal-presentation labels need a common rubric in both groups.')
bullet('Do not force every variable to 50/50. First define whether the goal is a controlled comparison or a deployment-like test, then balance or weight accordingly.')
add('No files were deleted or duplicated to achieve balance. The original generation remains unchanged; balanced future sampling is a recommendation, not a completed intervention.','small')

title(7,'Real-versus-synthetic comparison','The main risk is a shortcut: a detector can learn an accidental preparation difference instead of evidence of synthesis.')
table(['Issue','What we observed','Why it matters'],[
['Text','All synthetic clips use one sentence; no exact normalized real-segment match was found.','Content could reveal the class. The pilot cannot support a text-disjoint split across synthetic examples.'],
['Original encoding','Real recordings are Opus/AAC; synthetic originals are MP3/PCM.','Saving both as WAV does not erase previous compression.'],
['Clip boundaries','Real excerpts follow transcript alignment; synthetic files contain complete generated output.','Edge silence can almost separate the selected groups by itself.'],
['Identity exposure','85 real source IDs versus 116 nominal synthetic voice groups; versions repeat voices.','File counts and speaker counts answer different questions.'],
['Scope','Real source is read speech, not spontaneous conversation.','The combined data cannot be called representative of all Hebrew speech.'],
],[91,205,W-296])
add('The earlier 78-speaker reference had median durations of 3.67 s versus 4.49 s for synthetic voice-group medians. Median quiet-edge share was 0% versus 9.46%. A later real sample targeted durations near 4.5 s. These are selected-sample findings, not full-corpus population estimates.')
h('Duplicate text is different from duplicate audio')
add('The broader audit found 1,151 repeated real-segment text groups. Among eligible segments with at least six normalized words, 259 groups repeated and 145 spanned multiple speakers. These may be valid different readings. Keep them, but prevent prohibited text overlap between evaluation splits. A full search for almost-identical re-encoded or trimmed audio has not been completed.')
callout('Research informs the checks, not our result numbers','ASVspoof 2021 explicitly studies coding/transmission and compressed-audio conditions [1]. Smeu et al. show that leading silence can become a shortcut in other deepfake datasets [2]. These findings motivate our controls; their published benchmark results are not results on this Hebrew pilot.')

title(8,'Voice diversity: what ECAPA adds','ECAPA turns a short recording into a numerical voice representation, called an embedding. We use it to compare voices, not to decide whether a clip is fake [3, 4].')
add('The local analysis used <b>231 excerpts from 80 real speakers</b> plus all <b>284 synthetic clips from 116 nominal groups</b>. It extracted 192-dimensional vectors for the original and capped-edge-trimmed views: 1,030 vectors in total. Five real source IDs had no eligible excerpt under the sample rules and remain in the coverage audit.')
fig(ROOT/'voice_distribution/figures/speaker_projections.png',205)
add('Figure 1. Shared PCA and t-SNE projections, one center per nominal identity. Colors identify real speech or the generating family. The two PCA axes retain only 12.4% of total variance. Visual separation is not proof of a synthesis-specific signal.','small')
table(['Method','Question answered'],[
['Within/between-voice cosine similarity','Are repeated observations of a nominal voice more alike than different voices?'],
['Nearest-neighbor coverage','Which real voices lack a close synthetic neighbor in this representation?'],
['Kernel distance (MMD)','How different are the observed voice distributions under a specified similarity rule?'],
['PCA / t-SNE and acoustic features','Do multiple views reveal clusters, condition effects or review candidates?'],
],[177,W-177])
add('The real same-ID versus different-ID ranking check yielded descriptive AUC 0.9967; with positive pairs restricted to different sessions, 0.9963 over 200 positive pairs from 69 speakers. AUC measures ranking separation here. It is <b>not deepfake detection accuracy</b>, a validated identity threshold or a held-out performance estimate.')
add('We flagged similar different-name synthetic voices and inconsistent same-name model versions for listening. Similarity alone does not prove duplication. Text, phonetic content, codec and averaging choices still affect comparisons; no age or sex labels were inferred from ECAPA.','small')

title(9,'Remaining work and next-stage specification','Improve the small controlled experiment before expanding to many hours.')
table(['Step','Recommended action','Evidence required before proceeding'],[
['1','<b>Listen.</b> Review the seven flagged clips; cover nominal voices and unstable model-version pairs.','Accept / review / reject decisions with reasons; uncertainty remains visible.'],
['2','<b>Validate cuts.</b> Check text and word boundaries in proposed real excerpts.','No empty or out-of-bounds cuts; audible text agrees with the paired prompt.'],
['3','<b>Make text pairs.</b> Generate the same Hebrew text as each accepted real excerpt, using stock voices.','A pair shares text, not necessarily speaker identity. Measure actual new durations and costs.'],
['4','<b>Fix partitions.</b> Assign real speakers, synthetic voice groups and text groups before synthesis.','No prohibited identity, source-derived copy or text overlap across train/development/test.'],
['5','<b>Standardize conditions.</b> Use symmetric boundary, level-sensitivity and channel/codec procedures.','Condition coverage is comparable in both classes; raw files remain immutable.'],
['6','<b>Expand carefully.</b> Add texts and voices under bounded exposure targets; reserve an unseen family test.','Report performance by voice/family/condition with support counts, not only one overall score.'],
],[39,223,W-262])
add('Training data teach a detector; development data help choose settings; test data are kept separate for the final assessment. A benchmark is a fixed set and protocol for that assessment.','small')
h('The prepared proposal: 308 pairs, not yet generated')
add('The proposal contains 308 globally exact-text-unique real utterances from 81 speakers, up to four each, provisionally 77 texts per synthesis family. It uses 2-10 s and 6-14 words with source/alignment scores at least 0.8. This differs from the 80-speaker ECAPA sample, which required 3-8 s. Models, voices and final splits remain unassigned.')
add('The proposal contains about 17.10 minutes of real excerpts. Synthetic duration is not assumed equal. It still needs listening and near-text/document leakage checks. Exact-text uniqueness alone does not rule out similar passages.')
callout('Readiness decision','Retain the present pilot for calibration and voice auditing. Build a small, human-checked, same-text paired batch next. Keep a balanced diagnostic test separate from a test designed around the eventual use case, such as telephone audio or voice messages. No new generation is implied by this report.')

title(10,'Cost, reproducibility and references','Project measurements come from saved manifests and executed notebooks. Paper citations support methods and cautions, not local generation counts.')
table(['Reported item','USD'],[['284 speech-generation requests',f'{billing["reported_speech_cost_usd"]:.6f}'],['284 completed automatic audio reviews',f'{billing["reported_review_cost_usd"]:.7f}'],['Total reported request cost',f'{billing["reported_request_cost_sum_usd"]:.7f}']],[W-110,110])
add('Four failed review attempts remain in the audit history. The cost table follows the per-request cost ledger; account usage snapshots lagged and fluctuated. It is not a new invoice reconciliation or a forecast for 20 hours. The later ECAPA/distribution analysis made no paid API calls. Credentials are not included in this report or data artifacts.','small')
h('Where the evidence lives in the project')
table(['Artifact','Role'],[
['synthetic_pilot/manifest.csv and audio/','Generation metadata, original audio, analysis WAVs and JSON sidecars.'],
['automated_review.csv; review_attempt_history.jsonl','Final review records plus retained failure/retry history.'],
['synthetic_eda.ipynb; real_synthetic_comparison.ipynb','Executed pilot and real/synthetic comparison analyses.'],
['quality_matching_audit.ipynb; voice_distributions.ipynb','Executed duplicate/representation and ECAPA multi-view analyses.'],
['comparison_outputs/; quality_audit/; voice_distribution/','Tables, plots, input/model hashes, verification and proposed pair manifests.'],
],[223,W-223])
h('References')
references=[
('[1]','Delgado, H., et al. (2021). <i>ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan.</i> Version 0.4, 16 July 2021.','https://www.asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf'),
('[2]','Smeu, S., Boldisor, D.-A., Oneata, D., and Oneata, E. (2025). <i>Circumventing Shortcuts in Audio-visual Deepfake Detection Datasets with Unsupervised Learning.</i> CVPR, pp. 18815-18825.','https://openaccess.thecvf.com/content/CVPR2025/html/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.html'),
('[3]','Desplanques, B., Thienpondt, J., and Demuynck, K. (2020). <i>ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification.</i> Interspeech. DOI: 10.21437/Interspeech.2020-2650.','https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.pdf'),
('[4]','SpeechBrain. <i>spkrec-ecapa-voxceleb</i>, official model card and configuration. Pinned revision: 0f99f2d0ebe89ac095bcc5903c4dd8f72b367286. Accessed 17 November 2026.','https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'),
]
for label,description,url in references:
    add(f'{label} {description} <link href="{url}" color="#007E87">Open source</link>.','small')

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.states=[]
    def showPage(self):self.states.append(dict(self.__dict__));self._startPage()
    def save(self):
        total=len(self.states)
        for state in self.states:
            self.__dict__.update(state);self.draw_page_number(total);super().showPage()
        super().save()
    def draw_page_number(self,total):
        self.setStrokeColor(colors.HexColor('#DCE4EB'));self.line(46,42,WIDTH-46,42)
        self.setFont('Arial',8);self.setFillColor(MUTED)
        self.drawString(46,28,'HEBREWDEEPFAKE | SYNTHETIC DATA PILOT | 17 NOV 2026')
        self.drawRightString(WIDTH-46,28,f'{self._pageNumber} / {total}')
        self.setStrokeColor(TEAL);self.setLineWidth(3);self.line(46,HEIGHT-32,WIDTH-46,HEIGHT-32)

doc=SimpleDocTemplate(str(PATH),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=49,bottomMargin=59,title='HebrewDeepFake - Synthetic Data Generation and Readiness',author='HebrewDeepFake project',subject='Pilot inventory, metadata, quality, imbalance and next-stage protocol')
doc.build(story,canvasmaker=NumberedCanvas)

# Render with PDFium because system/bundled Poppler executables are unavailable.
import pypdfium2 as pdfium
pdf=pdfium.PdfDocument(str(PATH))
renders=[]
for i in range(len(pdf)):
    image=pdf[i].render(scale=1.4).to_pil()
    path=TMP/f'page_{i+1:02d}.png';image.save(path);renders.append(str(path))
from pypdf import PdfReader
reader=PdfReader(str(PATH)); text='\n'.join(page.extract_text() for page in reader.pages)
(TMP/'extracted_report.txt').write_text(text,encoding='utf-8')
check={'pdf':str(PATH),'pages':len(reader.pages),'bytes':PATH.stat().st_size,'sha256':hashlib.sha256(PATH.read_bytes()).hexdigest(),'rendered_pages':renders,'expected_pages':10,'page_text_lengths':[len(page.extract_text()) for page in reader.pages],'no_secret_pattern':not any(s in text for s in ['sk-or-v1-','OPENROUTER_API_KEY=']),'notes':'Every rendered page requires visual QA before delivery. Rasterizer fallback: PDFium; Poppler not present.'}
assert len(reader.pages)==10,check['page_text_lengths']
assert check['no_secret_pattern']
assert all(n>300 for n in check['page_text_lengths'])
(OUT/'report_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in check.items() if k!='rendered_pages'},indent=2))
