import os
import requests
import streamlit as st
import numpy as np
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="RetinalAI — Ocular Disease Detection",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Premium CSS — Medical Luxury Dark
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

:root {
    --bg-base:       #04070f;
    --bg-surface:    #080e1a;
    --bg-card:       #0c1424;
    --bg-card-alt:   #0f1829;
    --border:        rgba(255,255,255,0.06);
    --border-bright: rgba(255,255,255,0.12);
    --accent-cyan:   #00c8e0;
    --accent-blue:   #2563eb;
    --accent-violet: #7c3aed;
    --text-primary:  #e8f0fe;
    --text-secondary:#8899bb;
    --text-muted:    #445577;
    --success:       #10b981;
    --warning:       #f59e0b;
    --danger:        #ef4444;
    --emergency:     #dc2626;
    --font-main:     'Outfit', sans-serif;
    --font-display:  'Playfair Display', serif;
    --font-mono:     'JetBrains Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    color: var(--text-primary);
}
.stApp {
    background: var(--bg-base);
    background-image:
        radial-gradient(ellipse 120% 50% at 50% -10%, rgba(0,200,224,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(37,99,235,0.05) 0%, transparent 50%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.012'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 5rem; max-width: 1300px; margin: 0 auto; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 2px; }

/* ── NAVBAR ── */
.navbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1.1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.navbar-brand { display: flex; align-items: center; gap: 0.75rem; }
.navbar-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    box-shadow: 0 0 20px rgba(0,200,224,0.25);
}
.navbar-name {
    font-family: var(--font-display);
    font-size: 1.3rem; font-weight: 700;
    color: var(--text-primary); letter-spacing: -0.01em;
}
.navbar-name span { color: var(--accent-cyan); }
.navbar-badge {
    font-family: var(--font-mono); font-size: 0.6rem;
    color: var(--accent-cyan);
    background: rgba(0,200,224,0.07);
    border: 1px solid rgba(0,200,224,0.18);
    padding: 0.2rem 0.55rem; border-radius: 999px;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.navbar-tags { display: flex; gap: 0.45rem; align-items: center; }
.navbar-tag {
    font-size: 0.68rem; color: var(--text-muted);
    background: var(--bg-card); border: 1px solid var(--border);
    padding: 0.22rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em;
}

/* ── SECTION LABELS ── */
.section-label {
    font-family: var(--font-mono); font-size: 0.6rem; font-weight: 500;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.6rem;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: var(--border);
}

/* ── UPLOAD PANEL ── */
.upload-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px; padding: 1.75rem 1.5rem;
    position: relative; overflow: hidden;
}
.upload-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
}
.upload-drop-inner {
    border: 1.5px dashed rgba(0,200,224,0.18);
    border-radius: 12px; padding: 2rem 1rem;
    text-align: center; background: rgba(0,200,224,0.02);
}
.upload-icon-wrap {
    width: 52px; height: 52px; margin: 0 auto 0.9rem;
    background: linear-gradient(135deg, rgba(0,200,224,0.08), rgba(37,99,235,0.08));
    border: 1px solid rgba(0,200,224,0.12); border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
}
.upload-title {
    font-size: 0.88rem; font-weight: 600;
    color: var(--text-primary); margin-bottom: 0.25rem;
}
.upload-hint { font-size: 0.72rem; color: var(--text-muted); }

[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] > div { border: none !important; background: transparent !important; padding: 0 !important; }
[data-testid="stFileUploader"] section { padding: 0 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: 1.5px dashed rgba(0,200,224,0.18) !important;
    border-radius: 12px !important;
}

/* ── IMAGE PREVIEW ── */
.img-preview-wrap {
    background: var(--bg-card-alt); border: 1px solid var(--border);
    border-radius: 14px; padding: 0.7rem;
}
.img-preview-label {
    font-family: var(--font-mono); font-size: 0.58rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-muted); margin-top: 0.55rem; text-align: center;
}
[data-testid="stImage"] img { border-radius: 10px !important; width: 100% !important; }

/* ── STAT CHIPS ── */
.stat-chips { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-top: 0.9rem; }
.stat-chip {
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.6rem 0.75rem;
}
.stat-chip-label {
    font-size: 0.58rem; color: var(--text-muted);
    letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.18rem;
}
.stat-chip-value {
    font-family: var(--font-mono); font-size: 0.82rem;
    font-weight: 600; color: var(--accent-cyan);
}

/* ── RESULTS PANEL ── */
.results-panel {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 18px; padding: 1.75rem 1.6rem;
    position: relative; overflow: hidden; margin-bottom: 1.2rem;
}
.results-panel::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
}
.diag-hero {
    display: flex; align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 1.4rem; padding-bottom: 1.4rem;
    border-bottom: 1px solid var(--border);
}
.diag-left { flex: 1; }
.diag-tag {
    font-family: var(--font-mono); font-size: 0.58rem;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.4rem;
}
.diag-name {
    font-family: var(--font-display); font-size: 1.75rem;
    font-weight: 700; line-height: 1.1;
    letter-spacing: -0.01em; margin-bottom: 0.3rem;
}
.diag-severity {
    font-family: var(--font-mono); font-size: 0.62rem;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-muted);
}
.diag-icon-wrap {
    width: 60px; height: 60px;
    background: var(--bg-card-alt); border: 1px solid var(--border-bright);
    border-radius: 16px; display: flex; align-items: center;
    justify-content: center; font-size: 1.9rem; flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
}
.conf-row {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 0.45rem;
}
.conf-label {
    font-family: var(--font-mono); font-size: 0.62rem;
    letter-spacing: 0.15em; text-transform: uppercase; color: var(--text-muted);
}
.conf-value {
    font-family: var(--font-mono); font-size: 0.82rem;
    font-weight: 600; color: var(--text-primary);
}
.conf-bar-bg {
    height: 5px; border-radius: 999px;
    background: rgba(255,255,255,0.05); overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 999px;
    box-shadow: 0 0 10px rgba(0,200,224,0.35);
}

/* ── INFO CARD ── */
.info-card {
    background: var(--bg-card-alt); border: 1px solid var(--border);
    border-radius: 14px; padding: 1.2rem 1.35rem;
    margin-bottom: 1rem; position: relative; overflow: hidden;
}
.info-card::before {
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px; border-radius: 0 2px 2px 0;
}
.info-card.cyan::before  { background: var(--accent-cyan); }
.info-card.violet::before { background: var(--accent-violet); }
.info-card-header {
    font-family: var(--font-mono); font-size: 0.6rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 0.55rem;
}
.info-card-body { font-size: 0.82rem; color: var(--text-secondary); line-height: 1.7; }
.rec-pill {
    display: inline-block; margin-top: 0.65rem;
    font-size: 0.73rem; color: var(--accent-cyan);
    background: rgba(0,200,224,0.07);
    border: 1px solid rgba(0,200,224,0.16);
    padding: 0.28rem 0.75rem; border-radius: 999px;
}

/* ── LLM CARD ── */
.llm-card {
    background: var(--bg-card-alt);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 14px; padding: 1.2rem 1.35rem;
    margin-bottom: 1rem; position: relative; overflow: hidden;
}
.llm-card::before {
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    border-radius: 0 2px 2px 0; background: var(--accent-violet);
}
.llm-header {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 0.85rem;
}
.llm-title {
    font-family: var(--font-mono); font-size: 0.6rem;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent-violet); display: flex; align-items: center; gap: 0.4rem;
}
.llm-model-badge {
    font-family: var(--font-mono); font-size: 0.58rem; color: var(--text-muted);
    background: var(--bg-surface); border: 1px solid var(--border);
    padding: 0.12rem 0.45rem; border-radius: 4px;
}
.llm-line {
    display: flex; gap: 0.7rem; align-items: flex-start;
    padding: 0.42rem 0; border-bottom: 1px solid var(--border);
    font-size: 0.81rem; color: var(--text-secondary); line-height: 1.65;
}
.llm-line:last-child { border-bottom: none; padding-bottom: 0; }
.llm-num {
    font-family: var(--font-mono); font-size: 0.62rem;
    font-weight: 600; color: var(--accent-violet);
    opacity: 0.6; min-width: 18px; margin-top: 0.15rem; flex-shrink: 0;
}
.llm-error {
    font-size: 0.78rem; color: var(--warning);
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.14);
    border-radius: 8px; padding: 0.75rem 1rem;
    font-family: var(--font-mono);
}

/* ── GRADCAM ── */
.gradcam-card {
    background: var(--bg-card-alt); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.6rem 0.6rem 0.45rem; text-align: center;
}
.gradcam-label {
    font-family: var(--font-mono); font-size: 0.56rem;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--text-muted); margin-top: 0.45rem;
}

/* ── PROB BARS ── */
.prob-row {
    display: flex; align-items: center;
    gap: 0.75rem; margin-bottom: 0.5rem;
}
.prob-name {
    font-size: 0.74rem; color: var(--text-secondary);
    white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis; min-width: 155px;
}
.prob-bar-bg {
    flex: 1; height: 5px; border-radius: 999px;
    background: rgba(255,255,255,0.04); overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 999px; }
.prob-pct {
    font-family: var(--font-mono); font-size: 0.68rem;
    color: var(--text-muted); min-width: 40px; text-align: right;
}

/* ── EMPTY STATE ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 420px; text-align: center; opacity: 0.2;
}
.empty-state-icon { font-size: 3.5rem; margin-bottom: 1rem; filter: grayscale(1); }
.empty-state-text {
    font-family: var(--font-display); font-size: 1.1rem; color: var(--text-secondary);
}
.empty-state-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.35rem; }

/* ── PROGRESS (Streamlit) ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
    border-radius: 999px !important;
}
.stProgress > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 999px !important; height: 5px !important;
}

/* ── DISCLAIMER ── */
.disclaimer {
    display: flex; align-items: center; gap: 1rem;
    background: rgba(245,158,11,0.04);
    border: 1px solid rgba(245,158,11,0.12);
    border-radius: 12px; padding: 1rem 1.4rem; margin-top: 3rem;
}
.disclaimer-icon { font-size: 1.1rem; flex-shrink: 0; }
.disclaimer-text { font-size: 0.76rem; color: #7a5e22; line-height: 1.6; }
.disclaimer-text strong { color: #a8802a; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stButton > button {
    background: rgba(0,200,224,0.06) !important;
    border: 1px solid rgba(0,200,224,0.2) !important;
    color: var(--accent-cyan) !important;
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,200,224,0.12) !important;
    border-color: rgba(0,200,224,0.4) !important;
    box-shadow: 0 0 14px rgba(0,200,224,0.12) !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Constants
# ==============================
MODEL_PATH        = "best_efficientnetb3.h5"
FILE_ID           = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# ==============================
# Disease Info
# ==============================
disease_info = {
    "Diabetic Retinopathy": {
        "desc":   "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري. يُعدّ من الأسباب الرئيسية للعمى لدى البالغين.",
        "action": "يُنصح بفحص دوري كل 6 أشهر ومراقبة مستوى السكر في الدم.",
        "icon": "🩺", "severity": "danger"
    },
    "Disc Edema": {
        "desc":   "تورم في القرص البصري قد يشير إلى ارتفاع ضغط الدم داخل الجمجمة أو اضطرابات عصبية.",
        "action": "يتطلب تقييمًا عصبيًا عاجلاً وصور أشعة للدماغ.",
        "icon": "🧠", "severity": "danger"
    },
    "Healthy": {
        "desc":   "لم يُكتشف أي مؤشر مرضي. تبدو شبكية العين سليمة وبحالة جيدة.",
        "action": "حافظ على فحوصات دورية سنوية للعين للاطمئنان على صحتها.",
        "icon": "✅", "severity": "success"
    },
    "Myopia": {
        "desc":   "قِصَر النظر: صعوبة في رؤية الأشياء البعيدة بوضوح بسبب طول محور مقلة العين.",
        "action": "يمكن تصحيحه بالنظارات أو العدسات اللاصقة أو جراحة الليزر.",
        "icon": "👓", "severity": "warning"
    },
    "Pterygium": {
        "desc":   "نسيج ليفي وعائي ينمو على سطح القرنية من الملتحمة، وقد يؤثر على الرؤية.",
        "action": "قد يحتاج إلى استئصال جراحي إذا تقدّم نحو مركز القرنية.",
        "icon": "🔬", "severity": "warning"
    },
    "Retinal Detachment": {
        "desc":   "انفصال الشبكية عن طبقة الظهارة الصباغية، وهو طارئ طبي يستوجب تدخلاً فوريًا.",
        "action": "توجّه فورًا إلى أقرب طوارئ عيون — يمكن أن يؤدي التأخير إلى فقدان البصر نهائيًا.",
        "icon": "🚨", "severity": "emergency"
    },
    "Retinitis Pigmentosa": {
        "desc":   "مجموعة اضطرابات وراثية تُسبب تدهورًا تدريجيًا في خلايا الشبكية المستقبلة للضوء.",
        "action": "لا يوجد علاج شافٍ حتى الآن؛ التدبير يركز على إبطاء التقدم وتحسين جودة الحياة.",
        "icon": "🧬", "severity": "danger"
    },
}

severity_color = {
    "success":   "#10b981",
    "warning":   "#f59e0b",
    "danger":    "#ef4444",
    "emergency": "#dc2626",
}

# ==============================
# LLM
# ==============================
PROMPT_TEMPLATE = """You are an ophthalmology AI assistant.

Write exactly 5 short medical lines about this eye disease prediction:

Prediction: {disease}
Confidence: {confidence:.1f}%

Structure (5 lines only, no headers, no repetition):
1. Prediction statement.
2. Short clinical definition.
3. Key symptoms the patient may notice.
4. Severity level (Mild / Moderate / Severe / Emergency).
5. Recommended next step."""


def _clean_lines(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:5])


def _explain_via_ollama(disease, confidence, ollama_model, ollama_url):
    prompt  = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    payload = {
        "model": ollama_model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 120,
                    "repeat_penalty": 1.2, "num_ctx": 512},
    }
    r = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return _clean_lines(r.json().get("response", "").strip())


def _explain_via_claude(disease, confidence, api_key):
    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    r = requests.post(
        ANTHROPIC_API_URL,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 300,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return _clean_lines(r.json()["content"][0]["text"].strip())


def _test_ollama(url):
    try:
        r = requests.get(url.rstrip("/"), timeout=5)
        return (True, "✅ Ollama يعمل بنجاح") if r.status_code == 200 \
               else (False, f"⚠️ HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        return False, "❌ تعذّر الاتصال — شغّل: ollama serve"
    except Exception as e:
        return False, f"❌ {e}"


def local_llm_explain(disease, confidence, ollama_model="llama3",
                      ollama_url="http://localhost:11434",
                      backend="ollama", anthropic_api_key=""):
    try:
        if backend == "claude":
            if not anthropic_api_key.strip():
                return "ERROR: أدخل Anthropic API Key في الإعدادات."
            return _explain_via_claude(disease, confidence, anthropic_api_key.strip())
        return _explain_via_ollama(disease, confidence, ollama_model, ollama_url)
    except requests.exceptions.ConnectionError:
        return "ERROR: تعذّر الاتصال — " + (
            "شغّل: ollama serve" if backend == "ollama" else "تحقق من الإنترنت")
    except requests.exceptions.Timeout:
        return "ERROR: انتهت المهلة — جرّب نموذجاً أصغر مثل phi3"
    except requests.exceptions.HTTPError as e:
        s = e.response.status_code if e.response is not None else "?"
        if s == 401: return "ERROR: API Key غير صالح"
        if s == 404 and backend == "ollama":
            return f"ERROR: نفّذ: ollama pull {ollama_model}"
        return f"ERROR: HTTP {s}"
    except Exception as e:
        return f"ERROR: {e}"

# ==============================
# Vision Model
# ==============================
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل النموذج..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}",
                           MODEL_PATH, quiet=False)
    if not os.path.exists(MODEL_PATH):
        st.error("❌ النموذج غير موجود"); st.stop()
    if os.path.getsize(MODEL_PATH) < 5_000_000:
        st.error("❌ ملف النموذج تالف"); st.stop()
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {e}"); st.stop()

class_names = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa'
]

def preprocess(img):
    img = img.resize((300, 300))
    arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img))
    return np.expand_dims(arr, axis=0)

def predict(img, mdl):
    preds = mdl.predict(preprocess(img))
    idx   = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds)), preds[0]

def overlay_heatmap(img, heatmap):
    return cv2.addWeighted(np.array(img.resize((300, 300))), 0.75, heatmap, 0.25, 0)

def gradcam(img, mdl):
    arr = tf.keras.applications.efficientnet.preprocess_input(
        np.array(img.resize((300, 300))))
    arr = np.expand_dims(arr, axis=0)
    tgt = next((l for l in reversed(mdl.layers)
                if isinstance(l, tf.keras.layers.Conv2D)), None)
    gm  = tf.keras.models.Model(inputs=mdl.inputs,
                                  outputs=[tgt.output, mdl.output])
    with tf.GradientTape() as tape:
        co, pr = gm(arr)
        if isinstance(pr, list): pr = pr[0]
        loss = pr[:, 0] if pr.shape[-1] == 1 \
               else pr[:, tf.argmax(pr[0]).numpy()]
    grads   = tape.gradient(loss, co)
    grads   = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam     = tf.reduce_sum(weights[:, None, None, :] * co, axis=-1)[0].numpy()
    cam     = np.maximum(cam, 0)
    if np.max(cam) > 0: cam /= np.max(cam)
    cam = cv2.resize(np.power(cam, 0.3), (300, 300))
    return cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# ──────────────────────────────
model = load_model_cached()

# ══════════════════════════════
# SIDEBAR
# ══════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1.5rem;">
        <div style="font-size:2rem; margin-bottom:0.4rem;">👁</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.15rem;
                    font-weight:700; color:#e8f0fe; letter-spacing:-0.01em;">
            Retinal<span style="color:#00c8e0;">AI</span>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.56rem;
                    color:#445577; letter-spacing:0.15em; text-transform:uppercase;
                    margin-top:0.2rem;">v2.0 · EfficientNet-B3</div>
    </div>
    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.06); margin-bottom:1.2rem;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.58rem;
                letter-spacing:0.2em; text-transform:uppercase; color:#445577;
                margin-bottom:0.75rem;">⚙ LLM Settings</div>
    """, unsafe_allow_html=True)

    enable_llm  = st.toggle("تفعيل الشرح الطبي بالذكاء الاصطناعي", value=True)
    llm_backend = st.radio(
        "مزوّد النموذج",
        options=["🖥  Ollama — محلي", "☁  Claude API — سحابي"],
        index=0
    )
    backend_key = "ollama" if "Ollama" in llm_backend else "claude"

    if backend_key == "ollama":
        ollama_model = st.selectbox(
            "النموذج",
            ["phi3", "llama3", "mistral", "gemma", "llama2", "neural-chat"],
            index=0, help="phi3 — الأخف والأسرع"
        )
        ollama_url        = st.text_input("Ollama URL", "http://localhost:11434")
        anthropic_api_key = ""
        if st.button("🔌 اختبار الاتصال", use_container_width=True):
            ok, msg = _test_ollama(ollama_url)
            st.success(msg) if ok else st.error(msg)
        st.markdown("""
        <div style="margin-top:0.9rem; font-family:'JetBrains Mono',monospace;
                    font-size:0.62rem; color:#223344; line-height:2.2;">
            <span style="color:#00c8e0;">$</span> ollama serve<br>
            <span style="color:#00c8e0;">$</span> ollama pull phi3
        </div>
        """, unsafe_allow_html=True)
    else:
        ollama_model      = "llama3"
        ollama_url        = "http://localhost:11434"
        anthropic_api_key = st.text_input(
            "Anthropic API Key", type="password", placeholder="sk-ant-api03-...")
        st.markdown("""
        <div style="margin-top:0.7rem; font-size:0.7rem; color:#445577; line-height:1.8;">
            Model: <code style="background:rgba(124,58,237,0.1); color:#a78bfa;
            padding:0.1rem 0.4rem; border-radius:4px;">claude-haiku</code><br>
            <a href="https://console.anthropic.com" target="_blank"
               style="color:#00c8e0; text-decoration:none; font-size:0.68rem;">
                ↗ console.anthropic.com</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.06); margin:1.5rem 0 1rem;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.56rem;
                color:#1e3040; text-align:center; line-height:1.9;">
        EfficientNet-B3 · Grad-CAM<br>7 Retinal Disease Classes<br>
        Built with Streamlit + TensorFlow
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════
# NAVBAR
# ══════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">👁</div>
        <div class="navbar-name">Retinal<span>AI</span></div>
        <div class="navbar-badge">Medical AI</div>
    </div>
    <div class="navbar-tags">
        <div class="navbar-tag">EfficientNet-B3</div>
        <div class="navbar-tag">Grad-CAM</div>
        <div class="navbar-tag">7 Classes</div>
        <div class="navbar-tag">LLM Analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════
# MAIN COLUMNS
# ══════════════════════════════
left_col, right_col = st.columns([1, 1.7], gap="large")

# ── LEFT ──────────────────────
with left_col:
    st.markdown('<div class="section-label">01 &nbsp;·&nbsp; Image Input</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if not uploaded_file:
        st.markdown("""
        <div class="upload-drop-inner">
            <div class="upload-icon-wrap">👁️</div>
            <div class="upload-title">رفع صورة قاع العين</div>
            <div class="upload-hint">JPG · PNG · JPEG &nbsp;|&nbsp; اسحب وأفلت أو انقر</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        w, h  = image.size

        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">02 &nbsp;·&nbsp; Preview</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="img-preview-wrap">', unsafe_allow_html=True)
        preview = image.copy(); preview.thumbnail((420, 420))
        st.image(preview, use_container_width=True)
        st.markdown('<div class="img-preview-label">Original Fundus Image</div></div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-chips">
            <div class="stat-chip">
                <div class="stat-chip-label">Resolution</div>
                <div class="stat-chip-value">{w}×{h}</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-label">Format</div>
                <div class="stat-chip-value">{uploaded_file.type.split('/')[-1].upper()}</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-label">Size</div>
                <div class="stat-chip-value">{uploaded_file.size // 1024} KB</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-label">Backbone</div>
                <div class="stat-chip-value">EffNet-B3</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT ─────────────────────
with right_col:
    if uploaded_file:
        with st.spinner("جاري التحليل..."):
            pred, conf, all_preds = predict(image, model)
            heatmap = gradcam(image, model)
            overlay = overlay_heatmap(image, heatmap)

        info     = disease_info.get(pred, {})
        sev      = info.get("severity", "warning")
        color    = severity_color.get(sev, "#f59e0b")
        conf_pct = conf * 100

        # ── Diagnosis ──
        st.markdown('<div class="section-label">03 &nbsp;·&nbsp; Diagnosis</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="results-panel">
            <div class="diag-hero">
                <div class="diag-left">
                    <div class="diag-tag">Primary Diagnosis</div>
                    <div class="diag-name" style="color:{color};">{pred}</div>
                    <div class="diag-severity">
                        Severity &nbsp;·&nbsp;
                        <span style="color:{color};">{sev.upper()}</span>
                    </div>
                </div>
                <div class="diag-icon-wrap">{info.get('icon','🔬')}</div>
            </div>
            <div class="conf-row">
                <div class="conf-label">Confidence Score</div>
                <div class="conf-value">{conf_pct:.1f}%</div>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill"
                     style="width:{conf_pct:.1f}%;
                            background:linear-gradient(90deg,{color}99,{color});">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Clinical Info ──
        if info:
            st.markdown('<div class="section-label">04 &nbsp;·&nbsp; Clinical Notes</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card cyan">
                <div class="info-card-header">📋 &nbsp; About this condition</div>
                <div class="info-card-body">{info['desc']}</div>
                <div class="rec-pill">↗ &nbsp;{info['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── LLM ──
        if enable_llm:
            backend_label = "Claude Haiku" if backend_key == "claude" else ollama_model
            with st.spinner(f"جاري توليد الشرح عبر {backend_label}..."):
                llm_result = local_llm_explain(
                    pred, conf, ollama_model=ollama_model, ollama_url=ollama_url,
                    backend=backend_key, anthropic_api_key=anthropic_api_key)

            st.markdown('<div class="section-label">05 &nbsp;·&nbsp; AI Medical Analysis</div>',
                        unsafe_allow_html=True)

            if llm_result.startswith("ERROR:"):
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-header">
                        <div class="llm-title">🤖 &nbsp; LLM Analysis</div>
                        <div class="llm-model-badge">{backend_label}</div>
                    </div>
                    <div class="llm-error">⚠ &nbsp;{llm_result.replace('ERROR:','').strip()}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                lines = [l.strip() for l in llm_result.split("\n") if l.strip()]
                rows  = "".join(
                    f'<div class="llm-line"><div class="llm-num">0{i+1}</div>'
                    f'<div>{line}</div></div>'
                    for i, line in enumerate(lines)
                )
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-header">
                        <div class="llm-title">🤖 &nbsp; LLM Analysis</div>
                        <div class="llm-model-badge">{backend_label}</div>
                    </div>
                    {rows}
                </div>
                """, unsafe_allow_html=True)

        # ── Grad-CAM ──
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">06 &nbsp;·&nbsp; Grad-CAM Visualization</div>',
                    unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown('<div class="gradcam-card">', unsafe_allow_html=True)
            st.image(heatmap, use_container_width=True, channels="BGR")
            st.markdown('<div class="gradcam-label">Activation Heat Map</div></div>',
                        unsafe_allow_html=True)
        with g2:
            st.markdown('<div class="gradcam-card">', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True, channels="BGR")
            st.markdown('<div class="gradcam-label">Overlay · Original + CAM</div></div>',
                        unsafe_allow_html=True)

        # ── Probabilities ──
        st.markdown('<br>', unsafe_allow_html=True)
        with st.expander("📊 &nbsp; Class Probabilities — All 7 Classes"):
            st.markdown('<br>', unsafe_allow_html=True)
            for i in np.argsort(all_preds)[::-1]:
                pct     = float(all_preds[i]) * 100
                is_pred = class_names[i] == pred
                bc      = color if is_pred else "rgba(255,255,255,0.12)"
                nc      = "#e8f0fe" if is_pred else "#8899bb"
                fw      = "font-weight:600;" if is_pred else ""
                glow    = f"box-shadow:0 0 8px {color}55;" if is_pred else ""
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-name" style="color:{nc};{fw}">{class_names[i]}</div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill"
                             style="width:{pct:.1f}%;background:{bc};{glow}"></div>
                    </div>
                    <div class="prob-pct" style="{'color:'+color+';' if is_pred else ''}">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔬</div>
            <div class="empty-state-text">في انتظار صورة للتحليل</div>
            <div class="empty-state-sub">ارفع صورة قاع العين من اليسار للبدء</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════
# DISCLAIMER
# ══════════════════════════════
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-icon">⚠️</div>
    <div class="disclaimer-text">
        <strong>تنبيه طبي هام:</strong>
        هذا النظام أداةٌ مساعدة للفحص الأولي فقط ولا يُغني بأي حال عن استشارة طبيب متخصص.
        يُرجى مراجعة طبيب عيون معتمد للحصول على التشخيص النهائي والخطة العلاجية المناسبة.
    </div>
</div>
""", unsafe_allow_html=True)
