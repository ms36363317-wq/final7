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
# NEW CLEAN MEDICAL UI
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-main: #0f172a;
    --bg-card: #111827;
    --bg-soft: #1f2937;

    --primary: #3b82f6;
    --primary-light: #60a5fa;

    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;

    --text-main: #f9fafb;
    --text-secondary: #9ca3af;

    --border: rgba(255,255,255,0.08);
}

/* GENERAL */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

.stApp {
    background: var(--bg-main);
}

.block-container {
    max-width: 1200px;
}

/* NAVBAR */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.2rem 0;
    border-bottom: 1px solid var(--border);
}

.navbar-title {
    font-size: 1.4rem;
    font-weight: 600;
}

.navbar-title span {
    color: var(--primary);
}

/* CARDS */
.card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.4rem;
    border: 1px solid var(--border);
    margin-bottom: 1rem;
}

/* SECTION */
.section-title {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* RESULT */
.result-title {
    font-size: 1.6rem;
    font-weight: 600;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-size: 0.7rem;
    margin-top: 5px;
}

/* PROGRESS */
.progress {
    height: 6px;
    background: #1f2937;
    border-radius: 6px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 6px;
}

/* BUTTON */
.stButton > button {
    background: var(--primary);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    background: var(--primary-light);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--bg-soft);
}

/* IMAGE */
img {
    border-radius: 12px;
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
