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
    page_title="Assistant For Detection Of Retinal Diseases",
    initial_sidebar_state="collapsed",
    layout="wide" # تم تفعيل الوضع العريض ليتناسب مع الأعمدة الثلاثة
)

# ==============================
# Custom CSS (Light & Bright Theme)
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* الخلفية العامة مضيئة */
    .stApp {
        background: #f8fafc;
        color: #1e293b;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 4rem; max-width: 1400px; }

    /* ── Hero (Light Version) ── */
    .hero {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        overflow: hidden;
        background: #ffffff;
        border-radius: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%, rgba(56,189,248,0.1) 0%, transparent 70%),
            radial-gradient(ellipse 40% 30% at 20% 80%, rgba(129,140,248,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.25em;
        text-transform: uppercase; color: #0284c7; margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 3.5rem);
        font-weight: 800; line-height: 1.1;
        letter-spacing: -0.02em; color: #0f172a; margin: 0 0 1rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle {
        font-size: 1.1rem; font-weight: 400; color: #475569;
        max-width: 600px; margin: 0 auto; line-height: 1.7;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent);
        margin: 0 0 2.5rem;
    }

    /* ── Disease List Column (Left) ── */
    .disease-list-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    .disease-list-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem; font-weight: 700;
        color: #0f172a; margin-bottom: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0f2fe;
    }
    .disease-item {
        display: flex; align-items: center; gap: 0.75rem;
        padding: 0.75rem 0.5rem;
        border-radius: 10px;
        transition: background 0.2s;
        border-bottom: 1px solid #f1f5f9;
    }
    .disease-item:last-child { border-bottom: none; }
    .disease-item:hover { background: #f0f9ff; }
    .disease-item-icon { font-size: 1.2rem; width: 25px; text-align: center; }
    .disease-item-name { font-size: 0.9rem; color: #334155; font-weight: 500; }

    /* ── Upload & Cards (Light) ── */
    .upload-section {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 20px; padding: 2.5rem 2rem;
        text-align: center; margin-bottom: 2rem;
        transition: border-color 0.2s;
    }
    .upload-section:hover { border-color: #38bdf8; background: #f0f9ff; }
    .upload-label {
        font-family: 'Syne', sans-serif; font-size: 1.1rem;
        font-weight: 600; color: #0f172a; margin-bottom: 0.4rem;
    }
    .upload-hint { font-size: 0.85rem; color: #64748b; }

    /* تعديل لـ Streamlit Uploader ليتناسب مع الواجهة المضيئة */
    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploader"] label { color: #0284c7 !important; font-size: 0.9rem; font-weight:500; }
    [data-testid="stFileUploader"] button { background-color: #0ea5e9 !important; color: white !important; border: none !important;}
    [data-testid="stFileUploader"] button:hover { background-color: #0284c7 !important; }


    .img-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.75rem;
        text-align: center;
        max-width: 240px;
        margin: 0 auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .img-card-label {
        font-size: 0.7rem; font-weight: 600;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #64748b; margin-top: 0.6rem;
    }

    [data-testid="stImage"] img {
        border-radius: 10px;
        width: 100%;
        max-height: 220px;
        object-fit: cover;
    }

    /* ── Results Section (Light) ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
    }
    .stProgress > div > div {
        background: #e2e8f0 !important;
        height: 10px !important;
    }

    .confidence-label {
        font-size: 0.8rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: #64748b; margin-bottom: 0.3rem; font-weight:500;
    }
    .confidence-value {
        font-family: 'Syne', sans-serif; font-size: 2.8rem;
        font-weight: 800; color: #0f172a; line-height: 1;
    }
    .confidence-value span { font-size: 1.2rem; font-weight: 500; color: #64748b; }

    .disease-card {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-top: 1rem;
    }
    .disease-card-title {
        font-family: 'Syne', sans-serif; font-size: 1rem;
        font-weight: 700; color: #0284c7; margin-bottom: 0.5rem;
    }
    .disease-card-text { font-size: 0.9rem; color: #1e293b; line-height: 1.7; }

    .llm-card {
        background: #f5f3ff;
        border: 1px solid #ddd6fe;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .llm-card-title {
        font-family: 'Syne', sans-serif; font-size: 1rem;
        font-weight: 700; color: #4f46e5; margin-bottom: 0.75rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .llm-line {
        font-size: 0.9rem; color: #1e293b;
        line-height: 1.7; margin-bottom: 0.5rem;
        padding-right: 0.75rem;
        border-right: 3px solid #ddd6fe;
    }
    .llm-error {
        font-size: 0.85rem; color: #9a3412;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-radius: 8px; padding: 0.7rem 1rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 12px; padding: 1rem 1.5rem;
        font-size: 0.85rem; color: #92400e;
        text-align: center; margin-top: 3rem; line-height: 1.6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Sidebar Light */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] label {
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Constants & Data
# ==============================
MODEL_PATH = "best_efficientnetb3.h5"
FILE_ID = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"
OLLAMA_URL = "http://localhost:11434/api/generate"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

class_names = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa'
]

disease_info = {
    "Diabetic Retinopathy": {"desc": "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري.", "action": "يُنصح بفحص دوري كل 6 أشهر ومراقبة السكر.", "icon": "🩺", "severity": "#e11d48"},
    "Disc Edema": {"desc": "تورم في القرص البصري قد يشير إلى ارتفاع ضغط الدم داخل الجمجمة.", "action": "يتطلب تقييمًا عصبيًا عاجلاً.", "icon": "🧠", "severity": "#e11d48"},
    "Healthy": {"desc": "لم يُكتشف أي مؤشر مرضي. تبدو شبكية العين سليمة.", "action": "حافظ على فحوصات دورية سنوية.", "icon": "✅", "severity": "#16a34a"},
    "Myopia": {"desc": "قِصَر النظر: صعوبة في رؤية الأشياء البعيدة بوضوح.", "action": "تصحيح بالنظارات أو العدسات أو الليزر.", "icon": "👓", "severity": "#ca8a04"},
    "Pterygium": {"desc": "نسيج ليفي وعائي ينمو على سطح القرنية.", "action": "استئصال جراحي إذا أثر على الرؤية.", "icon": "🔬", "severity": "#ca8a04"},
    "Retinal Detachment": {"desc": "انفصال الشبكية، وهو طارئ طبي يستوجب تدخلاً فوريًا.", "action": "توجّه فورًا إلى طوارئ العيون.", "icon": "🚨", "severity": "#dc2626"},
    "Retinitis Pigmentosa": {"desc": "اضطرابات وراثية تُسبب تدهورًا تدريجيًا في الرؤية.", "action": "التدبير يركز على تحسين جودة الحياة.", "icon": "🧬", "severity": "#e11d48"},
}

# ==============================
# LLM Logic (معدلة قليلاً لتناسب التنسيق الجديد)
# ==============================
PROMPT_TEMPLATE = """You are an ophthalmology AI assistant. Write exactly 5 short, distinct medical lines in Arabic about this eye disease prediction for a patient report.

Prediction: {disease}
Confidence: {confidence:.1f}%

Structure (5 Arabic lines only, no headers, no English):
1. حالة التشخيص الحالي ونسبة الثقة.
2. تعريف طبي مبسط جداً للمرض.
3. الأعراض الرئيسية المتوقعة.
4. مستوى الخطورة (بسيطة / متوسطة / شديدة / طارئة).
5. الخطوة التالية الموصى بها."""

def _clean_lines(text: str) -> str:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:5])

def local_llm_explain(disease: str, confidence: float, ollama_model: str, ollama_url: str, backend: str, anthropic_api_key: str) -> str:
    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    try:
        if backend == "claude":
            if not anthropic_api_key.strip(): return "ERROR: أدخل API Key الخاص بـ Claude."
            response = requests.post(ANTHROPIC_API_URL, headers={"x-api-key": anthropic_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                                     json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}, timeout=30)
            response.raise_for_status()
            raw = response.json()["content"][0]["text"].strip()
        else:
            payload = {"model": ollama_model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1, "num_ctx": 512}}
            response = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
        return _clean_lines(raw)
    except Exception as e: return f"ERROR: فشل الاتصال بالنموذج اللغوي. {e}"

# ==============================
# Model Loading & Processing (نفسا)
# ==============================
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل النموذج (مرة واحدة فقط)..."):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    try: return load_model(MODEL_PATH)
    except Exception as e: st.error(f"❌ فشل تحميل النموذج: {e}"); st.stop()

def predict(img, model):
    img_res = img.resize((300, 300))
    arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img_res))
    preds = model.predict(np.expand_dims(arr, axis=0))
    idx = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds)), preds[0]

def gradcam(img, model):
    img_res = img.resize((300, 300))
    arr = tf.keras.applications.efficientnet.preprocess_input(np.array(img_res))
    arr = np.expand_dims(arr, axis=0)
    target_layer = next((l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None)
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outs, preds = grad_model(arr)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_outs)
    grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
    cam = tf.reduce_sum(tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :] * conv_outs, axis=-1)[0].numpy()
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0: cam /= np.max(cam)
    cam = cv2.resize(np.power(cam, 0.3), (300, 300))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(img_res), 0.7, heatmap, 0.3, 0)
    return heatmap, overlay

# ==============================
# Hero Section
# ==============================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Ophthalmology Assistant</div>
    <h1 class="hero-title">نظام الذكاء الاصطناعي لكشف <span>أمراض الشبكية</span></h1>
    <p class="hero-subtitle">
        قم برفع صورة قاع العين (Fundus Image) للحصول على تحليل فوري ودقيق مدعوم بتقنيات التعلم العميق والخرائط الحرارية.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

model = load_model_cached()

# ==============================
# Sidebar (LLM Settings - Light)
# ==============================
with st.sidebar:
    st.markdown('<div style="font-family:\'Syne\',sans-serif; font-size:1.1rem; font-weight:700; color:#4f46e5; margin-bottom:1rem;">⚙️ إعدادات التقرير الذكي</div>', unsafe_allow_html=True)
    enable_llm = st.toggle("تفعيل شرح AI الطبي", value=True)
    llm_backend = st.radio("مزوّد النموذج اللغوي", options=["Ollama (محلي)", "Claude API"], index=0)
    backend_key = "ollama" if "Ollama" in llm_backend else "claude"
    if backend_key == "ollama":
        ollama_model = st.selectbox("نموذج Ollama", options=["llama3", "mistral", "phi3"], index=0)
        ollama_url = st.text_input("URL", value="http://localhost:11434")
        anthropic_api_key = ""
    else:
        ollama_model, ollama_url = "", ""
        anthropic_api_key = st.text_input("Anthropic API Key", type="password")

# ==============================
# MAIN LAYOUT (3 Columns)
# ==============================
# تم تغيير النسب لتصبح [عمود القائمة، عمود الرفع، عمود النتائج]
col_list, col_upload, col_result = st.columns([0.8, 1.2, 1.8], gap="medium")

# ── COLUMN 1: Disease List (New) ──
with col_list:
    st.markdown('<div class="disease-list-container">', unsafe_allow_html=True)
    st.markdown('<div class="disease-list-header">🔍 الأمراض المستهدفة</div>', unsafe_allow_html=True)

    for name in class_names:
        info = disease_info.get(name, {})
        st.markdown(f"""
        <div class="disease-item">
            <div class="disease-item-icon">{info.get('icon','•')}</div>
            <div class="disease-item-name">{name}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── COLUMN 2: Upload ──
with col_upload:
    st.markdown('<div class="upload-label">1. رفع صورة العين</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("اختر صورة قاع العين...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        # تصغير العرض ليتناسب مع العمود
        st.image(image, use_container_width=True)
        st.markdown('<div class="img-card-label">الصورة الأصلية</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-section">
            <div style="font-size:3rem; margin-bottom:1rem; opacity:0.5">👁️</div>
            <div class="upload-label">اسحب الصورة هنا</div>
            <div class="upload-hint">يدعم JPG, PNG</div>
        </div>
        """, unsafe_allow_html=True)


# ── COLUMN 3: Results ──
with col_result:
    if uploaded_file:
        with st.spinner("🔍 جاري تحليل الشبكية..."):
            pred, conf, all_preds = predict(image, model)
            heatmap, overlay = gradcam(image, model)

        info = disease_info.get(pred, {})
        color = info.get('severity', '#0ea5e9')

        st.markdown(f'<div class="upload-label">2. نتيجة التشخيص الآلي</div>', unsafe_allow_html=True)

        # Diagnosis Header
        st.markdown(f"""
        <div style="background:white; padding: 1.5rem; border-radius:15px; border:1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom:1rem;">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
                <span style="font-size:2.5rem;">{info.get('icon','🔬')}</span>
                <span style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:{color}; letter-spacing:-0.02em;">{pred}</span>
            </div>
            <div class="confidence-label">مستوى الثقة في الشبكة العصبية</div>
            <div class="confidence-value">{conf*100:.1f}<span>%</span></div>
            <div style="margin-top:0.5rem;">""", unsafe_allow_html=True)
        st.progress(int(conf * 100))
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Basic Info Card
        if info:
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-card-title">📋 توصيف مبدئي</div>
                <div class="disease-card-text">{info['desc']}</div>
                <div style="margin-top:0.7rem; padding-top:0.7rem; border-top:1px solid #bae6fd;">
                    <span style="font-size:0.85rem; color:#0284c7; font-weight:600;">الإجراء المقترح: </span>
                    <span class="disease-card-text">{info['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # LLM Explanation
        if enable_llm:
            with st.spinner("🤖 جاري صياغة التقرير الطبي الذكي..."):
                llm_result = local_llm_explain(pred, conf, ollama_model, ollama_url, backend_key, anthropic_api_key)

            st.markdown(f"""
            <div class="llm-card">
                <div class="llm-card-title">🤖 تقرير المدقق الطبي (AI)</div>""", unsafe_allow_html=True)

            if llm_result.startswith("ERROR:"):
                st.markdown(f'<div class="llm-error">⚠️ {llm_result.replace("ERROR:","")}</div>', unsafe_allow_html=True)
            else:
                for line in llm_result.split("\n"):
                    st.markdown(f'<div class="llm-line">{line}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM Visuals
        st.markdown('<br><div class="upload-label">3. التحليل البصري (تحديد مناطق الإصابة)</div>', unsafe_allow_html=True)
        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(heatmap, channels="BGR", use_container_width=True)
            st.markdown('<div class="img-card-label">خريطة التركيز (Heatmap)</div></div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(overlay, channels="BGR", use_container_width=True)
            st.markdown('<div class="img-card-label">الدمج مع الصورة الأصلية</div></div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:350px; background:white; border-radius:20px; border:1px solid #e2e8f0; color:#94a3b8; text-align:center; box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);">
            <div style="font-size:4rem; margin-bottom:1rem;">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:600;">في انتظار رفع الصورة</div>
            <div style="font-size:0.9rem;">سيظهر تحليل النتائج والتقرير هنا</div>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Disclaimer (Light)
# ==============================
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>تنبيه طبي هام:</strong> هذا النظام هو نموذج أولي للذكاء الاصطناعي مصمم لأغراض بحثية ومساعدة كأداة فحص أولي فقط.
    <strong>لا يعتبر</strong> هذا التحليل تشخيصاً طبياً نهائياً. يجب دائماً استشارة طبيب عيون متخصص ومؤهل لتشخيص وعلاج أي حالة مرضية تتعلق بالعين.
</div>
""", unsafe_allow_html=True)
