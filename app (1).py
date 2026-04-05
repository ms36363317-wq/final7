import os
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
    page_title="👁️ Eye Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS — Light Theme (like app1)
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp { background: #f7f9fc; color: #1a1a2e; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 4rem; max-width: 1200px; }

    /* ── Header ── */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.2rem;
        padding-top: 2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* ── Section Header ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a73e8;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── Upload ── */
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #c5d8f6 !important;
        border-radius: 12px !important;
        background: #eef4ff !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"] label { color: #1a73e8 !important; font-weight: 500; }

    /* ── Image ── */
    [data-testid="stImage"] img {
        border-radius: 12px;
        width: 100%;
        object-fit: cover;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* ── Disease Badge ── */
    .disease-badge {
        display: inline-block;
        background: #1a73e8;
        color: white;
        padding: 6px 20px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    /* ── Confidence ── */
    .confidence-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1;
    }
    .confidence-value span {
        font-size: 1rem;
        font-weight: 400;
        color: #888;
    }
    .confidence-label {
        font-size: 0.82rem;
        color: #888;
        margin-bottom: 0.3rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: #1a73e8 !important;
        border-radius: 999px !important;
    }
    .stProgress > div > div {
        background: #e0eaff !important;
        border-radius: 999px !important;
        height: 8px !important;
    }

    /* ── Disease Info Card ── */
    .disease-card {
        background: #f0f4ff;
        border-left: 4px solid #1a73e8;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        font-size: 0.93rem;
        line-height: 1.75;
        color: #333;
    }
    .disease-card-title {
        font-weight: 600;
        color: #1a73e8;
        margin-bottom: 0.3rem;
        font-size: 0.95rem;
    }
    .disease-card-text { color: #444; }

    /* ── Warning Box ── */
    .disclaimer {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-size: 0.84rem;
        color: #7a5c00;
        margin-top: 2rem;
        line-height: 1.6;
    }

    /* ── Card wrapper ── */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1rem;
    }

    /* ── Prob bar label ── */
    .prob-label {
        font-size: 0.88rem;
        color: #444;
        margin-bottom: 3px;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #e0eaff !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: #1a73e8 !important; font-weight: 500; }

    /* ── Img card label ── */
    .img-card-label {
        font-size: 0.75rem;
        color: #888;
        text-align: center;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e8edf5; }
</style>
""", unsafe_allow_html=True)

# ==============================
# Constants
# ==============================
MODEL_PATH = "model.h5"
FILE_ID    = "11tjmQJITN0zHQ7x2wMPOF9L1JWnoZTxQ"

# ==============================
# Disease Info
# ==============================
disease_info = {
    "Diabetic Retinopathy": {
        "desc":   "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري. يُعدّ من الأسباب الرئيسية للعمى لدى البالغين.",
        "action": "يُنصح بفحص دوري كل 6 أشهر ومراقبة مستوى السكر في الدم.",
        "icon": "🩺", "color": "#e74c3c"
    },
    "Disc Edema": {
        "desc":   "تورم في القرص البصري قد يشير إلى ارتفاع ضغط الدم داخل الجمجمة أو اضطرابات عصبية.",
        "action": "يتطلب تقييمًا عصبيًا عاجلاً وصور أشعة للدماغ.",
        "icon": "🧠", "color": "#e74c3c"
    },
    "Healthy": {
        "desc":   "لم يُكتشف أي مؤشر مرضي. تبدو شبكية العين سليمة وبحالة جيدة.",
        "action": "حافظ على فحوصات دورية سنوية للعين للاطمئنان على صحتها.",
        "icon": "✅", "color": "#27ae60"
    },
    "Myopia": {
        "desc":   "قِصَر النظر: صعوبة في رؤية الأشياء البعيدة بوضوح بسبب طول محور مقلة العين.",
        "action": "يمكن تصحيحه بالنظارات أو العدسات اللاصقة أو جراحة الليزر.",
        "icon": "👓", "color": "#f39c12"
    },
    "Pterygium": {
        "desc":   "نسيج ليفي وعائي ينمو على سطح القرنية من الملتحمة، وقد يؤثر على الرؤية.",
        "action": "قد يحتاج إلى استئصال جراحي إذا تقدّم نحو مركز القرنية.",
        "icon": "🔬", "color": "#f39c12"
    },
    "Retinal Detachment": {
        "desc":   "انفصال الشبكية عن طبقة الظهارة الصباغية، وهو طارئ طبي يستوجب تدخلاً فوريًا.",
        "action": "توجّه فورًا إلى أقرب طوارئ عيون — التأخير قد يؤدي لفقدان البصر نهائيًا.",
        "icon": "🚨", "color": "#c0392b"
    },
    "Retinitis Pigmentosa": {
        "desc":   "مجموعة اضطرابات وراثية تُسبب تدهورًا تدريجيًا في خلايا الشبكية المستقبلة للضوء.",
        "action": "لا يوجد علاج شافٍ حتى الآن؛ التدبير يركز على إبطاء التقدم وتحسين جودة الحياة.",
        "icon": "🧬", "color": "#e74c3c"
    },
}

# ==============================
# Load Model
# ==============================
@st.cache_resource(show_spinner=False)
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل النموذج..."):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH, quiet=False
            )
    if not os.path.exists(MODEL_PATH):
        st.error("❌ النموذج غير موجود"); st.stop()
    if os.path.getsize(MODEL_PATH) < 5_000_000:
        st.error("❌ ملف النموذج تالف"); st.stop()
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {e}"); st.stop()

# ==============================
# Classes
# ==============================
class_names = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa'
]

# ==============================
# Helpers
# ==============================
def preprocess(img):
    img = img.resize((300, 300))
    arr = np.array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(img, model):
    preds = model.predict(preprocess(img), verbose=0)
    idx   = np.argmax(preds)
    return class_names[idx], float(np.max(preds)), preds[0]

def gradcam(img, model):
    arr = np.array(img.resize((300, 300)))
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    target_layer = next(
        (l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)), None
    )
    if target_layer is None:
        raise ValueError("No Conv2D layer found")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads   = tape.gradient(loss, conv_outputs)
    grads   = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam     = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)[0].numpy()
    cam     = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam /= np.max(cam)
    cam = np.power(cam, 0.3)
    cam = cv2.resize(cam, (300, 300))
    return cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

def overlay_heatmap(img, heatmap):
    arr = np.array(img.resize((300, 300)))
    return cv2.addWeighted(arr, 0.6, heatmap, 0.4, 0)

# ==============================
# Header
# ==============================
st.markdown('<div class="main-title">👁️ Eye Disease AI Diagnosis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload a retinal fundus image for AI-powered disease classification and visual explanation</div>',
    unsafe_allow_html=True
)

# ==============================
# Load Model
# ==============================
model = load_model_cached()

# ==============================
# Upload
# ==============================
uploaded_file = st.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================
# Main
# ==============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Analyzing image…"):
        pred, conf, all_preds = predict(image, model)
        heatmap = gradcam(image, model)
        overlay = overlay_heatmap(image, heatmap)

    info  = disease_info.get(pred, {})
    color = info.get("color", "#1a73e8")

    # ── Row 1: Image | Diagnosis ──────────────────────────────────────────
    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown('<div class="section-header">📷 Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">🔬 Diagnosis Result</div>', unsafe_allow_html=True)

        # Badge
        st.markdown(
            f'<div class="disease-badge" style="background:{color};">'
            f'{info.get("icon","🔬")} {pred}</div>',
            unsafe_allow_html=True
        )

        # Confidence
        st.markdown('<div class="confidence-label">Confidence Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-value">{conf*100:.1f}<span>%</span></div>', unsafe_allow_html=True)
        st.progress(int(conf * 100))

        # Disease info card
        if info:
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-card-title">📋 About this condition</div>
                <div class="disease-card-text">{info['desc']}</div>
                <div style="margin-top:0.6rem; padding-top:0.6rem;
                            border-top:1px solid #d0dff8;">
                    <span style="color:#1a73e8; font-weight:600; font-size:0.85rem;">Recommendation: </span>
                    <span class="disease-card-text">{info['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── All Probabilities ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 All Class Probabilities</div>', unsafe_allow_html=True)
    with st.expander("Show all probabilities"):
        for name, prob in sorted(zip(class_names, all_preds), key=lambda x: -x[1]):
            pct = float(prob) * 100
            bar_color = color if name == pred else "#dde8ff"
            label_color = color if name == pred else "#555"
            st.markdown(
                f'<div class="prob-label" style="color:{label_color}; font-weight:{"600" if name==pred else "400"};">'
                f'{name} — {pct:.1f}%</div>',
                unsafe_allow_html=True
            )
            st.progress(float(prob))

    # ── Grad-CAM ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔥 Grad-CAM Activation Map</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.image(image, use_container_width=True)
        st.markdown('<div class="img-card-label">Original</div>', unsafe_allow_html=True)
    with c2:
        st.image(heatmap, use_container_width=True, channels="BGR")
        st.markdown('<div class="img-card-label">Heatmap</div>', unsafe_allow_html=True)
    with c3:
        st.image(overlay, use_container_width=True, channels="BGR")
        st.markdown('<div class="img-card-label">Overlay</div>', unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research purposes only and is
        not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified ophthalmologist.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a retinal fundus image to begin diagnosis.")
