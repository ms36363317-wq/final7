import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="👁️ Eye Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── Background ── */
    .stApp { background: #060a10; color: #e8edf5; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 4rem; max-width: 1300px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #080c14 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #c8d8ea !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Syne', sans-serif !important;
        color: #38bdf8 !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
    [data-testid="stSidebar"] .stToggle label { color: #c8d8ea !important; }

    /* ── Hero ── */
    .hero {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; inset: 0;
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,180,255,0.12) 0%, transparent 70%),
            radial-gradient(ellipse 40% 30% at 20% 80%, rgba(0,80,200,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.75rem; font-weight: 500; letter-spacing: 0.25em;
        text-transform: uppercase; color: #38bdf8; margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800; line-height: 1.05;
        letter-spacing: -0.02em; color: #f0f6ff; margin: 0 0 1rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle {
        font-size: 1rem; font-weight: 300; color: #8ba3bf;
        max-width: 560px; margin: 0 auto; line-height: 1.7;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
        margin: 0 0 2.5rem;
    }

    /* ── Section Header ── */
    .section-header {
        font-family: 'Syne', sans-serif;
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: #5a7a96; margin-bottom: 0.75rem; margin-top: 0.25rem;
    }

    /* ── Upload ── */
    .upload-section {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(56,189,248,0.25);
        border-radius: 20px; padding: 2.5rem 2rem;
        text-align: center; margin-bottom: 1rem;
    }
    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploader"] > div {
        border: none !important; background: transparent !important; padding: 0 !important;
    }
    [data-testid="stFileUploader"] label { color: #38bdf8 !important; font-size: 0.9rem; }

    /* ── Image Card ── */
    .img-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.75rem 0.75rem 0.5rem;
        text-align: center;
    }
    .img-card-label {
        font-size: 0.68rem; font-weight: 500;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: #5a7a96; margin-top: 0.6rem; display: block;
    }
    [data-testid="stImage"] img {
        border-radius: 10px; width: 100%;
        max-height: 240px; object-fit: cover;
    }

    /* ── Diagnosis ── */
    .diag-label {
        font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
        color: #5a7a96; margin-bottom: 0.6rem;
    }
    .diag-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.7rem; font-weight: 800; letter-spacing: -0.01em;
    }
    .confidence-label {
        font-size: 0.78rem; letter-spacing: 0.15em;
        text-transform: uppercase; color: #5a7a96; margin-bottom: 0.4rem; margin-top: 1rem;
    }
    .confidence-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.6rem; font-weight: 800; color: #f0f6ff; line-height: 1;
    }
    .confidence-value span { font-size: 1rem; font-weight: 400; color: #5a7a96; }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
        border-radius: 999px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 999px !important; height: 8px !important;
    }

    /* ── Disease Card ── */
    .disease-card {
        background: rgba(56,189,248,0.05);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 14px; padding: 1.1rem 1.3rem; margin-top: 1rem;
    }
    .disease-card-title {
        font-family: 'Syne', sans-serif; font-size: 0.9rem;
        font-weight: 700; color: #38bdf8; margin-bottom: 0.4rem;
    }
    .disease-card-text { font-size: 0.84rem; color: #8ba3bf; line-height: 1.65; }

    /* ── Report Box ── */
    .report-box {
        background: rgba(56,189,248,0.04);
        border-left: 3px solid #38bdf8;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        font-size: 0.9rem; line-height: 1.9; color: #c8d8ea;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(245,158,11,0.07);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 12px; padding: 0.9rem 1.2rem;
        font-size: 0.78rem; color: #92762e;
        text-align: center; margin-top: 2rem; line-height: 1.6;
    }

    /* ── Info box (no image uploaded) ── */
    .stInfo { background: rgba(56,189,248,0.07) !important; border-color: rgba(56,189,248,0.2) !important; }
    .stInfo p { color: #8ba3bf !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary { color: #8ba3bf !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa',
]
IMG_SIZE = (300, 300)
LAST_CONV_LAYER = "top_conv"

DISEASE_INFO = {
    "Diabetic Retinopathy": {"desc": "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري.", "action": "فحص دوري كل 6 أشهر ومراقبة مستوى السكر.", "icon": "🩺"},
    "Disc Edema":           {"desc": "تورم في القرص البصري قد يشير إلى ضغط داخل الجمجمة.", "action": "تقييم عصبي عاجل وصور أشعة للدماغ.", "icon": "🧠"},
    "Healthy":              {"desc": "لم يُكتشف أي مؤشر مرضي. شبكية العين سليمة.", "action": "فحوصات دورية سنوية للاطمئنان.", "icon": "✅"},
    "Myopia":               {"desc": "قِصَر النظر نتيجة طول محور مقلة العين.", "action": "نظارات أو عدسات لاصقة أو جراحة ليزر.", "icon": "👓"},
    "Pterygium":            {"desc": "نسيج ليفي وعائي ينمو على سطح القرنية.", "action": "استئصال جراحي إذا تقدّم نحو مركز القرنية.", "icon": "🔬"},
    "Retinal Detachment":   {"desc": "انفصال الشبكية — طارئ طبي يستوجب تدخلاً فورياً.", "action": "توجّه فوراً لأقرب طوارئ عيون.", "icon": "🚨"},
    "Retinitis Pigmentosa": {"desc": "اضطرابات وراثية تُسبب تدهوراً تدريجياً في خلايا الشبكية.", "action": "إبطاء التقدم وتحسين جودة الحياة.", "icon": "🧬"},
}
SEVERITY_COLOR = {
    "Healthy": "#22c55e", "Myopia": "#f59e0b", "Pterygium": "#f59e0b",
    "Diabetic Retinopathy": "#ef4444", "Disc Edema": "#ef4444",
    "Retinal Detachment": "#dc2626", "Retinitis Pigmentosa": "#ef4444",
}

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vision_model():
    return load_model("best_efficientnetb3.keras")

@st.cache_resource(show_spinner=False)
def load_llm():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, llm

# ─── Helpers ──────────────────────────────────────────────────────────────────
def predict(img_pil, model):
    img = img_pil.resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(np.max(preds[0])), preds[0], arr

def make_gradcam_plusplus(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            conv_outputs, predictions = grad_model(img_array, training=False)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape1.gradient(loss, conv_outputs)
    second_grads = tape2.gradient(grads, conv_outputs)

    grads_np        = grads.numpy()[0]
    second_grads_np = second_grads.numpy()[0]
    conv_np         = conv_outputs.numpy()[0]

    denom   = 2 * grads_np**2 + conv_np * second_grads_np + 1e-8
    alpha   = second_grads_np / denom
    weights = np.sum(alpha * np.maximum(grads_np, 0), axis=(0, 1))

    cam = np.sum(weights * conv_np, axis=-1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    cam_resized   = cv2.resize(cam, IMG_SIZE)
    heatmap       = np.uint8(255 * cam_resized)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    original_np = np.clip(img_array[0], 0, 255).astype(np.uint8)
    overlay     = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    return original_np, heatmap_color, overlay

def generate_report(disease, confidence, tokenizer, llm):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"""You are an ophthalmology AI assistant.

Write exactly 5 short medical lines about this eye scan prediction:

Prediction: {disease}
Confidence: {confidence:.0%}

Structure:
1. Prediction statement.
2. Short definition of the condition.
3. Key symptoms the patient may experience.
4. Severity / urgency level.
5. Recommended next step.

Only write the 5 lines. No titles. No repetition."""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs, max_new_tokens=250, temperature=0.3,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    use_gradcam = st.toggle("Show Grad-CAM++ heatmap", value=True)
    use_llm     = st.toggle("Generate AI medical report (Phi-3)", value=True)
    st.markdown("---")
    st.markdown("**Detectable conditions:**")
    for c in CLASS_NAMES:
        info = DISEASE_INFO.get(c, {})
        st.markdown(f"{info.get('icon','•')} {c}")
    st.markdown("---")
    st.caption("Model: EfficientNetB3 · LLM: Phi-3-mini · XAI: Grad-CAM++")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Ophthalmology</div>
    <h1 class="hero-title">Eye Vision <span>AI</span></h1>
    <p class="hero-subtitle">
        Upload a retinal fundus image for AI-powered disease classification,
        Grad-CAM++ visual explanation, and an auto-generated medical report.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ─── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload retinal fundus image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

# ─── Main Flow ────────────────────────────────────────────────────────────────
if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")

    with st.spinner("Loading vision model…"):
        vision_model = load_vision_model()

    with st.spinner("🔍 Analyzing image…"):
        disease, confidence, all_probs, arr = predict(img_pil, vision_model)

    color = SEVERITY_COLOR.get(disease, "#38bdf8")
    info  = DISEASE_INFO.get(disease, {})

    # ── Row 1: Image | Diagnosis ──────────────────────────────────────────────
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown('<div class="section-header">📷 Input Image</div>', unsafe_allow_html=True)
        thumb = img_pil.copy()
        thumb.thumbnail((280, 280))
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(thumb, use_container_width=False, width=260)
        st.markdown('<span class="img-card-label">Retinal Fundus</span></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-header">🔬 Diagnosis Result</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-bottom:1.2rem;">
            <div class="diag-label">Detected Condition</div>
            <div style="display:flex; align-items:center; gap:0.7rem; margin-bottom:0.9rem;">
                <span style="font-size:1.8rem;">{info.get('icon','🔬')}</span>
                <span class="diag-name" style="color:{color};">{disease}</span>
            </div>
            <div class="confidence-label">Confidence Score</div>
            <div class="confidence-value">{confidence*100:.1f}<span>%</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(confidence * 100))

        # Disease info card
        if info:
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-card-title">📋 About this condition</div>
                <div class="disease-card-text">{info['desc']}</div>
                <div style="margin-top:0.65rem; padding-top:0.65rem;
                            border-top:1px solid rgba(56,189,248,0.1);">
                    <span style="font-size:0.75rem; color:#38bdf8; font-weight:500;">Recommendation: </span>
                    <span class="disease-card-text">{info['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── All Probabilities ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 All Class Probabilities"):
        for name, prob in sorted(zip(CLASS_NAMES, all_probs), key=lambda x: -x[1]):
            pct = float(prob) * 100
            bar_color = color if name == disease else "#1e3a4a"
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.75rem;
                         margin-bottom:0.55rem; font-size:0.82rem;">
                <div style="width:170px; color:#8ba3bf; white-space:nowrap;
                            overflow:hidden; text-overflow:ellipsis;">{name}</div>
                <div style="flex:1; background:rgba(255,255,255,0.05);
                            border-radius:999px; height:6px; overflow:hidden;">
                    <div style="width:{pct:.1f}%; height:100%;
                                background:{bar_color}; border-radius:999px;"></div>
                </div>
                <div style="width:44px; text-align:right; color:#5a7a96;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Grad-CAM++ ────────────────────────────────────────────────────────────
    if use_gradcam:
        st.markdown('<div class="divider" style="margin:2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">🔥 Grad-CAM++ Activation Map</div>', unsafe_allow_html=True)
        with st.spinner("Generating heatmap…"):
            try:
                orig_np, heatmap_np, overlay_np = make_gradcam_plusplus(arr, vision_model, LAST_CONV_LAYER)
                c1, c2, c3 = st.columns(3, gap="medium")
                with c1:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(orig_np, use_container_width=True)
                    st.markdown('<span class="img-card-label">Original</span></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(heatmap_np, use_container_width=True)
                    st.markdown('<span class="img-card-label">Heatmap</span></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(overlay_np, use_container_width=True)
                    st.markdown('<span class="img-card-label">Overlay</span></div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Grad-CAM++ failed: {e}")

    # ── LLM Report ────────────────────────────────────────────────────────────
    if use_llm:
        st.markdown('<div class="divider" style="margin:2rem 0;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📝 AI Medical Report</div>', unsafe_allow_html=True)
        with st.spinner("Generating medical report (Phi-3)…"):
            try:
                tokenizer, llm = load_llm()
                report = generate_report(disease, confidence, tokenizer, llm)
                st.markdown(
                    f'<div class="report-box">{report.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.warning(f"LLM report failed: {e}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Medical Disclaimer:</strong> This tool is for research purposes only and is
        not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified ophthalmologist.
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:320px; opacity:0.3; text-align:center;">
        <div style="font-size:3.5rem; margin-bottom:1rem;">👁️</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.15rem;
                    font-weight:600; color:#8ba3bf;">
            Upload a retinal fundus image to begin
        </div>
        <div style="font-size:0.85rem; color:#3a5a76; margin-top:0.5rem;">
            JPG · JPEG · PNG supported
        </div>
    </div>
    """, unsafe_allow_html=True)
