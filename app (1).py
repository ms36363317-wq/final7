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
    initial_sidebar_state="expanded",
    layout="wide"
)

# ==============================
# Custom CSS (Light Navy & Sky Blue Theme)
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* الخلفية المضيئة */
    .stApp {
        background-color: #f8fafc !important;
        color: #1e293b !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem; max-width: 1400px; }

    /* ── Sidebar (Navy Blue) ── */
    [data-testid="stSidebar"] {
        background-color: #1e3a8a !important; /* كحلي ملكي */
        border-right: 1px solid #1e293b;
    }
    
    .sidebar-title {
        color: #ffffff;
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #38bdf8;
    }

    .disease-item-sidebar {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.8rem;
        border-radius: 10px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
    
    .disease-name-sidebar {
        color: #f0f9ff;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        color: #1e3a8a; /* كحلي */
        text-align: center;
        margin-bottom: 2rem;
    }
    .hero-title span { color: #0ea5e9; } /* سماوي */

    /* ── Cards & UI ── */
    .img-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }

    .stProgress > div > div > div > div { background: #0ea5e9 !important; }

    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fef3c7;
        border-radius: 12px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #92400e;
        text-align: center;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Constants & Data
# ==============================
MODEL_PATH = "best_efficientnetb3.h5"
FILE_ID = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"

class_names = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa'
]

disease_info = {
    "Diabetic Retinopathy": {"desc": "تلف في أوعية الدم نتيجة السكري.", "icon": "🩺", "color": "#ef4444"},
    "Disc Edema": {"desc": "تورم في القرص البصري.", "icon": "🧠", "color": "#ef4444"},
    "Healthy": {"desc": "الشبكية سليمة تماماً.", "icon": "✅", "color": "#22c55e"},
    "Myopia": {"desc": "قصر النظر الشديد.", "icon": "👓", "color": "#f59e0b"},
    "Pterygium": {"desc": "نمو نسيجي على القرنية.", "icon": "🔬", "color": "#f59e0b"},
    "Retinal Detachment": {"desc": "انفصال الشبكية (حالة طارئة).", "icon": "🚨", "color": "#dc2626"},
    "Retinitis Pigmentosa": {"desc": "تدهور وراثي في الشبكية.", "icon": "🧬", "color": "#ef4444"},
}

# ==============================
# Sidebar (Diseases Only)
# ==============================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🔍 الأمراض المدعومة</div>', unsafe_allow_html=True)
    for name in class_names:
        info = disease_info.get(name, {"icon": "•"})
        st.markdown(f"""
        <div class="disease-item-sidebar">
            <span style="font-size:1.2rem;">{info['icon']}</span>
            <span class="disease-name-sidebar">{name}</span>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Main Logic
# ==============================
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

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
    cam = tf.reduce_sum(tf.reduce_mean(grads, axis=(1, 2))[:, None, None, :] * conv_outs, axis=-1)[0].numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(np.power(cam / (np.max(cam) + 1e-8), 0.3), (300, 300))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return heatmap, cv2.addWeighted(np.array(img_res), 0.7, heatmap, 0.3, 0)

# ── Header ──
st.markdown('<h1 class="hero-title">نظام فحص <span>أمراض الشبكية</span> الذكي</h1>', unsafe_allow_html=True)

model = load_model_cached()

# ── Layout ──
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 📤 رفع الصورة")
    uploaded_file = st.file_uploader("اختر صورة قاع العين", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('<b>الصورة المرفوعة</b>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file:
        with st.spinner("جاري التحليل..."):
            pred, conf, all_preds = predict(image, model)
            heatmap, overlay = gradcam(image, model)
        
        info = disease_info.get(pred, {})
        
        st.markdown(f"### 📋 نتيجة التشخيص: <span style='color:{info['color']}'>{pred}</span>", unsafe_allow_html=True)
        st.write(f"**نسبة الثقة:** {conf*100:.1f}%")
        st.progress(int(conf * 100))
        
        st.info(f"**وصف الحالة:** {info['desc']}")
        
        st.markdown("---")
        st.markdown("### 🔍 التحليل البصري (مناطق الإصابة)")
        v1, v2 = st.columns(2)
        v1.image(heatmap, caption="خريطة الحرارة", use_container_width=True)
        v2.image(overlay, caption="التحديد على العين", use_container_width=True)
    else:
        st.info("الرجاء رفع صورة لبدء عملية التحليل الآلي.")

st.markdown("""
<div class="disclaimer">
    ⚠️ <b>تنبيه طبي:</b> هذا النظام هو أداة مساعدة للفحص الأولي فقط ولا يغني عن استشارة طبيب العيون المتخصص.
</div>
""", unsafe_allow_html=True)
