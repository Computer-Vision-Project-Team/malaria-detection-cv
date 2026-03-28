import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
from image_utils import apply_gaussian_blur, segment_parasite

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Malaria Screening Tool", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>

/* COLUMN BORDERS */
div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
    border: 2px solid #4FC3F7;
    border-radius: 15px;
    padding: 25px;
    background-color: #1e1e1e;
}

div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
    border: 2px solid #FFD54F;
    border-radius: 15px;
    padding: 25px;
    background-color: #1e1e1e;
}

/* TITLE */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #4FC3F7;
    margin-bottom: 30px;
    position: relative;
    display: block;
}

/* Animated glowing underline */
.main-title::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -8px;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, transparent, #4FC3F7, transparent);
    animation: glowLine 2s infinite linear;
}

/* Glow animation */
@keyframes glowLine {
    0% {
        opacity: 0.3;
        transform: scaleX(0.6);
    }
    50% {
        opacity: 1;
        transform: scaleX(1);
        box-shadow: 0 0 10px #4FC3F7, 0 0 20px #4FC3F7;
    }
    100% {
        opacity: 0.3;
        transform: scaleX(0.6);
    }
}
/* INSTRUCTION PANEL */
.lab-panel {
    background: linear-gradient(135deg, #1e1e1e, #2b2b2b);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #555;   
    box-shadow: 0 0 8px rgba(79,195,247,0.15);
    margin: 0 auto 25px auto;
    width: 60%;                
    text-align: center;       
}

.lab-title {
    font-size: 20px;
    font-weight: bold;
    color: #4FC3F7;
    margin-bottom: 10px;
}

.lab-list {
    font-size: 15px;
    line-height: 1.8;
    color: #E0E0E0;
}

/* SECTION HEADINGS */
.section-left {
    color: #4FC3F7;
    font-size: 23px;
    font-weight: bold;
}

.section-right {
    color: #FFD54F;
    font-size: 23px;
    font-weight: bold;
}

/* SPACING HELPERS */
.space-small {
    margin-bottom: 5px;
}

.space-large {
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown('<div class="main-title"> Automated Malaria Screening Tool</div>', unsafe_allow_html=True)

# ---------------- INSTRUCTIONS ---------------- #
st.markdown("""
<div class="lab-panel">
<div class="lab-title">🧾 Laboratory Instructions</div>

<div class="lab-list">
✔ Upload <b>single red blood cell images</b><br>
✔ Ensure <b>Giemsa staining</b><br>
✔ Use <b>clear, focused images</b><br>
✔ Accepted formats: JPG, PNG<br>
✔ Click <b>Analyze Sample</b> to process
</div>

</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "malaria_cnn.h5")

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()

    return tf.keras.models.load_model(model_path)

model = load_model()

# ---------------- LAYOUT ---------------- #
col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ---------------- #
with col1:
    st.markdown('<div class="section-left space-large">🔬 Sample Upload</div>', unsafe_allow_html=True)

    st.markdown('<div class="space-small">Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png"], label_visibility="collapsed")

    analyze_btn = st.button("🧪 Analyze Sample")

# ---------------- RIGHT SIDE ---------------- #
with col2:
    st.markdown('<div class="section-right space-large">📊 Analysis Result</div>', unsafe_allow_html=True)

    if uploaded_file is not None and analyze_btn:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)

            # Validation
            h, w, _ = image_array.shape
            if h < 100 or w < 100:
                st.warning("⚠️ Image too small or invalid.")
                st.stop()

            # Preprocessing
            resized = cv2.resize(image_array, (224, 224))
            blurred = apply_gaussian_blur(resized)
            normalized = blurred / 255.0
            input_img = np.expand_dims(normalized, axis=0)

            # Prediction
            prediction = model.predict(input_img)[0][0]

            if prediction < 0.5:
                confidence = (1 - prediction) * 100
                segmented = segment_parasite(resized)

                st.error("Parasitized Cell Detected")
                st.image(segmented, width=350)
                st.write(f"Confidence: {confidence:.2f}%")

            else:
                confidence = prediction * 100

                st.success("Healthy Cell")
                st.image(resized, width=350)
                st.write(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Upload an image and click 'Analyze Sample'")