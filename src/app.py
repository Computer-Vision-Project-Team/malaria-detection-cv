import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from image_utils import segment_parasite

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Malaria Screening Tool", layout="wide", initial_sidebar_state="expanded")

# ---------------- STYLING ---------------- #
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css');

:root {
    --bg: #e9edf5;
    --card: #ffffff;
    --line: #d7dde7;
    --text: #0f172a;
    --muted: #5b6577;
    --primary: #2563eb;
    --teal: #0ea5a7;
}

.stApp {
    background: var(--bg);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef3fb 0%, #e7eef9 100%);
    border-right: 1px solid #cfd9e8;
}

/* Force sidebar visible even when Streamlit keeps collapsed state in browser */
section[data-testid="stSidebar"][aria-expanded="false"],
section[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 300px !important;
    max-width: 300px !important;
    transform: translateX(0) !important;
    margin-left: 0 !important;
}

/* Keep sidebar fixed open by hiding collapse controls */
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 0.85rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}

.sidebar-shell {
    background: #f8fbff;
    border: 1px solid #d6e2f2;
    border-radius: 14px;
    padding: 12px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
    margin-bottom: 10px;
}

.sidebar-title {
    margin: 0;
    color: #0f2948;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.sidebar-subtitle {
    margin: 4px 0 0;
    color: #4f6b8a;
    font-size: 12px;
    font-weight: 600;
}

.sidebar-divider {
    height: 1px;
    margin: 10px 0 12px;
    background: linear-gradient(90deg, rgba(25, 60, 106, 0.1) 0%, rgba(25, 60, 106, 0.45) 50%, rgba(25, 60, 106, 0.1) 100%);
}

.sidebar-stat {
    border-radius: 12px;
    padding: 14px 10px;
    color: #ffffff;
    text-align: center;
    margin-bottom: 11px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 10px 18px rgba(19, 42, 84, 0.22);
}

.sidebar-stat small {
    display: block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.6px;
    opacity: 0.92;
}

.sidebar-stat h4 {
    margin: 6px 0 0;
    font-size: 52px;
    line-height: 1;
    font-weight: 850;
}

.sidebar-stat.availability { background: linear-gradient(135deg, #1d4f9a 0%, #1c2f6e 100%); }
.sidebar-stat.accuracy { background: linear-gradient(135deg, #1b4a93 0%, #1a356f 100%); }
.sidebar-stat.time { background: linear-gradient(135deg, #1f569f 0%, #1c3b79 100%); }

div.block-container {
    max-width: 1180px;
    padding-top: 0.5rem;
    padding-bottom: 0.6rem;
}

[data-testid="stHeader"],
#MainMenu,
footer {
    display: none !important;
}

/* Hide Streamlit heading/link action icons globally */
[data-testid="stHeaderActionElements"],
button[aria-label*="Copy link"],
button[aria-label*="link to heading"] {
    display: none !important;
}

.topbar {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.topbar-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: #1d4ed8;
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
}

.topbar-title {
    margin: 0;
    font-size: 21px;
    line-height: 1;
    color: var(--text);
    font-weight: 800;
}

.topbar-sub {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
}

.card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 1px 0 rgba(16, 24, 40, 0.02), 0 2px 10px rgba(16, 24, 40, 0.04);
    margin-bottom: 12px;
}

.card-title {
    margin: 0 0 10px;
    font-size: 22px;
    color: var(--text);
    font-weight: 750;
    display: flex;
    gap: 10px;
    align-items: center;
}

.card-title i {
    color: var(--primary);
    font-size: 20px;
}

.sample-header-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 1px 0 rgba(16, 24, 40, 0.02), 0 2px 10px rgba(16, 24, 40, 0.04);
    margin: 8px 0 10px;
}

.lab-points {
    margin: 0;
    padding-left: 6px;
}

.lab-points li {
    list-style: none;
    color: #334155;
    font-size: 13px;
    margin: 7px 0;
    display: flex;
    align-items: flex-start;
    gap: 9px;
}

.lab-points li i {
    color: var(--teal);
    margin-top: 3px;
}

.result-placeholder {
    min-height: 485px;
    border-radius: 14px;
    border: 1px dashed #d1d8e4;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    color: #556071;
}

.result-placeholder i {
    font-size: 52px;
    color: #b5bdca;
    margin-bottom: 10px;
}

.result-head {
    border-radius: 12px;
    padding: 10px 14px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: 24px;
}

.result-head.bad {
    color: #7f1d1d;
    background: #fee2e2;
    border: 1px solid #fca5a5;
}

.result-head.good {
    color: #14532d;
    background: #dcfce7;
    border: 1px solid #86efac;
}

.diagnosis-card {
    border-radius: 12px;
    padding: 14px 16px;
    border-left: 4px solid #ef4444;
    background: #fee2e2;
    color: #7f1d1d;
}

.diagnosis-card h4 {
    margin: 0 0 8px;
    font-size: 18px;
}

.diagnosis-card p {
    margin: 4px 0;
    font-size: 13px;
    line-height: 1.45;
}

.confidence-title {
    text-align: center;
    font-size: 15px;
    font-weight: 700;
    color: #475569;
    margin-bottom: 12px;
    padding-bottom: 10px;
    border-bottom: 3px solid #0a7a7b;
}

.confidence-card {
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(15, 23, 42, 0.06);
}

.confidence-card.bad {
    background: #fee2e2;
    border-left: 4px solid #ef4444;
}

.confidence-card.good {
    background: #f1f5f9;
    border-left: 4px solid #22c55e;
}

.confidence-label {
    margin: 0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.confidence-value {
    margin: 6px 0 0;
    font-size: 40px;
    font-weight: 800;
    line-height: 1;
}

.process-wrap {
    margin-top: 12px;
    border-top: 1px solid #d9e0ea;
    padding-top: 12px;
}

.process-title {
    margin: 0 0 10px;
    font-size: 18px;
    font-weight: 800;
    color: #334155;
}

.process-card {
    border-radius: 12px;
    padding: 12px;
    min-height: 112px;
    border: 2px solid;
    text-align: center;
}

.process-card h5 {
    margin: 4px 0 6px;
    font-size: 12px;
    font-weight: 800;
}

.process-card h6 {
    margin: 0 0 6px;
    font-size: 18px;
    font-weight: 700;
}

.process-card p {
    margin: 0;
    font-size: 12px;
    color: #5b6577;
}

.process-scan { background: #e9f3ff; border-color: #7cc2ff; }
.process-highlight { background: #fff5d8; border-color: #e4c66a; }
.process-output { background: #f3e8ff; border-color: #c08bff; }

div[data-testid="stFileUploader"] {
    border: none !important;
    border-radius: 12px;
    padding: 0;
    background: transparent;
}

div[data-testid="stFileUploaderDropzone"] {
    min-height: 150px;
    border: 1px solid #d5dbe6 !important;
    border-radius: 12px !important;
    background: #eef2f8 !important;
}

div[data-testid="stFileUploaderDropzone"] section {
    padding-top: 12px;
    padding-bottom: 12px;
}

div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    font-size: 12px !important;
}

button[kind="primary"] {
    border-radius: 10px !important;
    height: 42px;
    font-size: 15px;
    font-weight: 650;
}

div[data-testid="stVerticalBlock"] > div:has(> div .card-title) {
    margin-bottom: 10px;
}

@media (max-width: 900px) {
    .topbar-title { font-size: 18px; }
    .topbar-sub { font-size: 12px; }
    .card-title { font-size: 20px; }
    .lab-points li { font-size: 12px; }
    .result-placeholder { min-height: 260px; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown(
    """
    <div class="topbar">
      <div class="topbar-icon"><i class="fa-solid fa-microscope"></i></div>
      <div>
        <h1 class="topbar-title">Malaria Screening Tool</h1>
        <p class="topbar-sub">Automated diagnostic analysis system</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '''
        <div class="sidebar-shell">
            <h3 class="sidebar-title">System Metrics</h3>
            <p class="sidebar-subtitle">Real-time diagnostics dashboard</p>
            <div class="sidebar-divider"></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.markdown(
        '''
        <div class="sidebar-stat availability">
            <small>AVAILABILITY</small>
            <h4>24/7</h4>
        </div>
        <div class="sidebar-stat accuracy">
            <small>ACCURACY</small>
            <h4>94%+</h4>
        </div>
        <div class="sidebar-stat time">
            <small>TIME</small>
            <h4>&lt; 1s</h4>
        </div>
        ''',
        unsafe_allow_html=True,
    )

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    model_path = os.path.join(project_root, "models", "malaria_cnn.h5")

    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)

    st.error("Model not found. Expected: models/malaria_cnn.h5")
    st.stop()

model = load_model()


def count_parasite_regions(image_rgb):
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([110, 40, 40])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for cnt in contours if cv2.contourArea(cnt) > 50)

# ---------------- LAYOUT ---------------- #
analysis_done = False
parasitized_pct = 0.0
uninfected_pct = 0.0
parasite_count = 0
diagnosis_label = ""
model_confidence = 0.0
model_uncertain = False

col1, col2 = st.columns(2)

# ---------------- LEFT SIDE ---------------- #
with col1:
    st.markdown(
        """
        <div class="card">
          <h3 class="card-title"><i class="fa-regular fa-clipboard"></i> Laboratory Instructions</h3>
          <ul class="lab-points">
            <li><i class="fa-regular fa-circle-check"></i><span>Upload single red blood cell images</span></li>
            <li><i class="fa-regular fa-circle-check"></i><span>Ensure Giemsa staining for optimal results</span></li>
            <li><i class="fa-regular fa-circle-check"></i><span>Use clear, focused images (100x magnification)</span></li>
            <li><i class="fa-regular fa-circle-check"></i><span>Supported formats: JPG, PNG (Max 200MB)</span></li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '''
        <div class="sample-header-card">
          <h3 class="card-title" style="margin:0;"><i class="fa-solid fa-arrow-up-from-bracket"></i> Sample Upload</h3>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    analyze_btn = st.button("Analyze Sample", type="primary", use_container_width=True)

# ---------------- RIGHT SIDE ---------------- #
with col2:
    st.markdown(
        """
        <div class="card">
          <h3 class="card-title"><i class="fa-solid fa-microscope"></i> Analysis Results</h3>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_file is not None and analyze_btn:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            image_array = np.array(image)

            # Validation
            h, w, _ = image_array.shape
            if h < 100 or w < 100:
                st.warning("Image too small or invalid.")
                st.stop()

            # Preprocessing
            resized = cv2.resize(image_array, (224, 224))
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)
            normalized = blurred / 255.0
            input_img = np.expand_dims(normalized, axis=0)

            # Prediction
            prediction = model.predict(input_img)[0][0]
            uninfected_pct = prediction * 100
            parasitized_pct = (1 - prediction) * 100
            model_confidence = max(uninfected_pct, parasitized_pct)
            model_uncertain = 50.0 <= model_confidence <= 70.0
            parasite_count = count_parasite_regions(resized)
            analysis_done = True

            if prediction < 0.5:
                confidence = (1 - prediction) * 100
                segmented = segment_parasite(resized)
                diagnosis_label = "PARASITIZED"

                st.markdown('<div class="result-head bad"><i class="fa-solid fa-triangle-exclamation"></i>Parasitized Cell Detected</div>', unsafe_allow_html=True)
                st.image(segmented, width=340)

            else:
                confidence = prediction * 100
                diagnosis_label = "UNINFECTED"

                st.markdown('<div class="result-head good"><i class="fa-solid fa-circle-check"></i>Healthy Cell</div>', unsafe_allow_html=True)
                st.image(resized, width=340)

            if model_uncertain:
                st.warning(
                    f"Model is uncertain for this sample (confidence: {model_confidence:.1f}%). "
                    "Please review manually or test with another clear image."
                )

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.markdown(
            """
            <div class="result-placeholder">
              <i class="fa-regular fa-file-lines"></i>
              <p style="font-size: 16px; margin: 0;">Upload an image and click 'Analyze Sample' to view results</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

if analysis_done:
    st.markdown('<div class="process-wrap">', unsafe_allow_html=True)

    dcol1, dcol2 = st.columns([1.15, 1])

    with dcol1:
        if diagnosis_label == "PARASITIZED":
            st.markdown(
                f'''
                <div class="diagnosis-card">
                  <h4>PARASITIZED</h4>
                  <p><b>Diagnosis:</b> Cell is infected with malaria parasites</p>
                  <p><b>Evidence:</b> Dark purple parasite structures visible</p>
                  <p><b>Regions Found:</b> {parasite_count} infected area(s)</p>
                  <p><b>Recommendation:</b> Medical intervention required</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'''
                <div class="diagnosis-card" style="border-left-color:#22c55e;background:#dcfce7;color:#14532d;">
                  <h4>UNINFECTED</h4>
                  <p><b>Diagnosis:</b> Cell appears healthy and uninfected</p>
                  <p><b>Evidence:</b> No clear parasite structures detected</p>
                  <p><b>Regions Found:</b> {parasite_count} suspicious area(s)</p>
                  <p><b>Recommendation:</b> Continue routine observation</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    with dcol2:
        st.markdown('<div class="confidence-title">Confidence Analysis</div>', unsafe_allow_html=True)
        if diagnosis_label == "PARASITIZED":
            st.markdown(
                f'''
                <div class="confidence-card bad">
                  <p class="confidence-label" style="color:#b91c1c;">PARASITIZED</p>
                  <p class="confidence-value" style="color:#dc2626;">{parasitized_pct:.1f}%</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'''
                <div class="confidence-card good">
                  <p class="confidence-label" style="color:#16a34a;">UNINFECTED</p>
                  <p class="confidence-value" style="color:#16a34a;">{uninfected_pct:.1f}%</p>
                </div>
                ''',
                unsafe_allow_html=True,
            )

    st.markdown('<h3 class="process-title">Technical Process</h3>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown(
            '<div class="process-card process-scan"><h5>[SCAN]</h5><h6>Cell Scanning</h6><p>Deep learning model analyzes cellular structure and identifies parasite markers</p></div>',
            unsafe_allow_html=True,
        )
    with p2:
        st.markdown(
            '<div class="process-card process-highlight"><h5>[HIGHLIGHT]</h5><h6>Visual Detection</h6><p>Purple parasite regions are outlined in red for clear identification</p></div>',
            unsafe_allow_html=True,
        )
    with p3:
        st.markdown(
            '<div class="process-card process-output"><h5>[OUTPUT]</h5><h6>Classification</h6><p>Binary prediction: Parasitized or Uninfected with confidence scores</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)