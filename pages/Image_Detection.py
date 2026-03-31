import logging
from pathlib import Path
from typing import NamedTuple
import time
import csv
import os
import random
from streamlit_image_comparison import image_comparison
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from sample_utils.download import download_file
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_top_nav, render_modern_card

st.set_page_config(
    page_title="Image Integrity | SRIMS",
    page_icon="📸",
    layout="wide"
)

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Image")

render_sidebar_alerts()

import uuid

def log_detection(filename, lat, lon, label, damage_type, confidence, media_type):
    severity = "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low"
    task_id = str(uuid.uuid4())[:8]
    row = [filename, lat, lon, label, damage_type, round(confidence, 2), severity, media_type, "Pending", task_id]
    with open("gps_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

if "model" not in st.session_state:
    st.session_state.model = YOLO(MODEL_LOCAL_PATH)

net = st.session_state.model
CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-camera-retro"></i> Precision Image Diagnostic</h1>
    <p style="color: #8892B0;">Upload high-resolution infrastructure imagery for forensic structural analysis.</p>
</div>
""", unsafe_allow_html=True)

col_menu_1, col_menu_2 = st.columns([2, 1])

with col_menu_1:
    st.markdown("### <i class='fa-solid fa-file-image'></i> Source Asset", unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image (PNG/JPG)", type=['png', 'jpg'])
    score_threshold = st.slider("Neural Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

with col_menu_2:
    st.markdown("### <i class='fa-solid fa-location-crosshairs'></i> Spatial Metadata", unsafe_allow_html=True)
    lat_val = st.number_input("Latitudinal Coordinate", value=17.3850, format="%.4f")
    lon_val = st.number_input("Longitudinal Coordinate", value=78.4867, format="%.4f")
    location_label = st.text_input("Site Identifier", value="Hub-Alpha")

st.markdown("<br>", unsafe_allow_html=True)

if image_file:
    image = Image.open(image_file)
    _image = np.array(image)

    resized = cv2.resize(_image, (640, 640))
    results = net.predict(resized, conf=score_threshold)

    rhi = 100
    cost = 0

    # RESET LOG
    header = ["filename","lat","lon","location_label","damage_type","confidence","severity","media_type","status","task_id"]
    with open("gps_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    if len(results[0].boxes) > 0:
        for r in results:
            for b in r.boxes.cpu().numpy():
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = b.xyxy[0]
                area = (x2 - x1) * (y2 - y1) / 1000

                if cls == 3: # Pothole
                    if area < 5: c, impact = 1500, 8
                    elif area < 15: c, impact = 4000, 15
                    else: c, impact = 8000, 25
                else: # Crack
                    if area < 5: c, impact = 500, 3
                    elif area < 15: c, impact = 1500, 6
                    else: c, impact = 3000, 10

                cost += c
                rhi -= impact
                
                jitter_lat = lat_val + random.uniform(-0.0001, 0.0001)
                jitter_lon = lon_val + random.uniform(-0.0001, 0.0001)

                log_detection(image_file.name, jitter_lat, jitter_lon, location_label, CLASSES[cls], conf, "image")

        rhi = max(rhi, 0)
        
        st.markdown("### <i class='fa-solid fa-microchip'></i> Analysis Telemetry", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Integrity Index (RHI)", int(rhi))
        with m_col2:
            st.metric("Estimated Repair (ROI)", f"₹{int(cost)}")
    else:
        st.success("✅ MISSION CLEAR: No structural hazards identified in this asset.")

    st.divider()

    annotated = results[0].plot()
    annotated = cv2.resize(annotated, (_image.shape[1], _image.shape[0]))
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.subheader("<i class='fa-solid fa-swatchbook'></i> Interactive Forensics Slider", unsafe_allow_html=True)
    
    # Image Comparison Slider
    image_comparison(
        img1=Image.fromarray(_image),
        img2=Image.fromarray(annotated_rgb),
        label1="RAW SOURCE",
        label2="AI ANALYTICS",
        width=None,
        make_responsive=True,
        starting_position=50,
        show_labels=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn = st.columns([1, 1, 1])
    with col_btn[1]:
        buffer = BytesIO()
        Image.fromarray(annotated_rgb).save(buffer, format="PNG")
        st.download_button(
            "💾 EXPORT ANALYTIC PNG",
            buffer.getvalue(),
            file_name="SRIMS_Diagnostic.png",
            mime="image/png",
            width='stretch'
        )
    
    if len(results[0].boxes) > 0:
        st.toast(f"✅ {len(results[0].boxes)} hazards ingested to command center!")
