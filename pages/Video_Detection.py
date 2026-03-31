import os
from pathlib import Path
import cv2
import streamlit as st
import time
import csv
import random
from ultralytics import YOLO
from sample_utils.download import download_file
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_top_nav, render_modern_card

st.set_page_config(
    page_title="Video Analytics | SRIMS",
    page_icon="🎥",
    layout="wide"
)

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Video")

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

os.makedirs("./temp", exist_ok=True)
input_path = "./temp/input.mp4"
output_path = "./temp/output.mp4"

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-video"></i> Strategic Video Analytics</h1>
    <p style="color: #8892B0;">Deploy YOLOv8 computer vision for high-throughput mobile infrastructure audits.</p>
</div>
""", unsafe_allow_html=True)

col_menu_1, col_menu_2 = st.columns([2, 1])

with col_menu_1:
    st.markdown("### <i class='fa-solid fa-upload'></i> Source Media", unsafe_allow_html=True)
    video = st.file_uploader("Upload Infrastructure Footage", type="mp4")
    conf = st.slider("Neural Confidence Threshold", 0.0, 1.0, 0.3, 0.05)

with col_menu_2:
    st.markdown("### <i class='fa-solid fa-location-dot'></i> Deployment Context", unsafe_allow_html=True)
    lat_val = st.number_input("Latitudinal Origin", value=17.3850, format="%.4f")
    lon_val = st.number_input("Longitudinal Origin", value=78.4867, format="%.4f")
    location_label = st.text_input("Operational Sector", value="Hyderabad Hub")

st.markdown("<br>", unsafe_allow_html=True)

if video and st.button("🚀 INITIATE NEURAL PROCESSING"):

    with open(input_path, "wb") as f:
        f.write(video.getbuffer())

    # RESET LOG
    header = ["filename","lat","lon","location_label","damage_type","confidence","severity","media_type","status","task_id"]
    with open("gps_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)
    total = int(cap.get(7))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    rhi = 100
    cost = 0
    
    # Progress Section
    progress_container = st.empty()
    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        video_placeholder = st.empty()
    
    with col_v2:
        st.markdown("### <i class='fa-solid fa-gauge-high'></i> Telemetry", unsafe_allow_html=True)
        rhi_box = st.empty()
        cost_box = st.empty()
        hazard_log = st.empty()

    frame_count = 0
    hazards_found = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 15 == 0:
            resized = cv2.resize(frame, (640, 640))
            results = net.predict(resized, conf=conf, verbose=False)
            
            for r in results:
                for b in r.boxes.cpu().numpy():
                    cls = int(b.cls[0])
                    conf_score = float(b.conf[0])
                    x1, y1, x2, y2 = b.xyxy[0]
                    area = (x2 - x1) * (y2 - y1) / 1000

                    if cls == 3: # Pothole
                        if area < 5: c, impact = 1500, 5
                        elif area < 15: c, impact = 4000, 10
                        else: c, impact = 8000, 15
                    else: # Crack
                        if area < 5: c, impact = 500, 2
                        elif area < 15: c, impact = 1500, 4
                        else: c, impact = 3000, 6

                    cost += c * 0.5 
                    rhi -= impact
                    
                    jitter_lat = lat_val + random.uniform(-0.0001, 0.0001)
                    jitter_lon = lon_val + random.uniform(-0.0001, 0.0001)

                    log_detection(video.name, jitter_lat, jitter_lon, location_label, CLASSES[cls], conf_score, "video")
                    hazards_found.append(f"{CLASSES[cls]} ({int(conf_score*100)}%)")

            rhi = max(rhi, 0)
            annotated = results[0].plot()
            annotated = cv2.resize(annotated, (width, height))
            out.write(annotated)
            
            # Update metrics in Navy/Cyan style
            rhi_box.metric("Road Health index", int(rhi))
            cost_box.metric("Estimated Liability", f"₹{int(cost)}")
            
            # Display live feed
            video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width='stretch')
            
            if hazards_found:
                hazard_log.code("\n".join(hazards_found[-5:]), language="text")

        frame_count += 1
        progress_container.progress(min(frame_count / total, 1.0))

    cap.release()
    out.release()

    st.success("✅ MISSION COMPLETE: Data Ingested")

    if os.path.exists(output_path):
        st.divider()
        st.subheader("<i class='fa-solid fa-clapperboard'></i> Full Processed Audit", unsafe_allow_html=True)
        with open(output_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("💾 DOWNLOAD ARCHIVE", video_bytes, file_name="srims_audit.mp4")
    
    st.toast("✅ Video analytics synced with central dashboard!")
