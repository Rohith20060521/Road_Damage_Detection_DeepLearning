import logging
import queue
import time
from pathlib import Path
from typing import List, NamedTuple
import csv
import os
import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO
from sample_utils.download import download_file
from sample_utils.get_STUNServer import getSTUNServer
import random
from utils.alerts import render_sidebar_alerts
from utils.style import apply_custom_style, render_top_nav, render_modern_card

st.set_page_config(page_title="Live Telemetry | SRIMS", page_icon="📡", layout="wide")

# Apply global styling
apply_custom_style()

# Top Navbar
render_top_nav("Live")

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
STUN_SERVER = [{"urls": ["stun:" + str(getSTUNServer())]}]
CLASSES=["Longitudinal Crack","Transverse Crack","Alligator Crack","Potholes"]

class Detection(NamedTuple):
    class_id:int
    label:str
    score:float
    box:np.ndarray

# Page Header
st.markdown("""
<div class="animate-in" style="margin-bottom: 30px;">
    <h1><i class="fa-solid fa-satellite-dish"></i> Real-time Telemetry Stream</h1>
    <p style="color: #8892B0;">Direct neural link for live mobile infrastructure inspection.</p>
</div>
""", unsafe_allow_html=True)

col_menu_1, col_menu_2 = st.columns([2, 1])

with col_menu_1:
    st.markdown("### <i class='fa-solid fa-sliders'></i> Signal Tuning", unsafe_allow_html=True)
    score_threshold = st.slider("Neural Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

with col_menu_2:
    st.markdown("### <i class='fa-solid fa-map-pin'></i> Deployment coordinates", unsafe_allow_html=True)
    lat_val = st.number_input("Latitudinal Origin", value=17.3850, format="%.4f")
    lon_val = st.number_input("Longitudinal Origin", value=78.4867, format="%.4f")
    location_label = st.text_input("Operational Sector", value="Central Square")

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def callback(frame: av.VideoFrame):
    img=frame.to_ndarray(format="bgr24")
    res=net.predict(cv2.resize(img,(640,640)),conf=score_threshold, verbose=False)

    dets=[]
    for r in res:
        for b in r.boxes.cpu().numpy():
            dets.append(Detection(int(b.cls),CLASSES[int(b.cls)],float(b.conf[0]),b.xyxy[0]))

    result_queue.put(dets)

    return av.VideoFrame.from_ndarray(res[0].plot(),format="bgr24")

st.markdown("<br>", unsafe_allow_html=True)

webrtc_ctx = webrtc_streamer(
    key="rdd",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=callback,
    media_stream_constraints={"video":True,"audio":False},
    async_processing=True
)

st.divider()

if webrtc_ctx.state.playing:
    # RESET LOG
    header = ["filename","lat","lon","location_label","damage_type","confidence","severity","media_type","status","task_id"]
    with open("gps_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    st.markdown("### <i class='fa-solid fa-list-check'></i> Live Hazard ingestion", unsafe_allow_html=True)
    labels_placeholder = st.empty()
    while webrtc_ctx.state.playing:
        try:
            result = result_queue.get(timeout=1.0)
            if result:
                for det in result:
                    jitter_lat = lat_val + random.uniform(-0.0001, 0.0001)
                    jitter_lon = lon_val + random.uniform(-0.0001, 0.0001)

                    log_detection("live_stream.png", jitter_lat, jitter_lon, location_label, det.label, det.score, "webcam")
                
                labels_placeholder.dataframe(result, width='stretch')
        except queue.Empty:
            if not webrtc_ctx.state.playing:
                break
            continue
