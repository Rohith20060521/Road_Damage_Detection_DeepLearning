import os
from pathlib import Path
import cv2
import streamlit as st
from ultralytics import YOLO
from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
    page_icon="📷",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1 {
    color: white;
    text-align: center;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

if "model" not in st.session_state:
    st.session_state.model = YOLO(MODEL_LOCAL_PATH)

net = st.session_state.model

os.makedirs("./temp", exist_ok=True)

input_path = "./temp/input.mp4"
output_path = "./temp/output.mp4"

st.title("🎥 Video Road Damage Detection")

video = st.file_uploader("Upload Video", type="mp4")

conf = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05
)

st.divider()

if video and st.button("🚀 Process Video"):

    with open(input_path, "wb") as f:
        f.write(video.getbuffer())

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

    progress = st.progress(0)

    col1, col2 = st.columns(2)
    rhi_box = col1.empty()
    cost_box = col2.empty()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (640, 640))
        results = net.predict(resized, conf=conf)

        if frame_count % 15 == 0:
            for r in results:
                for b in r.boxes.cpu().numpy():
                    cls = int(b.cls[0])
                    x1, y1, x2, y2 = b.xyxy[0]
                    area = (x2 - x1) * (y2 - y1) / 1000

                    if cls == 3:
                        if area < 5:
                            c, impact = 1500, 5
                        elif area < 15:
                            c, impact = 4000, 10
                        else:
                            c, impact = 8000, 15
                    else:
                        if area < 5:
                            c, impact = 500, 2
                        elif area < 15:
                            c, impact = 1500, 4
                        else:
                            c, impact = 3000, 6

                    cost += c * 0.5
                    rhi -= impact

        rhi = max(rhi, 0)

        annotated = results[0].plot()
        annotated = cv2.resize(annotated, (width, height))

        out.write(annotated)

        rhi_box.metric("Road Health Index", int(rhi))
        cost_box.metric("Repair Cost", f"₹{int(cost)}")

        frame_count += 1
        progress.progress(frame_count / total)

    cap.release()
    out.release()

    st.success("Processing Completed")

    if os.path.exists(output_path):
        st.subheader("Processed Video")

        with open(output_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            "Download Processed Video",
            video_bytes,
            file_name="output.mp4"
        )
