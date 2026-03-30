import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection using UAV and Deep Learning",
    page_icon="🚧",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #FFFFFF;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.metric-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-size: 18px;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚧 Road Damage Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>AI-powered analysis using UAV imagery and YOLOv8</p>", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.markdown("<div class='metric-box'>⚡ Fast Detection<br><b>Real-time</b></div>", unsafe_allow_html=True)
col2.markdown("<div class='metric-box'>🎯 High Accuracy<br><b>YOLOv8 Model</b></div>", unsafe_allow_html=True)
col3.markdown("<div class='metric-box'>📊 Multi Input<br><b>Image / Video / Webcam</b></div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 📌 Project Overview")

st.write("""
This project presents an AI-based system for detecting road damages using UAV (drone) imagery and deep learning techniques.

The system leverages the YOLOv8 model to identify and classify different types of road damages such as cracks and potholes from images, videos, and real-time inputs.
""")

st.markdown("---")

st.markdown("## 🚀 Features")

col1, col2 = st.columns(2)

col1.markdown("""
### 🔍 Detection
- 📷 Image Detection  
- 🎥 Video Detection  
- 🎯 Real-time Detection  
""")

col2.markdown("""
### ⚙️ System
- ⚡ Fast YOLOv8 Inference  
- 🧠 Deep Learning Model  
- 📊 Interactive Dashboard  
""")

st.markdown("---")

st.markdown("## 🛠️ Damage Types")

col1, col2 = st.columns(2)

col1.markdown("""
- Longitudinal Crack  
- Transverse Crack  
""")

col2.markdown("""
- Alligator Crack  
- Potholes  
""")

st.markdown("---")

st.markdown("## 🧑‍💻 How It Works")

st.markdown("""
1️⃣ Select module from sidebar  
2️⃣ Upload input (image/video/webcam)  
3️⃣ AI detects road damage  
4️⃣ View results with bounding boxes & metrics  
""")

st.markdown("---")

st.markdown("## ⚙️ Tech Stack")

st.markdown("""
- 🐍 Python  
- 🎨 Streamlit  
- 🤖 YOLOv8 (Ultralytics)  
- 📷 OpenCV  
- 🔥 PyTorch  
""")

st.markdown("---")

st.markdown("<p style='text-align:center; color:gray;'>Built using Deep Learning for Smart Infrastructure Monitoring</p>", unsafe_allow_html=True)
