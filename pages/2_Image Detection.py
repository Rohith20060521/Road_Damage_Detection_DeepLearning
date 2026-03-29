import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

if image_file is not None:

    # Load the image
    image = Image.open(image_file)
    
    # SRIMS Analytics Placeholders
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        rhi_metric = st.empty()
    with metrics_col2:
        cost_metric = st.empty()

    warning_placeholder = st.empty()
    st.divider()
    
    col1, col2 = st.columns(2)

    # Perform inference
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Save the results
    current_rhi = 100
    current_cost = 0
    frame_potholes = 0
    frame_cracks = 0
    two_wheeler_hazard_detected = False

    def format_inr(number):
        s, *d = str(int(number)).partition(".")
        r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
        return "₹" + "".join([r] + d)

    detections = []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        # Extract segmentation masks if available (yolov8-seg models)
        has_masks = result.masks is not None
        masks_data = result.masks.data.cpu().numpy() if has_masks else None

        for idx, _box in enumerate(boxes):
            # Safe numpy scalar extraction patches for Numpy 1.25+ 
            class_id = int(_box.cls[0])
            
            # Area calculation
            if has_masks and masks_data is not None:
                # Count the number of active mask pixels for this specific defect instance
                pixel_area = np.sum(masks_data[idx] > 0.5)
            else:
                # Fallback to arbitrary bounding box area if using standard detection model
                box_coords_f = _box.xyxy[0]
                pixel_area = (box_coords_f[2] - box_coords_f[0]) * (box_coords_f[3] - box_coords_f[1])

            if class_id == 3: # Potholes
                frame_potholes += 1
                # Example: Pothole cost is flat rate ₹500 + dynamic ₹20 per pixel area
                current_cost += 500 + (pixel_area * 20)
            elif class_id in [0, 1, 2]: # Cracks
                frame_cracks += 1
                # Example: Crack cost is flat rate ₹100 + dynamic ₹5 per pixel area
                current_cost += 100 + (pixel_area * 5)
                
            # Calculate box width for hazard logic
            box_coords = _box.xyxy[0].astype(int)
            box_width = box_coords[2] - box_coords[0]
            
            # Determine Hazard Label
            hazard = "Two-Wheeler Hazard" if (box_width * (1280/640)) < 50 else "Heavy Vehicle Hazard"
            if hazard == "Two-Wheeler Hazard":
                two_wheeler_hazard_detected = True
                
            orig_label = CLASSES[class_id]
            new_label = f"{orig_label} - {hazard}"
            result.names[class_id] = new_label

            detections.append(
                Detection(
                    class_id=class_id,
                    label=new_label,
                    score=float(_box.conf[0]),
                    box=box_coords,
                )
            )

    # Update Analytics
    if frame_potholes > 0 or frame_cracks > 0:
        current_rhi -= (frame_potholes * 10 + frame_cracks * 2)
    
    rhi_metric.metric("Road Health Index (RHI)", current_rhi)
    cost_metric.metric("Estimated Repair Cost", format_inr(current_cost))
        
    if two_wheeler_hazard_detected:
        warning_placeholder.warning("High Risk for Bikes/Scooters Detected!", icon="⚠️")

    annotated_frame = results[0].plot()
    
    for i, name in enumerate(CLASSES):
        net.names[i] = name
        if len(results) > 0:
            results[0].names[i] = name
            
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation = cv2.INTER_AREA)

    # Original Image
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    # Predicted Image
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Download predicted image
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        downloadButton = st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )