import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
import folium
from streamlit_folium import st_folium

# Deep learning framework
from ultralytics import YOLO

from sample_utils.download import download_file

st.set_page_config(
    page_title="Video Detection",
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

# Load MiDaS model
midas_cache_key = "midas_small"
if midas_cache_key in st.session_state:
    midas = st.session_state[midas_cache_key]
    midas_transforms = st.session_state["midas_transforms"]
else:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    st.session_state[midas_cache_key] = midas
    st.session_state["midas_transforms"] = midas_transforms

# Initialize Secondary ResNet Classifier
resnet_cache_key = "resnet_hazard"
if resnet_cache_key in st.session_state:
    hazard_classifier = st.session_state[resnet_cache_key]
else:
    # Assuming binary classification (e.g., 0: Normal, 1: Critical)
    hazard_classifier = models.resnet18(weights=None)
    num_ftrs = hazard_classifier.fc.in_features
    hazard_classifier.fc = torch.nn.Linear(num_ftrs, 2)
    
    # Load your local weights
    try:
        hazard_classifier.load_state_dict(torch.load("hazard_classifier.pth"))
    except FileNotFoundError:
        pass
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    hazard_classifier.to(device)
    hazard_classifier.eval()
    st.session_state[resnet_cache_key] = hazard_classifier

# Standard ResNet ImageNet transformation pipeline
resnet_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# Create temporary folder if doesn't exists
if not os.path.exists('./temp'):
   os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Processing state
if 'processing_button' in st.session_state and st.session_state.processing_button == True:
    st.session_state.runningInference = True
else:
    st.session_state.runningInference = False

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

def processVideo(video_file, score_threshold):
    
    # Write the file into disk
    write_bytesio_to_file(temp_file_input, video_file)
    
    videoCapture = cv2.VideoCapture(temp_file_input)

    # Check the video
    if (videoCapture.isOpened() == False):
        st.error('Error opening the video file')
    else:
        _width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        _height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = videoCapture.get(cv2.CAP_PROP_FPS)
        _frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        _duration = _frame_count/_fps
        _duration_minutes = int(_duration/60)
        _duration_seconds = int(_duration%60)
        _duration_strings = str(_duration_minutes) + ":" + str(_duration_seconds)

        st.write("Video Duration :", _duration_strings)
        st.write("Width, Height and FPS :", _width, _height, _fps)

        inferenceBarText = "Performing inference on video, please wait."
        inferenceBar = st.progress(0, text=inferenceBarText)

        # SRIMS Analytics Placeholders
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            rhi_metric = st.empty()
        with metrics_col2:
            cost_metric = st.empty()

        # Initial metrics values
        current_rhi = 100
        current_cost = 0

        # Function to format cost in Indian Rupee format
        def format_inr(number):
            s, *d = str(number).partition(".")
            r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
            return "₹" + "".join([r] + d)

        rhi_metric.metric("Road Health Index (RHI)", current_rhi)
        cost_metric.metric("Estimated Repair Cost", format_inr(current_cost))

        # Warning Placeholder
        warning_placeholder = st.empty()

        imageLocation = st.empty()

        # Folium Map Setup
        try:
            gps_data = pd.read_csv("gps_log.csv")
            start_lat = gps_data['lat'].iloc[0]
            start_lon = gps_data['lon'].iloc[0]
        except Exception as e:
            # Fallback to center of Hyderabad if CSV missing
            start_lat, start_lon = 17.44, 78.49
            gps_data = pd.DataFrame({'timestamp': [], 'lat': [], 'lon': []})

        m = folium.Map(location=[start_lat, start_lon], zoom_start=15)
        # Store the map placeholder
        map_placeholder = st.empty()
        # Initial render
        with map_placeholder:
            st_folium(m, height=400, width=700, returned_objects=[])

        # Track added pins to avoid duplicates on paused/slow frames
        added_pins = set()
        
        # Track processed object IDs to avoid counting the same defect across multiple frames
        processed_ids = set()

        # Issue with opencv-python with pip doesn't support h264 codec due to license, so we cant show the mp4 video on the streamlit in the cloud
        # If you can install the opencv through conda using this command, maybe you can render the video for the streamlit
        # $ conda install -c conda-forge opencv
        # fourcc_mp4 = cv2.VideoWriter_fourcc(*'h264')
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        cv2writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

        # Read until video is completed
        _frame_counter = 0
        while(videoCapture.isOpened()):
            ret, frame = videoCapture.read()
            if ret == True:
                
                # Convert color-chanel
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MiDaS Depth Estimation
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                input_batch = midas_transforms(frame).to(device)
                with torch.no_grad():
                    prediction = midas(input_batch)
                    
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                depth_map = prediction.cpu().numpy()
                depth_map_resized = cv2.resize(depth_map, (640, 640), interpolation=cv2.INTER_AREA)

                # Perform inference
                _image = np.array(frame)

                image_resized = cv2.resize(_image, (640, 640), interpolation = cv2.INTER_AREA)
                results = net.track(image_resized, conf=score_threshold, persist=True, tracker="bytetrack.yaml")
                
                # Save the results
                frame_potholes = 0
                frame_cracks = 0
                two_wheeler_hazard_detected = False
                
                detections = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    has_masks = result.masks is not None
                    masks_data = result.masks.data.cpu().numpy() if has_masks else None
                    
                    # Extract tracking IDs from the prediction result
                    track_ids = [None] * len(boxes)
                    if result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                        
                    for i, _box in enumerate(boxes):
                        class_id = int(_box.cls[0])
                        track_id = track_ids[i]
                        
                        # Area and Depth Calculation
                        if has_masks and masks_data is not None:
                            mask = masks_data[i] > 0.5
                            pixel_area = np.sum(mask)
                            average_depth = np.mean(depth_map_resized[mask]) if pixel_area > 0 else 0.0
                        else:
                            box_coords_f = _box.xyxy[0]
                            pixel_area = (box_coords_f[2] - box_coords_f[0]) * (box_coords_f[3] - box_coords_f[1])
                            average_depth = 10.0 # arbitrary fallback
                            
                        # Determine Hazard Label based on volumetric risk
                        hazard = "Heavy Vehicle Hazard"
                        if average_depth > 500.0:
                            hazard = "Severe Two-Wheeler Hazard"
                            two_wheeler_hazard_detected = True
                        elif average_depth > 300.0:
                            hazard = "Two-Wheeler Hazard"
                            two_wheeler_hazard_detected = True

                        # Only calculate cost/RHI penalty if we have not tracked this specific ID before
                        if track_id is None or track_id not in processed_ids:
                            if track_id is not None:
                                processed_ids.add(track_id)
                                
                            volume_estimate = pixel_area * average_depth
                                
                            if class_id == 3: # Potholes
                                frame_potholes += 1
                                # Flat baseline repair cost + dynamically scaled cost for volume
                                current_cost += 500 + (volume_estimate * 0.05) 
                            elif class_id in [0, 1, 2]: # Cracks
                                frame_cracks += 1
                                current_cost += 100 + (volume_estimate * 0.02)
                            
                        # Calculate box width
                        box_coords = _box.xyxy[0].astype(int)
                        
                        # ResNet Cropping & Inference
                        x1, y1, x2, y2 = box_coords
                        y1, y2 = max(0, y1), min(frame.shape[0], y2)
                        x1, x2 = max(0, x1), min(frame.shape[1], x2)
                        
                        crop = frame[y1:y2, x1:x2]
                        
                        hazard_prob = 0.0
                        if crop.size > 0:
                            crop_tensor = resnet_transforms(crop).unsqueeze(0).to(device)
                            with torch.no_grad():
                                output = hazard_classifier(crop_tensor)
                                probs = torch.nn.functional.softmax(output, dim=1) 
                                hazard_prob = probs[0][1].item()
                                
                        if hazard_prob > 0.80:
                            hazard = "Critical Risk"
                            two_wheeler_hazard_detected = True
                        else:
                            pass # default fall back to MiDaS hazard logic
                        
                        # Overload the result object's name for plotting
                        # Ultralytics uses the model's names dictionary for plotting
                        orig_label = CLASSES[class_id]
                        new_label = f"{orig_label} - {hazard} ({hazard_prob:.2f})"
                        # A hack to temporarily override the class name for this specific result object's plotting
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
                    current_cost += (frame_potholes * 1500 + frame_cracks * 500)
                    rhi_metric.metric("Road Health Index (RHI)", current_rhi)
                    cost_metric.metric("Estimated Repair Cost", format_inr(current_cost))
                    
                if two_wheeler_hazard_detected:
                    warning_placeholder.warning("High Risk for Bikes/Scooters Detected!", icon="⚠️")
                else:
                    warning_placeholder.empty()

                # Geospatial Mapping Update
                if (frame_potholes > 0 or frame_cracks > 0) and not gps_data.empty:
                    # Calculate current video timestamp in seconds
                    current_vs_time = _frame_counter / _fps
                    # Find nearest timestamp in GPS log
                    idx = (np.abs(gps_data['timestamp'] - current_vs_time)).idxmin()
                    nearest_lat = gps_data.loc[idx, 'lat']
                    nearest_lon = gps_data.loc[idx, 'lon']
                    
                    loc_tuple = (nearest_lat, nearest_lon)
                    if loc_tuple not in added_pins:
                        folium.Marker(
                            [nearest_lat, nearest_lon],
                            popup="Defect Detected",
                            icon=folium.Icon(color="red", icon="info-sign"),
                        ).add_to(m)
                        added_pins.add(loc_tuple)
                        # Re-render map in placeholder
                        with map_placeholder:
                            st_folium(m, height=400, width=700, returned_objects=[])

                annotated_frame = results[0].plot()
                
                # Restore original class names to model so it doesn't permanently overwrite the dictionary
                for i, name in enumerate(CLASSES):
                    net.names[i] = name
                    if len(results) > 0:
                        results[0].names[i] = name
                
                _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation = cv2.INTER_AREA)

                print(_image_pred.shape)
                
                # Write the image to file
                _out_frame = cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR)
                cv2writer.write(_out_frame)
                
                # Display the image
                imageLocation.image(_image_pred)

                _frame_counter = _frame_counter + 1
                inferenceBar.progress(_frame_counter/_frame_count, text=inferenceBarText)
            
            # Break the loop
            else:
                inferenceBar.empty()
                break

        # When everything done, release the video capture object
        videoCapture.release()
        cv2writer.release()

    # Download button for the video
    st.success("Video Processed!")

    col1, col2 = st.columns(2)
    with col1:
        # Also rerun the appplication after download
        with open(temp_file_infer, "rb") as f:
            st.download_button(
                label="Download Prediction Video",
                data=f,
                file_name="RDD_Prediction.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
    with col2:
        if st.button('Restart Apps', use_container_width=True, type="primary"):
            # Rerun the application
            st.rerun()

st.title("Road Damage Detection - Video")
st.write("Detect the road damage in using Video input. Upload the video and start detecting. This section can be useful for examining and process the recorded videos.")

video_file = st.file_uploader("Upload Video", type=".mp4", disabled=st.session_state.runningInference)
st.caption("There is 1GB limit for video size with .mp4 extension. Resize or cut your video if its bigger than 1GB.")

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=st.session_state.runningInference)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction. You can change the threshold before running the inference.")

if video_file is not None:
    if st.button('Process Video', use_container_width=True, disabled=st.session_state.runningInference, type="secondary", key="processing_button"):
        _warning = "Processing Video " + video_file.name
        st.warning(_warning)
        processVideo(video_file, score_threshold)