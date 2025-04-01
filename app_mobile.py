# app_mobile.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import time
from ultralytics import YOLO
import nest_asyncio
nest_asyncio.apply()


# Import streamlit-webrtc for live camera feed
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------
# Page configuration and styling
# -----------------------------
st.set_page_config(page_title="Speed Bump Detection", layout="centered")
st.markdown("""
    <style>
    .main-header {
        font-size: 1.8rem;
        margin-bottom: 0.8rem;
    }
    .stButton button {
        width: 100%;
        height: 3rem;
        font-size: 1rem;
    }
    </style>
    <h1 class="main-header">Speed Bump Detection</h1>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar Settings
# -----------------------------
with st.expander("⚙️ Settings"):
    model_type = "best.pt"
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.35, 0.05)
    iou_threshold = 0.45

# -----------------------------
# Load and Warm-up Model
# -----------------------------
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = model(dummy_img)  # warm-up
    return model

try:
    model = load_model(model_type)
    st.success(f"✅ Model {model_type} loaded")
except Exception as e:
    st.error(f"Error loading model: {e}")

# -----------------------------
# Helper functions for detection
# -----------------------------
def detect_objects(image, conf, iou):
    results = model(image, conf=conf, iou=iou)
    return results[0]

def display_results(results, image):
    res_plotted = results.plot()
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    st.image(res_plotted_rgb, caption="Detection Results", use_column_width=True)
    if len(results.boxes) > 0:
        st.write(f"**Detected {len(results.boxes)} objects:**")
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])
            st.write(f"• {class_name}: {confidence:.2f}")
    else:
        st.info("No objects detected")

# -----------------------------
# YOLO Transformer for live video stream using streamlit-webrtc
# -----------------------------
class YOLOTransformer(VideoTransformerBase):
    def __init__(self):
        # Load a new instance of the model (this runs in a separate thread)
        self.model = YOLO(model_type)
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        _ = self.model(dummy_img)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def transform(self, frame):
        # Get the current frame as a BGR image
        img = frame.to_ndarray(format="bgr24")
        results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
        annotated_frame = results[0].plot()
        return annotated_frame

# -----------------------------
# Tabs for different modes
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📸 Camera", "🖼️ Image", "🎥 Video"])

with tab1:
    st.write("Use your camera to detect objects in real-time")
    camera_mode = st.radio("Choose camera mode:", ["Take Picture", "Live Stream"], horizontal=True)
    
    if camera_mode == "Take Picture":
        camera_input = st.camera_input("Capture image")
        if camera_input is not None:
            image = np.array(Image.open(camera_input))
            with st.spinner("Detecting objects..."):
                results = detect_objects(image, conf_threshold, iou_threshold)
                display_results(results, image)
    else:
        st.write("Live Stream")
        # Use streamlit-webrtc for a live video feed with real-time processing
        webrtc_streamer(key="live", video_transformer_factory=YOLOTransformer)

with tab2:
    st.write("Upload an image to detect objects")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        if st.button("Detect Objects in Image", use_container_width=True):
            with st.spinner("Processing..."):
                results = detect_objects(image, conf_threshold, iou_threshold)
                display_results(results, image)
                with st.expander("Show Original Image"):
                    st.image(image, caption="Original Image", use_column_width=True)

with tab3:
    st.write("Upload a short video to detect objects")
    st.warning("For mobile devices, keep videos short (< 10s) for best performance")
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video = cv2.VideoCapture(tfile.name)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        if duration > 15:
            st.warning(f"Video is {duration:.1f} seconds long. Processing may be slow on mobile.")
        
        skip_frames = 5  # Process every 5th frame for mobile
        if st.button("Process Video", use_container_width=True):
            stframe = st.empty()
            progress_bar = st.progress(0)
            frame_idx = 0
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                if frame_idx % skip_frames == 0:
                    progress = int(min(frame_idx / frame_count * 100, 100))
                    progress_bar.progress(progress)
                    results = detect_objects(frame, conf_threshold, iou_threshold)
                    res_plotted = results.plot()
                    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    stframe.image(res_plotted_rgb, caption=f"Frame {frame_idx}", use_column_width=True)
                frame_idx += 1
                if frame_idx > 100:
                    st.info("Processing limited to first 100 frames for mobile performance")
                    break
            video.release()
            progress_bar.progress(100)
            st.success("✅ Video processing complete!")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.info("📱 Optimized for mobile devices")
st.caption("For more features, try the desktop version")
