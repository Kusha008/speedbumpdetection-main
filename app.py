# import streamlit as st
# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO
# import threading
# import time

# # Enable OpenCV optimizations
# cv2.setUseOptimized(True)

# # Load YOLOv8 model (same model and weights as before)
# model = YOLO("best.pt")

# st.title("Real-Time Object Detection with YOLOv8")
# st.sidebar.write("### Camera Options")

# # Select camera index
# camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)

# # Confidence threshold slider (for label display, if desired)
# conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# # Open the webcam
# cap = cv2.VideoCapture(camera_index)
# if not cap.isOpened():
#     st.error("Could not open the webcam.")
#     st.stop()

# # Try reducing capture buffer
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# # Global variables for sharing frames and detection results between threads
# latest_frame = None
# latest_detection = None
# frame_lock = threading.Lock()

# def capture_frames():
#     """Continuously capture frames from the webcam."""
#     global latest_frame
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             with frame_lock:
#                 latest_frame = frame
#         else:
#             break
#         # Yield briefly to allow other threads to run
#         time.sleep(0.005)

# def detect_frames():
#     """Continuously run detection on the latest captured frame."""
#     global latest_detection
#     while cap.isOpened():
#         with frame_lock:
#             if latest_frame is None:
#                 continue
#             # Work on a copy so that detection is decoupled from capture
#             frame_copy = latest_frame.copy()
#         # Run detection (this may be slow, but it's in its own thread)
#         results = model(frame_copy)[0]
#         latest_detection = results
#         # Yield briefly
#         time.sleep(0.005)

# # Start the capture and detection threads
# capture_thread = threading.Thread(target=capture_frames, daemon=True)
# detection_thread = threading.Thread(target=detect_frames, daemon=True)

# capture_thread.start()
# detection_thread.start()

# # Streamlit placeholder for displaying video
# stframe = st.empty()

# while True:
#     with frame_lock:
#         if latest_frame is None:
#             continue
#         # Copy the current frame for display
#         display_frame = latest_frame.copy()
#         detection = latest_detection

#     # If detection results are available, overlay the bounding boxes and labels
#     if detection is not None:
#         for box in detection.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             class_id = int(box.cls[0])
#             label = f"{model.names[class_id]}: {conf:.2%}"
#             cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(
#                 display_frame, label, (x1, max(0, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
#             )

#     # Convert the frame to RGB for proper display in Streamlit
#     display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
#     stframe.image(display_frame, use_column_width=True)

import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import threading
import time
import gc

# ---------------------------
# CONFIGURATION
# ---------------------------
MAX_DISPLAY_FPS = 30  # Throttle the Streamlit display to 20 fps (feel free to adjust)
DETECTION_SLEEP  = 0.001  # Sleep in detection thread (small yield)
CAPTURE_SLEEP    = 0.001  # Sleep in capture thread (small yield)
CACHE_CLEAR_INTERVAL = 15 # Clear CUDA/memory every 30 detections

# ---------------------------
# MODEL LOADING
# ---------------------------
cv2.setUseOptimized(True)
model = YOLO("best.pt")

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("Real-Time Object Detection with YOLOv8")

st.sidebar.write("### Camera Options")
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2], index=0)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ---------------------------
# OPEN THE CAMERA
# ---------------------------
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    st.error("Could not open the webcam.")
    st.stop()

# Attempt to reduce buffering (may not work on all systems/cameras)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ---------------------------
# THREAD-SHARED VARIABLES
# ---------------------------
latest_frame = None       # Holds the most recent camera frame
latest_detections = None  # Holds the most recent YOLO detection results
frame_lock = threading.Lock()
stop_threads = False      # Flag to stop threads cleanly

# ---------------------------
# CAPTURE THREAD
# ---------------------------
def capture_frames():
    global latest_frame, stop_threads

    while not stop_threads and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Store the latest frame in a thread-safe manner
        with frame_lock:
            latest_frame = frame

        time.sleep(CAPTURE_SLEEP)

# ---------------------------
# DETECTION THREAD
# ---------------------------
def detect_frames():
    global latest_detections, stop_threads

    detect_count = 0
    while not stop_threads and cap.isOpened():
        frame_copy = None
        # Grab the latest frame
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()

        if frame_copy is not None:
            # Run YOLO detection
            results = model(frame_copy)[0]
            latest_detections = results

            detect_count += 1
            # Periodically clear cache/memory to avoid slowdowns
            if detect_count % CACHE_CLEAR_INTERVAL == 0:
                gc.collect()
                torch.cuda.empty_cache()

        time.sleep(DETECTION_SLEEP)

# ---------------------------
# START THREADS
# ---------------------------
capture_thread = threading.Thread(target=capture_frames, daemon=True)
detection_thread = threading.Thread(target=detect_frames, daemon=True)
capture_thread.start()
detection_thread.start()

# Placeholder in Streamlit to display frames
stframe = st.empty()

# ---------------------------
# MAIN LOOP (DISPLAY)
# ---------------------------
last_display_time = 0
try:
    while True:
        now = time.time()
        # Throttle the display to avoid overloading Streamlit’s rendering pipeline
        if (now - last_display_time) < (1.0 / MAX_DISPLAY_FPS):
            time.sleep(0.001)
            continue
        last_display_time = now

        frame_for_display = None
        detections_for_display = None

        # Safely retrieve the latest frame and detection
        with frame_lock:
            if latest_frame is not None:
                frame_for_display = latest_frame.copy()
            if latest_detections is not None:
                detections_for_display = latest_detections

        if frame_for_display is None:
            # No frame yet, just loop
            continue

        # If we have detections, overlay them
        if detections_for_display is not None:
            for box in detections_for_display.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                # Only draw if above user-selected confidence threshold
                if conf >= conf_threshold:
                    label = f"{model.names[class_id]}: {conf:.2%}"
                    cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame_for_display, label, 
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2
                    )

        # Convert BGR to RGB for proper display
        frame_for_display = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)

        # Show in Streamlit
        stframe.image(frame_for_display, use_column_width=True)

except KeyboardInterrupt:
    pass
finally:
    # Stop the threads
    stop_threads = True
    capture_thread.join()
    detection_thread.join()
    cap.release()
