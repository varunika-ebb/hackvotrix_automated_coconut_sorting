import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import math

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="🥥 Coconut Analyzer", layout="wide")

# ---------------------------
# Load YOLO Models
# ---------------------------
@st.cache_resource
def load_models():
    model_coconut = YOLO("best (2).pt")   # Coconut detection & weight
    model_position = YOLO("best-pos.pt")  # Position detection
    return model_coconut, model_position

model_coconut, model_position = load_models()

# ---------------------------
# Coconut Detection Helpers (from app.py)
# ---------------------------
def estimate_weight(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    width_px = x2 - x1
    height_px = y2 - y1
    width_cm = width_px * 0.02
    height_cm = height_px * 0.02
    depth_cm = width_cm
    volume = (4/3) * math.pi * (width_cm/2) * (height_cm/2) * (depth_cm/2)
    weight = volume * 1.0  
    weight_scaled = np.interp(weight, [100, 5000], [400, 1000])
    return round(weight_scaled, 2)

def draw_boxes_coconut(results, frame):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            weight = estimate_weight(box)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model_coconut.names[cls_id]} | {conf:.2f} | {weight} g"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame

# ---------------------------
# Position Detection Helpers (from app1.py but only Browse)
# ---------------------------
def draw_boxes_position(results, frame):
    h, w, _ = frame.shape
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dir_x = "Left" if cx < w // 2 else "Right"
            dir_y = "Top" if cy < h // 2 else "Bottom"
            direction = f"{dir_y}-{dir_x}"

            cls_name = model_position.names[int(cls)]
            color = (0, 255, 0) if "good" in cls_name.lower() else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{cls_name} {conf:.2f} Dir:{direction}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("🥥 Navigation")
choice = st.sidebar.radio("Go to:", ["Coconut Detection", "Position Detection"])

# ---------------------------
# 1️⃣ Coconut Detection (UNCHANGED from app.py)
# ---------------------------
if choice == "Coconut Detection":
    st.title("🥥 Automated Coconut Sorting - YOLO Model")
    st.write("Upload an image or use live camera for automatic detection and weight estimation.")

    option = st.sidebar.radio("📌 Select Mode", ["Browse Image", "Live Camera Detection"])

    # Image upload
    if option == "Browse Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            img_cv = np.array(img)
            results = model_coconut.predict(img_cv, conf=0.5)

            annotated_frame = draw_boxes_coconut(results, img_cv.copy())
            st.image(annotated_frame, caption="Detection Result", use_column_width=True)

            st.subheader("🔎 Detection Summary")
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    weight = estimate_weight(box)

                    st.markdown(
                        f"""
                        <div style="
                            background: #f0f9ff;
                            border-left: 6px solid #00796b;
                            padding: 10px;
                            margin-bottom: 10px;
                            border-radius: 8px;">
                            <b>Class:</b> {model_coconut.names[cls_id]}<br>
                            <b>Confidence:</b> {conf:.2f}<br>
                            <b>Estimated Weight:</b> {weight} g
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    # Live camera
    elif option == "Live Camera Detection":
        run_camera = st.checkbox("Start Live Camera Detection", key="live_camera_checkbox")

        if run_camera:
            cap = cv2.VideoCapture(0)
            camera_placeholder = st.empty()

            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model_coconut.predict(img_rgb, conf=0.5)
                annotated_frame = draw_boxes_coconut(results, img_rgb.copy())

                camera_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)
                run_camera = st.session_state.get("live_camera_checkbox")

            cap.release()
            cv2.destroyAllWindows()

# ---------------------------
# 2️⃣ Position Detection (Browse Image ONLY)
# ---------------------------
elif choice == "Position Detection":
    st.title("🥥 Coconut Detection & Direction Finder")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        results = model_position.predict(img_cv)
        annotated_frame = draw_boxes_position(results, img_cv.copy())

        st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption="Detection Result")
