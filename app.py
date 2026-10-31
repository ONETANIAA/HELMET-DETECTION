import os
import tempfile
from pathlib import Path
import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="HelmetVision | YOLOv8", page_icon="🪖", layout="wide")

MODEL_PATH = Path(__file__).parent / "best.pt"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(circle at 85% 4%,rgba(104,76,190,.2),transparent 28%),linear-gradient(135deg,#090d1a,#10172b 55%,#111329);color:#f5f6ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#10172b,#17152f);border-right:1px solid #ffffff18}
.block-container{padding-top:2rem;max-width:1400px}
.hero{padding:1.7rem 1.9rem;border:1px solid #b0a0ff38;border-radius:24px;background:linear-gradient(115deg,#2a3664d1,#4d307e99);margin-bottom:1.4rem}
.eyebrow{color:#c3b7ff;font-size:.78rem;font-weight:800;letter-spacing:.16em}
.hero h1{color:white;font-size:clamp(2rem,4vw,3.1rem);margin:.35rem 0}
.hero p,.muted{color:#cbd2ef}
.metric{background:linear-gradient(145deg,#273158e6,#241e46e6);border:1px solid #b0a0ff33;border-radius:16px;padding:1rem;min-height:100px}
.metric-label{color:#b9c2e5;font-size:.8rem;font-weight:600}
.metric-value{color:white;font-size:1.8rem;font-weight:800}
div[data-testid="stFileUploader"]{background:#161e37a0;border:1px dashed #ae9eff70;border-radius:16px;padding:.7rem}
.stButton button,.stDownloadButton button{border:0;border-radius:11px;font-weight:700;background:linear-gradient(90deg,#6d5ce7,#9a64dc);color:white}
button[data-baseweb="tab"]{background:#212a4bb3;border-radius:10px 10px 0 0;color:#cbd2ef}
button[data-baseweb="tab"][aria-selected="true"]{background:#5547a7;color:white}
footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(path):
    return YOLO(path)

if not MODEL_PATH.exists():
    st.error("File best.pt tidak ditemukan. Letakkan best.pt satu folder dengan app.py.")
    st.stop()
try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

def class_name(class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, f"Class {class_id}"))
    return str(names[class_id]) if class_id < len(names) else f"Class {class_id}"

def get_counts(result):
    counts = {}
    if result.boxes is not None and result.boxes.cls is not None:
        for cid in result.boxes.cls.detach().cpu().numpy().astype(int):
            counts[cid] = counts.get(cid, 0) + 1
    return counts

def show_metrics(counts):
    cols = st.columns(3)
    items = [("TOTAL DETECTIONS", sum(counts.values()))]
    ids = sorted(counts)
    for i in range(2):
        cid = ids[i] if i < len(ids) else None
        items.append((class_name(cid).upper() if cid is not None else "OTHER / NONE",
                      counts.get(cid, 0) if cid is not None else 0))
    for col, (label, value) in zip(cols, items):
        col.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🪖 HelmetVision")
    st.caption("YOLOv8 Computer Vision System")
    st.divider()
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.25, 0.05)
    st.markdown("### Model information")
    st.markdown("**YOLOv8 Custom Model**")
    st.caption("Classes: With Helmet · Without Helmet")
    st.caption("Upload an image or video to start detection.")

st.markdown("""
<div class="hero">
<div class="eyebrow">COMPUTER VISION • YOLOv8</div>
<h1>Helmet Detection System</h1>
<p>Detect motorcycle riders with and without helmets from images and videos.
Upload your media, run the model, and review the detection results.</p>
</div>
""", unsafe_allow_html=True)

image_tab, video_tab, about_tab = st.tabs(["🖼️ Image Detection", "🎞️ Video Detection", "ⓘ About"])

with image_tab:
    st.subheader("Analyze an image")
    upload = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"], key="img")
    if upload:
        image = Image.open(upload).convert("RGB")
        left, right = st.columns(2, gap="large")
        left.markdown("**Original image**")
        left.image(image, use_container_width=True)
        if st.button("Run image detection", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with YOLOv8..."):
                result = model.predict(np.array(image), conf=confidence, verbose=False)[0]
                annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                counts = get_counts(result)
            right.markdown("**Detection result**")
            right.image(annotated, use_container_width=True)
            st.markdown("### Detection summary")
            show_metrics(counts)
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button("Download result image", buf.getvalue(), "helmet_detection_result.png", "image/png")
    else:
        st.info("Upload an image to begin detection.")

with video_tab:
    st.subheader("Analyze a video")
    upload_video = st.file_uploader("Upload video", type=["mp4","mov","avi","mkv"], key="vid")
    if upload_video:
        st.video(upload_video)
        if st.button("Run video detection", type="primary", use_container_width=True):
            input_path = output_path = None
            cap = writer = None
            try:
                suffix = Path(upload_video.name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(upload_video.getbuffer())
                    input_path = f.name
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise RuntimeError("Video tidak dapat dibuka. Coba gunakan MP4.")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    output_path = f.name
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
                if not writer.isOpened():
                    raise RuntimeError("Gagal membuat video hasil.")
                progress = st.progress(0, text="Processing video...")
                counts = {}
                frame_no = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = model.predict(frame, conf=confidence, verbose=False)[0]
                    writer.write(result.plot())
                    for cid, n in get_counts(result).items():
                        counts[cid] = counts.get(cid, 0) + n
                    frame_no += 1
                    if total > 0:
                        progress.progress(min(frame_no / total, 1.0), text=f"Processing frame {frame_no}/{total}")
                cap.release(); cap = None
                writer.release(); writer = None
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.success(f"Detection completed — {frame_no} frames processed.")
                st.markdown("### Processed video")
                st.video(video_bytes)
                st.download_button("Download detected video", video_bytes, "helmet_detection_result.mp4", "video/mp4", use_container_width=True)
                st.markdown("### Detection summary")
                st.caption("Counts are accumulated across frames, not unique riders.")
                show_metrics(counts)
            except Exception as e:
                st.error(f"Video processing failed: {e}")
            finally:
                if cap is not None: cap.release()
                if writer is not None: writer.release()
                for p in (input_path, output_path):
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except OSError: pass
    else:
        st.info("Upload a video to preview it and run helmet detection.")

with about_tab:
    st.subheader("About this project")
    st.markdown("""
    **Helmet Detection using YOLOv8**

    This application uses a custom-trained YOLOv8 model to detect motorcycle riders:
    - **With Helmet**
    - **Without Helmet**

    Supported images: JPG, JPEG, PNG, WEBP.  
    Supported videos: MP4, MOV, AVI, MKV.

    Results may vary depending on image quality, camera angle, lighting, and confidence threshold.
    """)

st.markdown("<p style='text-align:center;color:#7883a8;font-size:.78rem;margin-top:2rem'>HelmetVision • YOLOv8 Helmet Detection</p>", unsafe_allow_html=True)
