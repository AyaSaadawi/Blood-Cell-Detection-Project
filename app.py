import sys
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import tempfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if 'torch.classes' in sys.modules:
    del sys.modules['torch.classes']

model = YOLO('./model.pt')

st.title("🩸 Blood Cell Detection using YOLOv12")
st.sidebar.header("Upload Media")

uploaded_file = st.sidebar.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

def process_image(image):
    """Process an image and return detection results"""
    image_np = np.array(image)
    results = model(image_np)
    boxes = results[0].boxes
    
    for bbox in boxes:
        x1, y1, x2, y2 = bbox.xyxy[0].int().tolist()
        conf = bbox.conf[0].item()  
        cls = int(bbox.cls[0]) 

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, f'Class {cls} Conf: {conf:.2f}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), results

def process_video(video_path):
    """Process a video and return frames with detections"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    video_placeholder = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 640))
        
        results = model(frame)
        boxes = results[0].boxes
        
        for bbox in boxes:
            x1, y1, x2, y2 = bbox.xyxy[0].int().tolist()
            conf = bbox.conf[0].item()
            cls = int(bbox.cls[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {cls} Conf: {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        
        video_placeholder.image(frame_rgb, caption="Processed Video Frame")
    
    cap.release()
    return frames

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        st.subheader("📷 Uploaded Image")
        image = Image.open(uploaded_file).resize((640, 640))
        st.image(image, caption="Original Image", use_container_width=True)

        try:
            processed_image, results = process_image(image)
            st.success("✅ Prediction Successful!")
            st.image(processed_image, caption="Detection Result", use_column_width=True)
            
            st.write("📊 Prediction DataFrame:")
            st.dataframe(results[0].pandas().xyxy[0])

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            
    elif uploaded_file.type.startswith('video'):
        st.subheader("🎥 Uploaded Video")
        
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name)
        
        try:
            st.info("⏳ Processing video... This may take a while depending on video length.")
            processed_frames = process_video(tfile.name)
            st.success("✅ Video processing completed!")
            
            if st.button("💾 Save Processed Video"):
                height, width, _ = processed_frames[0].shape
                output_path = "processed_video.mp4"
                out = cv2.VideoWriter(output_path, 
                                     cv2.VideoWriter_fourcc(*'mp4v'), 
                                     30, (width, height))
                
                for frame in processed_frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                
                with open(output_path, "rb") as f:
                    st.download_button("⬇️ Download Processed Video", f, file_name="processed_video.mp4")
            
        except Exception as e:
            st.error(f"❌ Error during video processing: {str(e)}")

val_metrics_dict = {
    'metrics/precision(B)': 0.808,
    'metrics/recall(B)': 0.892,
    'metrics/mAP50(B)': 0.897,
    'metrics/mAP50-95(B)': 0.605,
    'fitness': 0.634
}

st.subheader("📈 Evaluation Metrics")
metrics_df = pd.DataFrame(list(val_metrics_dict.items()), columns=['Metric', 'Value'])
st.dataframe(metrics_df)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x='Metric', y='Value', data=metrics_df[metrics_df['Metric'].str.contains('mAP50')])
plt.title('mAP50 Values')
st.pyplot(plt)  