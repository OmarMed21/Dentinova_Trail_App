import streamlit as st
from ultralytics import YOLO
import cv2
import torch
import numpy as np

torch.cuda.empty_cache()

def generate_rgb_colors(num_colors=12, saturation=255, value=255):
    """Generates a list of random RGB colors with adjustable saturation and value."""
    colors = []
    for _ in range(num_colors):
        hue = np.random.randint(0, 180)
        colors.append(cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0].tolist())
    return colors

colors = generate_rgb_colors(12) 
model = YOLO('teeth_detection.pt')
st.title("Teeth F Detection App")
st.markdown("""
**Class Descriptions:**

- **0:** Sound Tooth Structure
- **1:** Faint Visual Change in Enamel
- **2:** Distinct Visual Change in Enamel
- **3:** Localized Enamel Breakdown
- **5:** Distinct Cavity
- **6:** Extensive Distinct Cavity
""") 

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame, verbose=False)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if score > 0.2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(class_id)], 2)
                cv2.putText(frame, str(class_id), (x1 + 10, y1 + 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1, colors[int(class_id)], 1)
                # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption=uploaded_file.name, use_column_width=False) 
