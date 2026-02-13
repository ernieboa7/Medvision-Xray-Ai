import streamlit as st
import numpy as np
import cv2
from src.utils.model_loader import build_model

st.title("MedVision - X-ray Diagnosis Demo")
st.write("Upload a chest X-ray image for prediction.")

@st.cache_resource
def load_ai_model():
    model = build_model()
    model.load_weights("models/xray_model.h5")
    return model

model = load_ai_model()

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width=500)

    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_resized)[0][0]

        label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader("Prediction Result")
        st.success(f"{label}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
