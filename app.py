import streamlit as st
import numpy as np
import cv2

from src.utils.model_loader import build_model
from src.utils.gradcam import make_gradcam_heatmap, overlay_heatmap

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MedVision AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MedVision - X-ray Diagnosis Demo")
st.write("Upload a chest X-ray image for AI-based pneumonia screening.")

st.warning(
    "⚠️ This demo is for educational purposes only and is NOT a medical diagnosis."
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_ai_model():
    model = build_model()
    model.load_weights("models/xray_model.h5")
    return model

model = load_ai_model()

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # Decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", width=500)

    # Preprocess
    img_resized = cv2.resize(img, (128, 128))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    if st.button("Predict"):

        prediction = model.predict(img_resized)[0][0]

        label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        st.subheader("Prediction Result")

        if label == "PNEUMONIA":
            st.error(f"🫁 {label}")
        else:
            st.success(f"✅ {label}")

        st.write(f"Confidence: {confidence * 100:.2f}%")

        # -------------------------------------------------
        # AI SUGGESTION
        # -------------------------------------------------
        st.markdown("### 🤖 AI Suggestion")

        if label == "PNEUMONIA":
            st.write(
                "The AI model detects patterns consistent with pneumonia. "
                "A clinical examination and professional medical consultation "
                "are recommended."
            )
        else:
            st.write(
                "No pneumonia-related patterns detected by the AI model. "
                "If symptoms persist, please consult a healthcare professional."
            )

        # -------------------------------------------------
        # GRAD-CAM HEATMAP
        # -------------------------------------------------
        try:
            heatmap = make_gradcam_heatmap(
                img_resized,
                model,
                last_conv_layer_name="sequential_3"
            )

            overlay = overlay_heatmap(heatmap, img)

            st.markdown("### 🔥 AI Attention Heatmap")
            st.image(overlay, caption="AI Focus Regions", width=500)

        except Exception:
            st.info("Heatmap visualization not available.")

        # -------------------------------------------------
        # INFORMATION LINKS
        # -------------------------------------------------
        st.markdown("---")
        st.markdown("### 📚 Learn More")

        st.markdown(
            "- [WHO Pneumonia Information](https://www.who.int/news-room/fact-sheets/detail/pneumonia)\n"
            "- [CDC Pneumonia Guide](https://www.cdc.gov/pneumonia/index.html)\n"
            "- [Chest X-ray Basics](https://radiopaedia.org/articles/chest-radiograph)"
        )
