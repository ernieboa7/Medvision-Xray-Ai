import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="MedVision AI", page_icon="🩺")

# --------------------------------------------------
# Professional Header
# --------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1f4e79, #2c7be5);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
">
    <h1 style="margin-bottom: 0.5rem;">
        🩺 MedVision - X-ray Diagnosis Demo
    </h1>
    <p style="
        background-color: rgba(255,255,255,0.15);
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 14px;
        margin-top: 0.5rem;
    ">
        Educational use only. Not medical diagnosis.
    </p>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/xray_resnet50.keras")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# Grad-CAM (Proper Graph Model)
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = "conv5_block3_out"

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer("resnet50").get_layer(last_conv_layer_name).output,
         model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# --------------------------------------------------
# Upload Section
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", width=400)

    # -----------------------------
    # Basic Image Validation
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 10:
        st.error("Invalid or blank image detected.")
        st.stop()

    if img.shape[0] < 200 or img.shape[1] < 200:
        st.warning("Image resolution is very low. Results may be unreliable.")

    # -----------------------------
    # Preprocess
    # -----------------------------
    img_resized = cv2.resize(img, (224, 224))
    img_array = preprocess_input(img_resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict"):

        prediction = float(model.predict(img_array, verbose=0)[0][0])

        pneumonia_prob = prediction
        normal_prob = 1 - prediction

        st.subheader("Prediction Result")

        # -----------------------------
        # Intelligent Decision Logic
        # -----------------------------
        if pneumonia_prob > 0.65:
            label = "PNEUMONIA"
            confidence = pneumonia_prob

            st.error("PNEUMONIA DETECTED")
            st.info(
                "⚠️ The model suggests possible signs of pneumonia. "
                "Please consult a qualified medical professional."
            )

        elif normal_prob > 0.65:
            label = "NORMAL"
            confidence = normal_prob

            st.success("NORMAL")
            st.info(
                "No significant signs of pneumonia detected by this model."
            )

        else:
            label = "LOW CONFIDENCE"
            confidence = max(pneumonia_prob, normal_prob)

            st.warning("LOW CONFIDENCE RESULT")
            st.info(
                "The model is uncertain. This image may not be a typical chest X-ray "
                "or may contain features outside the training dataset."
            )

        st.write(f"Confidence: {confidence*100:.2f}%")

        # -----------------------------
        # Grad-CAM (Only if confident)
        # -----------------------------
        if label != "LOW CONFIDENCE":

            heatmap = make_gradcam_heatmap(img_array, model)

            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

            st.image(superimposed, caption="Grad-CAM Visualization")
