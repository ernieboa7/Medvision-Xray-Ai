import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="MedVision AI", page_icon="🩺", layout="wide")

# --------------------------------------------------
# Load External CSS
# --------------------------------------------------
def load_css():
    with open("styles/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# Header Section
# --------------------------------------------------
st.markdown("""
<div class="medvision-header">
    <h1>🩺 MedVision AI</h1>
    <p>AI-Powered Chest X-ray Pneumonia Detection</p>
    <small>Educational use only. Not a medical diagnosis.</small>
</div>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/xray_resnet50.keras")

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'preprocess_input': preprocess_input}
    )

model = load_model()

# --------------------------------------------------
# Grad-CAM
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model):
    base_model = model.get_layer("resnet50")
    last_conv_layer = base_model.get_layer("conv5_block3_out")

    with tf.GradientTape() as tape:
        conv_outputs = base_model(img_array, training=False)
        tape.watch(conv_outputs)

        x = conv_outputs
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense_1")(x)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

# --------------------------------------------------
# Main Card Layout
# --------------------------------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 10:
        st.error("Invalid or blank image detected.")
        st.stop()

    img_resized = cv2.resize(img, (224, 224))
    img_array = preprocess_input(img_resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Run AI Diagnosis"):

        with st.spinner("Analyzing X-ray..."):
            prediction = float(model.predict(img_array, verbose=0)[0][0])

        if prediction > 0.8:
            label = "PNEUMONIA"
            confidence = prediction
        elif prediction < 0.2:
            label = "NORMAL"
            confidence = 1 - prediction
        else:
            label = "UNCERTAIN"
            confidence = abs(prediction - 0.5) * 2

        with col2:
            st.subheader("Diagnosis Result")

            if label == "PNEUMONIA":
                st.error("Pneumonia Detected")
            elif label == "NORMAL":
                st.success("Normal")
            else:
                st.warning("Uncertain Result")

            st.markdown(
                f'<div class="confidence-box">Confidence: {confidence*100:.2f}%</div>',
                unsafe_allow_html=True
            )

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM Visualization", use_column_width=True)

st.markdown('</div>', unsafe_allow_html=True)