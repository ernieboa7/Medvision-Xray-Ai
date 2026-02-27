# --------------------------------------------------
# Suppress TensorFlow Logs (MUST be before TF import)
# --------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

MODEL_PATH = Path("models/xray_resnet50_clean.keras")

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        safe_mode=False,
        custom_objects={"preprocess_input": preprocess_input}
    )    

model = load_model()

# --------------------------------------------------
# 🔍 RECURSIVE Conv2D Finder (NEVER FAILS)
# --------------------------------------------------
def find_last_conv_layer(layer):

    # If layer is Conv2D
    if isinstance(layer, tf.keras.layers.Conv2D):
        return layer

    # If layer has sublayers (nested model)
    if hasattr(layer, "layers"):
        for sublayer in reversed(layer.layers):
            result = find_last_conv_layer(sublayer)
            if result is not None:
                return result

    return None


# --------------------------------------------------
# 💡 PRODUCTION-SAFE GRAD-CAM
# --------------------------------------------------
def make_gradcam_heatmap(img_array, model):

    last_conv_layer = None

    # Search recursively through model
    for layer in reversed(model.layers):
        last_conv_layer = find_last_conv_layer(layer)
        if last_conv_layer is not None:
            break

    # If no conv layer exists → skip safely
    if last_conv_layer is None:
        return None

    try:
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            class_channel = predictions[:, 0]

        grads = tape.gradient(class_channel, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        return heatmap.numpy()

    except Exception:
        # If anything goes wrong → fail gracefully
        return None


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

    # Basic validation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.std(gray) < 10:
        st.error("Invalid or blank image detected.")
        st.stop()

    if img.shape[0] < 200 or img.shape[1] < 200:
        st.warning("Image resolution is very low. Results may be unreliable.")

    # Preprocess
    img_resized = cv2.resize(img, (224, 224))
    img_array = preprocess_input(img_resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):

        # 🔥 Use direct call (more stable than predict())
        #prediction = float(model(img_array, training=False)[0][0])
        prediction = float(model.predict(img_array, verbose=0)[0][0])

        pneumonia_prob = prediction
        normal_prob = 1 - prediction

        st.subheader("Prediction Result")

        if pneumonia_prob > 0.65:
            label = "PNEUMONIA"
            confidence = pneumonia_prob
            st.error("PNEUMONIA DETECTED")
            st.info("⚠️ The model suggests possible signs of pneumonia.")

        elif normal_prob > 0.65:
            label = "NORMAL"
            confidence = normal_prob
            st.success("NORMAL")
            st.info("No significant signs of pneumonia detected.")

        else:
            label = "LOW CONFIDENCE"
            confidence = max(pneumonia_prob, normal_prob)
            st.warning("LOW CONFIDENCE RESULT")

        confidence = min(confidence, 0.999)
        st.write(f"Confidence: {confidence*100:.2f}%")

        # --------------------------------------------------
        # Grad-CAM (Now NEVER crashes)
        # --------------------------------------------------
        if label != "LOW CONFIDENCE":

            heatmap = make_gradcam_heatmap(img_array, model)

            if heatmap is not None:

                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

                st.image(superimposed, caption="Grad-CAM Visualization")

            else:
                st.warning("Grad-CAM visualization not available for this model architecture.")