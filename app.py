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

st.title("🩺 MedVision - X-ray Diagnosis Demo")
st.warning("Educational use only. Not medical diagnosis.")

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
# SAFE Grad-CAM (Graph connected version)
# --------------------------------------------------

def make_gradcam_heatmap(img_array, model):

    # 1 Get ResNet backbone
    base_model = model.get_layer("resnet50")

    # 2 Get last conv layer
    last_conv_layer = base_model.get_layer("conv5_block3_out")

    # 3 Forward pass manually through backbone
    with tf.GradientTape() as tape:

        # Pass through backbone ONLY
        conv_outputs = base_model(img_array, training=False)

        # Watch conv outputs
        tape.watch(conv_outputs)

        # Now manually pass through classifier head
        x = conv_outputs
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("batch_normalization")(x)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("dense_1")(x)

        loss = predictions[:, 0]

    # 4 Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()
# --------------------------------------------------
# Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", width=400)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) < 10:
        st.error("Invalid or blank image detected.")
        st.stop()

    img_resized = cv2.resize(img, (224, 224))
    img_array = preprocess_input(img_resized.astype("float32"))
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):

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

        st.subheader("Prediction Result")

        
        if label == "PNEUMONIA":
            st.error("PNEUMONIA DETECTED")
            st.info(
            "⚠️ This ML model suggests possible signs of pneumonia. "
            "Please consult a qualified medical professional for proper evaluation and diagnosis."
            )

            st.markdown("### Suggested Reading")
            st.markdown(
                        """
                        - [WHO – Pneumonia Overview](https://www.who.int/news-room/fact-sheets/detail/pneumonia)
                        - [CDC – Pneumonia Information](https://www.cdc.gov/pneumonia/index.html)
                        - [Mayo Clinic – Pneumonia Symptoms & Treatment](https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/syc-20354204)
                        """
        )

        elif label == "NORMAL":
            st.success("NORMAL")
            st.info(
                    "No significant signs of pneumonia detected by this ML model. "
                    "If symptoms persist, please consult a healthcare provider."
                    )

        else:
            st.warning("UNCERTAIN RESULT")
            st.info(
                    "The model is not confident in this prediction. "
                    "Consider retesting with a clearer image or consult a medical professional."
                    )
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM Visualization")