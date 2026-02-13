# MedVision – Chest X-ray Diagnosis Demo

MedVision is a deep learning web app that predicts pneumonia from chest X-ray images.

## Features
- CNN-based X-ray classification
- Streamlit web demo
- Upload and predict in browser
- Confidence score output

## Demo
Upload a chest X-ray and get instant prediction.

## Tech Stack
- Python
- TensorFlow / Keras
- Streamlit
- OpenCV

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
Project Structure
app.py          → Web app
src/train       → Model training
models          → Saved model
sample_images   → Test samples
Example Output
Prediction: Pneumonia
Confidence: 95%

Author
Ernest Eboagwu


---

# ✅ Step 6 — Initialize Git Repository

Inside project root:

```bash
git init
git add .
git commit -m "Initial commit - MedVision X-ray AI"