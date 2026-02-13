import argparse
import numpy as np
import cv2

from src.utils.model_loader import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

print("Building model...")
model = build_model()
model.load_weights("models/xray_model.h5")

print("Reading image...")
img = cv2.imread(args.image)

# Resize to training size
img = cv2.resize(img, (128, 128))

img = img / 255.0
img = np.expand_dims(img, axis=0)

print("Predicting...")
prediction = model.predict(img)

#print("Prediction:", prediction)

score = float(prediction[0][0])

# Fake threshold interpretation
if score > 0.95:
    label = "PNEUMONIA"
else:
    label = "NORMAL"

confidence = score * 100

print(f"Prediction: {label}")
print(f"Confidence: {confidence:.2f}%")

print("Prediction value:", prediction)
