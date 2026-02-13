import numpy as np
import tensorflow as tf
from tensorflow import keras
from build_test_cnn import model
#from train_cnn import *
#from gantest2 import *

from PIL import Image
import cv2




# Load your GAN and CNN models
gan_model = keras.models.load_model('gan_model.h5')  # Replace with your GAN model file
cnn_model = keras.models.load_model('xray_model.h5')  # Replace with your CNN model file

# Set hyperparameters
num_iterations = 5
batch_size = 20
latent_dim = 100

# Initialize an empty list to hold the dataset
initial_dataset = []

# Iterate the process
for i in range(num_iterations):
    # Generate fake pneumonia images using the GAN
    generated_data = gan_model.predict(np.random.randn(batch_size, latent_dim))
    
    # Analyze fake images using the CNN
    generated_data = (generated_data + 1) / 2.0  # Normalize to [0, 1]
    resized_data = []
    for img in generated_data:
        resized_img = cv2.resize(img, (200, 200))
        resized_data.append(np.array(resized_img))

    predictions = cnn_model.predict(np.array(resized_data))
    
    
    #predictions = cnn_model.predict(generated_data)
    
    # Append generated data to the initial dataset list
    initial_dataset = np.concatenate(initial_dataset, resized_data)
    loss, acc, prec, rec = model.evaluate(initial_dataset)
    
    # Print the number of generated images added to the dataset
    print(f"Iteration {i+1} - Generated Data Added: {len(generated_data)} images")
    
    print("-" * 40)

# Concatenate all generated data arrays into a single array
initial_dataset = np.concatenate(initial_dataset, resized_data)