from PIL import Image
import numpy as np
import os

def normalize_and_resize(image_path, target_size):
    # Load the image
    image = Image.open(image_path)

    # Normalize the pixel values to be in the range [0, 1]
    normalized_image = np.array(image) / 255.0

    # Resize the image to a fixed resolution
    resized_image = image.resize(target_size)

    # Alternatively, you can resize the normalized numpy array
    resized_normalized_image = Image.fromarray((normalized_image * 255).astype(np.uint8)).resize(target_size)

    return resized_image, resized_normalized_image

if __name__ == "__main__":
    # Replace this with the path to the folder containing your images
    image_folder = 'archive/chest_xray/test/PNEUMONIA'
    
    target_size = (150, 150)  # A tuple of integers representing the desired fixed resolution

    # List all the files in the image folder
    image_files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)]

    # Create a directory to save the resized images
    output_folder = "resized_images_test"
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all the images and resize each one
    for i, image_path in enumerate(image_files):
        resized_image, resized_normalized_image = normalize_and_resize(image_path, target_size)

        # Save the resized images
        resized_image.save(os.path.join(output_folder, f"PNEUMONIA{i}.jpg"))
        resized_normalized_image.save(os.path.join(output_folder, f"NORMAL{i}.jpg"))
