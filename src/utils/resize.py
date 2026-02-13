from PIL import Image
import numpy as np

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
    image_path = r'archive\chest_xray\val\NORMAL\NORMAL2-IM-1427-0001.jpeg'
    target_size = (150, 150)  # A tuple of integers representing the desired fixed resolution

    resized_image, resized_normalized_image = normalize_and_resize(image_path, target_size)

    # Save the resized images if needed
    resized_image.save("path_to_save_resized_image.jpg")
    resized_normalized_image.save("path_to_save_resized_normalized_image.jpg")



