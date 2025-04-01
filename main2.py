import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

# Define the root path to your dataset folder
# For example, if your dataset is located at "C:/Datasets/CT_Images", update the line below accordingly.
dataset_path = '/content/drive/MyDrive/Patients_CT'  # Update with your actual dataset path

# Recursively retrieve image files from all nested directories
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_files.append(os.path.join(root, file))

print("Total images found:", len(image_files))

# Set the target size as required by VGG-16
target_size = (224, 224)  # Standard input size for VGG-16

# Load the pre-trained VGG-16 model (excluding the top classification layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(target_size[0], target_size[1], 3))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

def preprocess_image(image_path):
    """
    Load and preprocess an image:
      - Resizes the image to the target size.
      - Converts the image to a numpy array.
      - Expands the dimensions to match model input requirements.
      - Applies VGG16-specific preprocessing.
    """
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Check if any image files were found
if image_files:
    # Process the first image in the list as an example
    sample_image = image_files[0]
    preprocessed_img = preprocess_image(sample_image)
    features = model.predict(preprocessed_img)

    print("Feature shape:", features.shape)  # Expected shape: (1, 7, 7, 512) for VGG-16 block5_pool

    # Visualize one channel of the feature map (e.g., channel 0)
    plt.imshow(features[0, :, :, 0], cmap='viridis')
    plt.title("Feature Map - Channel 0")
    plt.colorbar()
    plt.show()
else:
    print("No images found in the specified dataset path!")
