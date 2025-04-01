# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
# Import the correctly named functions
from skimage.feature import graycomatrix, graycoprops  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Define the root path to your dataset folder
# Update this with your actual dataset path
dataset_path = '/content/drive/MyDrive/Patients_CT'  # Example path in Google Colab; replace with your path
image_size = (512, 512)

# Recursively retrieve image files from all nested directories
image_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_files.append(os.path.join(root, file))

# Preprocessing Function
def load_image(path):
    try:
        image = Image.open(path).convert('L')  # Grayscale conversion
        image = image.resize(image_size)
        image = np.array(image) / 255.0  # Normalize to [0,1]
        return image
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

# Load all images
images = []
for image_path in image_files:
    image = load_image(image_path)
    if image is not None:
        images.append(image)

images = np.array(images)
print(f"Loaded {len(images)} images")

# Feature Extraction
def extract_features(image):
    features = {}
    features['mean'] = np.mean(image)
    features['std'] = np.std(image)
    glcm = greycomatrix((image * 255).astype(np.uint8), distances=[1], angles=[0], levels=256)
    features['contrast'] = greycoprops(glcm, 'contrast')[0, 0]
    features['homogeneity'] = greycoprops(glcm, 'homogeneity')[0, 0]
    features['energy'] = greycoprops(glcm, 'energy')[0, 0]
    features['correlation'] = greycoprops(glcm, 'correlation')[0, 0]
    return features

# Extract features for all images
features = [extract_features(image) for image in images]
features_df = pd.DataFrame(features)
print("Features extracted")

# Simulate Labels and Train Classifier
# Since no labels/masks are provided, simulate binary labels (0 = no hemorrhage, 1 = hemorrhage)
np.random.seed(42)
labels = np.random.randint(0, 2, size=len(images))  # Random labels for demo

# Train Random Forest Classifier
X_train, X_val, y_train, y_val = train_test_split(features_df, labels, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_val, y_val)
print(f"Random Forest Validation Accuracy: {accuracy}")
