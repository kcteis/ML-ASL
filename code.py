import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# CONFIG
DATASET_PATH = "asl_dataset_train" 
IMG_SIZE = 64 
USE_PCA = True
PCA_COMPONENTS = 100 


# LOAD DATA
def load_data(dataset_path, img_size):
    images = []
    labels = []

    # Sort classes for consistency
    classes = sorted(os.listdir(dataset_path))

    print(f"Found {len(classes)} classes.")

    for label in classes:
        class_path = os.path.join(dataset_path, label)

        # Skip non-folder files
        if not os.path.isdir(class_path):
            continue

        image_files = os.listdir(class_path)
        print(f"Loading class '{label}' with {len(image_files)} images")

        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)

            try:
                # Read image
                img = cv2.imread(img_path)

                if img is None:
                    continue  # skip unreadable files

                # Resize
                img = cv2.resize(img, (img_size, img_size))

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                images.append(gray)
                labels.append(label)

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    return np.array(images), np.array(labels)


print("Loading dataset...")
X, y = load_data(DATASET_PATH, IMG_SIZE)
print(f"Dataset loaded: {X.shape}")