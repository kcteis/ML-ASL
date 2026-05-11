import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# ==============================
# PARAMETERS
# ==============================
DATASET_PATH = "asl_alphabet_train"
IMG_SIZE = 512
OUTPUT_CSV = "asl_landmark_features.csv"

# ==============================
# INITIALIZE MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==============================
# EXTRACT HAND LANDMARK FEATURES
# ==============================
def extract_landmarks(image):
    """
    Extracts 21 hand landmarks (x, y, z)
    Total features = 21 * 3 = 63
    """

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image
    results = hands.process(image_rgb)

    # If hand detected
    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        features = []

        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        return features

    # If no hand detected
    return None

# ==============================
# PROCESS DATASET
# ==============================
def process_dataset(dataset_path):

    data = []
    labels = []

    classes = sorted(os.listdir(dataset_path))

    print(f"Found {len(classes)} classes")

    for label in classes:

        class_path = os.path.join(dataset_path, label)

        if not os.path.isdir(class_path):
            continue

        image_files = os.listdir(class_path)

        print(f"\nProcessing class: {label}")
        print(f"Images found: {len(image_files)}")

        success_count = 0
        failed_count = 0

        for img_name in image_files:

            img_path = os.path.join(class_path, img_name)

            try:
                # Read image
                image = cv2.imread(img_path)

                if image is None:
                    failed_count += 1
                    continue

                # Resize image
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

                # Denoise
                image = cv2.GaussianBlur(image, (3, 3), 0)

                # Improve contrast
                image = cv2.convertScaleAbs(image, alpha=1.2, beta=15)
                
                # Extract landmarks
                features = extract_landmarks(image)

                if features is not None:
                    data.append(features)
                    labels.append(label)
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failed_count += 1

        print(f"Successful: {success_count}")
        print(f"Failed: {failed_count}")

    return np.array(data), np.array(labels)

# ==============================
# MAIN
# ==============================
print("Starting feature extraction...")

X, y = process_dataset(DATASET_PATH)

print("\nFeature Extraction Complete")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)

# ==============================
# CREATE DATAFRAME
# ==============================

# Create column names
columns = []

for i in range(21):
    columns.extend([
        f"x_{i}",
        f"y_{i}",
        f"z_{i}"
    ])

# Create dataframe
df = pd.DataFrame(X, columns=columns)

# Add labels
df["label"] = y

# ==============================
# SAVE FEATURES
# ==============================
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nFeatures saved to: {OUTPUT_CSV}")
print(df.head())