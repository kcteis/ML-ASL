import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

model = joblib.load("reports/SVM_model.joblib")
encoder = joblib.load("reports/label_encoder.joblib")
scaler = joblib.load("reports/feature_scaler.joblib")   
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

#Start Video Capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot access webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    #Draw hand Landmarks and Predict ASL Letter
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([
                landmark.x, 
                landmark.y, 
                landmark.z
            ])
        
        # Predict the ASL letter
        features_array = np.array(features).reshape(1, -1)
        features_array = scaler.transform(features_array)
        prediction = model.predict(features_array)
        predicted_letter = encoder.inverse_transform(prediction)[0]

        # Display the predicted letter on the frame
        cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the video feed
    cv2.imshow('ASL Recognition', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()