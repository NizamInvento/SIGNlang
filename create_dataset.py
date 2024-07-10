import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

# Iterate through each directory in the data directory
for dir_ in os.listdir(DATA_DIR):
    label = dir_  # Label for the current class
    label_dir = os.path.join(DATA_DIR, dir_)
    
    # Iterate through each image in the current class directory
    for img_name in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_name)
        
        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe Hands
        results = hands.process(img_rgb)
        
        # Check if hands are detected in the image
        if results.multi_hand_landmarks:
            # Concatenate features of all detected hands into a single feature vector
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = [lm.x for lm in hand_landmarks.landmark]
                y_ = [lm.y for lm in hand_landmarks.landmark]
                
                # Calculate centroid of the hand
                centroid_x = np.mean(x_)
                centroid_y = np.mean(y_)
                
                # Normalize hand landmarks with respect to the centroid
                normalized_x = [x - centroid_x for x in x_]
                normalized_y = [y - centroid_y for y in y_]
                
                # Add normalized hand landmarks to data_aux
                data_aux.extend(normalized_x)
                data_aux.extend(normalized_y)
                
            # Pad or truncate data_aux to a fixed length (if necessary)
            max_length = 42  # Example: Set max_length to 42
            if len(data_aux) < max_length:
                data_aux += [0] * (max_length - len(data_aux))
            elif len(data_aux) > max_length:
                data_aux = data_aux[:max_length]
                
            # Append data and label to lists
            data.append(data_aux)
            labels.append(label)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)


print(len(data))
print(len(labels))