import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models



def load_data_from_bins(data_directory):
    images = []
    labels = []

    for filename in os.listdir(data_directory):
        if filename.endswith('.bin'):
            object_name = filename.split('.')[0]
            data = np.fromfile(os.path.join(data_directory, filename), dtype=np.float32)
            images.append(data.reshape(256, 256, 1))  # Adjust according to your data
            labels.append(object_name)

    return np.array(images), np.array(labels)

# Load and preprocess data
data_directory = 'E:\git\Imaginari\.quickdrawcache'  # Update this path
images, labels = load_data_from_bins(data_directory)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Normalize images
images = images.astype('float32') / 255.0

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(np.unique(labels_encoded)), activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Real-time prediction
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, 64, 64, 1)

    # Predict the gesture
    prediction = model.predict(reshaped_frame)
    predicted_class = prediction.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]  # Decode the class label

    # Display the prediction
    cv2.putText(frame, f"Gesture: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
