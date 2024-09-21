import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load the trained model from the binary file
model = keras.models.load_model('anvil.bin')

# Read the binary file
with open('img.bin', 'rb') as f:
    image_data = f.read()

# Convert the binary data to a numpy array
image_array = np.frombuffer(image_data, dtype=np.uint8)

# Reshape the array to the correct dimensions
image_array = image_array.reshape((28, 28))

# Normalize the pixel values
image_array = image_array / 255.0

# Create an input tensor for the model
input_tensor = image_array.reshape((1, 28, 28, 1))

# Make a prediction using the model
prediction = model.predict(input_tensor)

# Print the prediction
print(prediction)