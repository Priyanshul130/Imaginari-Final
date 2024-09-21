import tensorflow as tf
import numpy as np
from quickdraw import QuickDrawDataGroup
import matplotlib.pyplot as plt



model = tf.keras.models.load_model("doodleNet-model.h5")

qd= QuickDrawDataGroup()

for category in qd.drawing_names:
    print(f"Category: {category}")
    doodle = qd.get_drawing(category)

# Preprocess the doodle image (resize, normalize, etc.)
image = doodle.get_image().resize((28, 28))  # Resize to 28x28 pixels for the model
image = np.array(image.convert('L'))  # Convert to grayscale if needed
image = image / 255.0  # Normalize the image (0-1 range)
image = image.reshape(1, 28, 28, 1)  # Reshape to match the input shape for the model (batch_size, height, width, channels)

# Display the preprocessed image (optional)
plt.imshow(image.squeeze(), cmap='gray')
plt.show()

# Make predictions using the pre-trained model
predictions = model.predict(image)
predicted_class = np.argmax(predictions, axis=1)

# Load the class names for the model (assuming it's saved in a text file)
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Print the predicted class label
predicted_label = class_names[predicted_class[0]]
print("Predicted Class: ", predicted_label)
