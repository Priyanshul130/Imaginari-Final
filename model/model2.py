import os
import tensorflow as tf
import numpy as np

# Assuming NMS function exists (you may need to implement this or import it)
def nms(preds1, preds2, preds3):
    # Placeholder NMS logic: averaging the predictions
    return np.mean([preds1, preds2, preds3], axis=0)

# Assuming visualize_detections function exists to visualize the final predictions
def visualize_detections(predictions, image):
    # Placeholder visualization logic
    print(f"Predictions: {predictions}")
    # Visualization can be done using Matplotlib, OpenCV, etc.

data_directory = ""

for filename in os.listdir(data_directory):
    if filename.endswith('.bin'):   
        # Load the model (you can modify to load multiple models as needed)
        model = tf.keras.models.load_model(os.path.join(data_directory, filename))
        
        # Load the doodle image and preprocess it
        image = tf.io.read_file('doodle.jpg')
        image = tf.image.decode_jpeg(image, channels=3)  # Assuming the image is in JPEG format
        image = tf.image.resize(image, (224, 224))
        image = image / 255.0  # Normalize the image to [0, 1]
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

        # Assuming you have already loaded model1, model2, and model3 somewhere before this loop
        preds1 = model1.predict(image)
        preds2 = model2.predict(image)
        preds3 = model3.predict(image)

        # Combine the predictions using non-maximum suppression (NMS)
        final_pred = (preds1+preds2+preds3)/3
        print(final_pred)
