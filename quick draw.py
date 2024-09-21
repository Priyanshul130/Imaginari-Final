import tensorflow as tf

# Load the models from their bin files
model1 = tf.keras.models.load_model('.quickdrawcache/anvil.bin')


# Load the doodle image and preprocess it
image = tf.io.read_file('doodle.jpg')
image = tf.image.resize(image, (224, 224))
image = image / 255.0
image = tf.expand_dims(image, axis=0)

# Make predictions on the doodle image using each model
preds1 = model1.predict(image)


# Combine the predictions using non-maximum suppression (NMS)
final_pred = preds1

# Visualize the final prediction
visualize_detections(final_pred, image)