import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../models/mnist_model.h5")

# Load MNIST test data
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalize

# Select an image for prediction
index = 0  # Change this value to test different images
img = x_test[index]

# Make a prediction
prediction = model.predict(np.expand_dims(img, axis=0))
predicted_label = np.argmax(prediction)

# Display the image with prediction
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted_label}, Actual: {y_test[index]}")
plt.axis('off')
plt.show()
