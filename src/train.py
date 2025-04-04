import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images (0-255 → 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Save the trained model
os.makedirs("../models", exist_ok=True)
model.save("../models/mnist_model.h5")

print("Model training complete. Saved in 'models/mnist_model.h5'")
