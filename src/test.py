import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../models/mnist_model.h5")

# Load MNIST test data
_, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0  # Normalize

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
