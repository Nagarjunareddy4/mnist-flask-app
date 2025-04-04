import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "../models/mnist_model.h5"
model = load_model(MODEL_PATH)

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28)  # Reshape for model input
    return image

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Process the image and make a prediction
        image = Image.open(file)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = np.argmax(prediction)

        return render_template("index.html", prediction=predicted_label)

    return render_template("index.html", prediction=None)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
