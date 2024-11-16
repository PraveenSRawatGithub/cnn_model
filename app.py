from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Folder to save uploaded images inside the static folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_and_preprocess_image(image_path):
    """
    Preprocesses an image from the file system.
    Args:
        image_path (str): Path to the image file.
    Returns:
        numpy.ndarray: The preprocessed image as a numpy array.
    """
    img = Image.open(image_path)
    img_array = np.array(img.resize((150, 150))) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # Ensure the correct data type
    return img_array

def predict_image(image_array):
    """
    Predicts the class of an image using the TFLite model.
    Args:
        image_array (numpy.ndarray): Preprocessed image array.
    Returns:
        str: Predicted class label ('Cat' or 'Dog').
    """
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = (output_data > 0.5).astype(int)
    class_labels = {0: "Cat", 1: "Dog"}
    return class_labels[int(predicted_class[0][0])]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction=None, image_url=None)

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction=None, image_url=None)

        if file:
            # Save the file to the server
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            test_image = load_and_preprocess_image(file_path)

            # Make prediction
            result = predict_image(test_image)

            # Pass the prediction and image path to the template
            return render_template('index.html', prediction=result, image_url=file.filename)

    return render_template('index.html', prediction=None, image_url=None)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
