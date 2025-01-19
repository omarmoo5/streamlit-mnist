import time

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="mnist_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the canvas image to fit the TFLite model
def preprocess_image(image):
    # Convert to RGB (3 channels)
    image = image.convert("RGB")
    # Resize to 300x300 (model's expected input size)
    image = image.resize((300, 300))
    # Normalize pixel values to 0-1
    image_array = np.array(image).astype(np.float32) / 255.0
    # Add batch dimension
    image_array = image_array.reshape(1, 300, 300, 3)
    return image_array

# Check if the image is empty (blank canvas)
def is_image_blank(image):
    # Convert image to grayscale
    grayscale_image = ImageOps.grayscale(image)
    # Compute the sum of all pixel values
    pixel_sum = np.array(grayscale_image).sum()
    # If the sum is 0, the image is blank
    return pixel_sum == 0

# Predict the digit using the TFLite model and return the predicted digit with its confidence
def predict_digit(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), np.max(output_data)

# Streamlit app setup
st.title("MNIST Digit Classifier")

# Sidebar controls for the canvas
st.sidebar.header("Canvas Settings")
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FFFFFF")
bg_color = st.sidebar.color_picker("Background color hex: ", "#000000")
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create the canvas
st.write("### Draw a digit on the canvas below:")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Transparent fill color
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=300,
    drawing_mode="freedraw",
    update_streamlit=realtime_update,
    key="canvas",
)

interpreter = load_tflite_model()

# Display prediction
if canvas_result.image_data is not None:
    # Display the drawing
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))

    if not is_image_blank(img) and realtime_update or st.button("Predict"):
        input_image = preprocess_image(img)
        try:
            t1 = time.perf_counter_ns()
            digit, confidence = predict_digit(interpreter, input_image)
            t2 = time.perf_counter_ns()
            st.write(f"### Predicted Digit: {digit}")
            st.write(f"### Confidence: {confidence*100:.2f}%")
            st.write(f"### Time taken: {(t2-t1)/1e6:.2f} ms")
        except Exception as e:
            st.write("An error occurred during prediction.")
            st.write(e)