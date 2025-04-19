import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Print TensorFlow version for debugging
st.write(f"TensorFlow Version: {tf.__version__}")

# Define a custom TrueDivide layer
class TrueDivideLayer(Layer):
    def __init__(self, **kwargs):
        super(TrueDivideLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Handle single input (e.g., input / constant) or two inputs (x / y)
        if isinstance(inputs, (list, tuple)):
            x, y = inputs
            return tf.math.truediv(x, y)
        else:
            # Assume division by a constant (e.g., 127.5 for MobileNetV2 preprocessing)
            return tf.math.truediv(inputs, tf.constant(127.5, dtype=inputs.dtype))

    def get_config(self):
        config = super(TrueDivideLayer, self).get_config()
        return config

# Function to load and predict
@st.cache_resource
def load_model():
    try:
        with custom_object_scope({'TrueDivide': TrueDivideLayer}):
            model = tf.keras.models.load_model("mobilenet.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def model_prediction(image):
    model = load_model()
    if model is None:
        return None
    img = load_img(image, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return np.argmax(predictions[0])

# Class names (based on your dataset with 36 classes)
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
    'grapes', 'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
    'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
    'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
]

# Streamlit app interface
st.title("Fruit and Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    result_index = model_prediction(uploaded_file)
    if result_index is not None:
        result = class_names[result_index]
        st.write(f"Prediction: **{result}**")
    else:
        st.error("Failed to make a prediction due to model loading error.")
