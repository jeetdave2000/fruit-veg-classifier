
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Manually list class names (sorted alphabetically)
class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 
               'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 
               'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 
               'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 
               'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 
               'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
               'sweetpotato', 'tomato', 'turnip', 'watermelon']

st.set_page_config(page_title="Fruit & Veg Classifier 🍎🥦", layout="centered")
st.title("🍓 Fruit & Vegetable Classifier")
st.write("Upload an image of a fruit or vegetable, and the model will classify it!")

@st.cache_resource
def load_model_cached():
    return load_model("mobilenet.h5")

model = load_model_cached()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    pred_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}%)")


