import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

# Download model from Google Drive if not already present
model_path = "trained_model.h5"
if not os.path.exists(model_path):
    st.info("Model file not found. Downloading from Google Drive...")
    url = "https://drive.google.com/file/d/1AW50CAeO0_w5L0zUAnQNNabT7L5mJ2Pb/view?usp=sharing"  # Replace with your actual file ID
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("Model downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download model: {e}")

# Tensorflow Model Prediction
def model_prediction(test_image):
    if not os.path.exists("trained_model.h5"):
        st.error("Model file not found. Please check if it was downloaded properly.")
        return -1  # Invalid prediction index

    model = tf.keras.models.load_model("trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

# Main Page
if app_mode == "Home":
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.warning("Home image not found. Please upload 'home_img.jpg' in your repo.")

# About Project
elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Model is predicting...")
            result_index = model_prediction(test_image)

            if result_index != -1:
                try:
                    with open("labels.txt") as f:
                        label = [line.strip() for line in f.readlines()]
                    st.success(f"Model is predicting it's a **{label[result_index]}**")
                except Exception as e:
                    st.error("Error reading labels.txt: {}".format(e))
