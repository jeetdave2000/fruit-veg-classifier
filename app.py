import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("mobilenet_final.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

from collections import Counter
import os
#///addd this new///
# This works if your data is in folders per class
train_dir = "/your/train/path"
class_counts = {folder: len(os.listdir(os.path.join(train_dir, folder))) for folder in os.listdir(train_dir)}

df_dist = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])

st.subheader("📊 Class Distribution in Training Data")

# Plotly Pie Chart
fig = px.pie(df_dist, names='Class', values='Count', title='Class Distribution')
st.plotly_chart(fig, use_container_width=True)
###///end////

#Main Page
if(app_mode=="Home"):
    st.header("FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    st.image(image_path)

#About Project
elif(app_mode=="About Project"):
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

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
   # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")

        # Load model
        model = tf.keras.models.load_model("mobilenet_final.keras")

        # Prepare image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)[0]

        # Read labels
        with open("labels.txt") as f:
            label = [line.strip() for line in f.readlines()]

        result_index = np.argmax(prediction)
        predicted_label = label[result_index]
        st.success(f"Model is predicting it's a **{predicted_label}**")

        # 🎯 Visualization of confidence scores
        import pandas as pd
        import plotly.express as px

        df_plot = pd.DataFrame({
            'Class': label,
            'Confidence': prediction * 100
        })

        df_plot = df_plot.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

        st.subheader("🔍 Prediction Confidence by Class")

        fig = px.bar(df_plot.head(10), x='Confidence', y='Class', orientation='h',
                     color='Confidence', color_continuous_scale='Viridis',
                     title='Top 10 Prediction Probabilities')

        st.plotly_chart(fig, use_container_width=True)
