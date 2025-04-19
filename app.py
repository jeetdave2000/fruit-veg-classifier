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
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        with open("labels.txt") as f:
            content = f.readlines()
        label = []
        for i in content:
            label.append(i[:-1])
        st.success("Model is Predicting it's a {}".format(label[result_index]))

## visualization
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# After predicting on the image
prediction = model.predict(img_array)[0]  # get 1D array if using model.predict(img_array)

# Get class labels from your training generator or model metadata
class_labels = list(class_names.values())  # or use train.class_indices

# Create a dataframe for plotting
df_plot = pd.DataFrame({
    'Class': class_labels,
    'Confidence': prediction * 100
})

# Sort by confidence
df_plot = df_plot.sort_values(by='Confidence', ascending=False).reset_index(drop=True)

# Streamlit section
st.subheader("🔍 Prediction Confidence by Class")

# Plot with Plotly
fig = px.bar(df_plot.head(10), x='Confidence', y='Class', orientation='h', color='Confidence',
             color_continuous_scale='Viridis', title='Top 10 Prediction Probabilities')

st.plotly_chart(fig, use_container_width=True)
