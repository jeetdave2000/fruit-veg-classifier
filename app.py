
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
  import matplotlib.pyplot as plt

# After prediction
predictions = model.predict(img_array)[0]
top_indices = predictions.argsort()[-5:][::-1]  # Top 5 predictions
top_classes = [class_names[i] for i in top_indices]
top_probs = [predictions[i] * 100 for i in top_indices]

st.success(f"Prediction: **{top_classes[0]}** ({top_probs[0]:.2f}%)")

# Plot the bar chart
fig, ax = plt.subplots()
bars = ax.barh(top_classes[::-1], top_probs[::-1], color='limegreen')
ax.set_xlabel("Confidence (%)")
ax.set_title("Top 5 Class Probabilities")
ax.invert_yaxis()  # Highest on top

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', va='center')

st.pyplot(fig)
import plotly.express as px

df = {
    'Class': top_classes,
    'Confidence': top_probs
}
fig = px.bar(df, x='Confidence', y='Class', orientation='h',
             title="Top 5 Class Probabilities", color='Confidence',
             color_continuous_scale='Blues')
st.plotly_chart(fig)

