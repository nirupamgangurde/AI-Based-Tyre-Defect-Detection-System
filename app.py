import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained Keras model (.keras extension)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('tyre.h5')  # Replace with your actual model filename
    return model

model = load_model()

# Class labels (ensure the order matches your training labels)
class_names = ['Defective', 'Good']  # Adjust if needed

# App title
st.title("🛞 Tyre Quality Classifier")
st.write("Upload a tyre image to classify it as **Good** or **Defective**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Image preprocessing
def preprocess_image(image, target_size=(224, 224)):  # Change target_size if needed
    image = image.resize(target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0  # Normalization
    return image_array

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Tyre Image', use_container_width=True)

    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")
