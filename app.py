import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import os
model_path = r"plant_disease_cnn.h5"
class_names_path = r"plant_disease_class_names.pkl"

try:
    model = tf.keras.models.load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()
try:
    with open(class_names_path, 'rb') as f:
        class_names = pickle.load(f)
    if len(class_names) != 38:
        st.warning(f"Found {len(class_names)} classes, expected 38.")
except Exception as e:
    st.error(f"Error loading class names: {e}")
    st.stop()

# 3. Streamlit app
st.title("Plant Disease Classifier")
st.write("Upload a leaf image to predict the plant disease (or healthy).")
st.write("Supported formats: JPG, JPEG, PNG")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    try:
        image = image.resize((128, 128))
        image = image.convert("RGB")
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        st.write(f"**Prediction**: {predicted_class}")
        st.write(f"**Confidence**: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")

st.write("---")
st.write("**Instructions**:")
st.write("- Upload a clear image of a plant leaf.")
st.write("- The model will predict one of 38 classes (e.g., `Tomato___healthy`, `Apple___Black_rot`).")
st.write("- Ensure the image is well-lit and focused for best results.")