import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import asyncio

# Async function to load model
@st.cache_resource
async def load_model():
    model = tf.keras.models.load_model('https://github.com/dheerajreddy71/diseasedetection/blob/main/model.h5')
    return model

# Async function for prediction
async def predict_disease(image, model):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Disease names
class_names = ['Apple Scab', 'Apple Black Rot', 'Healthy', 'Corn Blight', 'Potato Late Blight']

# Recommendations for each disease
def recommend_treatment(disease):
    recommendations = {
        'Apple Scab': 'Use fungicides like Captan or Mancozeb.',
        'Apple Black Rot': 'Remove infected plant parts and spray fungicides.',
        'Healthy': 'No action needed.',
        'Corn Blight': 'Use resistant varieties and fungicides.',
        'Potato Late Blight': 'Use Chlorothalonil or Copper fungicides.',
    }
    return recommendations.get(disease, "No recommendation available")

# Streamlit app setup
st.title("Plant Disease Detection and Recommendation")

st.write("Upload an image to detect the disease and get recommendations.")

# Load the model asynchronously
model = asyncio.run(load_model())

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    predicted_class, confidence = asyncio.run(predict_disease(image, model))

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write("**Treatment Recommendation:**")
    st.write(recommend_treatment(predicted_class))
