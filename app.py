import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Disease names (example classes)
class_names = ['Apple Scab', 'Apple Black Rot', 'Healthy', 'Corn Blight', 'Potato Late Blight']

# Function to predict the disease
def predict_disease(image, model):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize the image to the input size expected by the model
    img_array = np.asarray(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100  # Confidence percentage
    return predicted_class, confidence

# Recommendation system for each disease
def recommend_treatment(disease):
    recommendations = {
        'Apple Scab': 'Use fungicides like Captan or Mancozeb to control Apple Scab. Regular pruning helps with air circulation.',
        'Apple Black Rot': 'Remove infected plant parts and spray fungicides such as Captan or Benomyl.',
        'Healthy': 'No action needed. Your plant is healthy!',
        'Corn Blight': 'Use resistant varieties and fungicides like Mancozeb. Rotate crops regularly.',
        'Potato Late Blight': 'Use fungicides like Chlorothalonil or Copper fungicides. Remove affected plants promptly.',
    }
    return recommendations.get(disease, "No recommendation available")

# Streamlit app setup
st.title("Plant Disease Detection and Recommendation System")

st.write("Upload an image of the plant's leaf, and the model will detect the disease and recommend treatments.")

# Image upload and prediction
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    predicted_class, confidence = predict_disease(image, model)

    # Display the results
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Provide recommendations
    st.write("**Treatment Recommendation:**")
    st.write(recommend_treatment(predicted_class))

