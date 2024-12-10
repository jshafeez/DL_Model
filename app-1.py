import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Path to the pre-trained model
MODEL_PATH = "D:/mammals/deep/mammals_model.h5"

# Load the pre-trained model
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# Class labels for prediction
class_labels = [
    'african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox',
    'armadillo', 'baboon', 'badger', 'blue_whale', 'brown_bear', 'camel'
]

# Streamlit Application Title
st.title("Mammals Image Classification")
st.write("Upload an image of a mammal, and the model will predict its species.")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing the image...")

        # Preprocess the image for the model
        img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust size as per model requirements
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize pixel values

        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]  # Get the class with the highest score
        confidence = np.max(predictions)  # Get the highest confidence score

        # Display the results
        st.success(f"**Predicted Class:** {predicted_class}")
        st.info(f"**Confidence Score:** {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
else:
    st.write("Please upload an image file to get started.")
