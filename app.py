import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import importlib.util

# Load preprocess() from app/utils.py
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'utils.py'))
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
preprocess = utils.preprocess

# Load model and class labels
model = load_model('model/sign_model.h5')
labels = sorted(os.listdir('dataset'))  # Uses folder names as class labels

# Streamlit UI
st.title("🤟 Sign Language Translator")
st.write("Upload an image of a hand sign:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to OpenCV format
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    img = preprocess(img_cv)

    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_label = labels[np.argmax(prediction)]

    st.success(f"Predicted Sign: **{predicted_label}**")
