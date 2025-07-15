import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os
import importlib.util

# Dynamically load utils
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'utils.py'))
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
preprocess = utils.preprocess

# Load model
model = load_model('model/sign_model.h5')
labels = sorted(os.listdir('dataset')) if os.path.exists('dataset') else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

st.title("🤟 Sign Language Translator")
uploaded_file = st.file_uploader("Upload a hand sign image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = preprocess(img_cv)
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    predicted_label = labels[np.argmax(prediction)]

    st.success(f"Predicted Sign: **{predicted_label}**")
