import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import importlib.util

utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app', 'utils.py'))
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
preprocess = utils.preprocess

# Load model
model = load_model('model/sign_model.h5')

# Get labels from dataset folders
labels = sorted(os.listdir('dataset'))

# Streamlit UI
st.title("🤟 Real-Time Sign Language Translator")

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    img = preprocess(roi)

    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_label = labels[np.argmax(prediction)]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, predicted_label, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
