import cv2
import numpy as np

def preprocess(frame):
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0
    return frame
