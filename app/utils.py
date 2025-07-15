import cv2

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return img
