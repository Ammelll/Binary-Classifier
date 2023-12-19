from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import imutils

import cv2
img = cv2.imread("stitch.png")
img = imutils.resize(img,768,768)

input_img = np.expand_dims(img, axis=0)

model = load_model('panorama_model_100.h5')
print(model.predict(input_img))
