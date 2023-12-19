import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.style.use('classic')

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense

import os
import cv2
from PIL import Image
import numpy as np
import imutils

image_directory = 'all_images/'
SIZE=768
dataset = []
label = []

good_images = os.listdir(image_directory + 'good/')
for i, image_name in enumerate(good_images):   
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'good/' + image_name)
        image = imutils.resize(image,SIZE,SIZE)
        dataset.append(image)
        label.append(0)

bad_images = os.listdir(image_directory + 'bad/')
for i, image_name in enumerate(bad_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'bad/' + image_name)
        image = imutils.resize(image,SIZE,SIZE)
        dataset.append(image)
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

from keras.utils import normalize
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)


INPUT_SHAPE = (int(SIZE/6), SIZE, 3)  


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))  
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',           
              metrics=['accuracy'])

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 49, 
                         verbose = 1, 
                         epochs = 1,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )


model.save('malaria_model_10epochs.h5') 


img = X_test[0]
plt.imshow(img)
plt.show()