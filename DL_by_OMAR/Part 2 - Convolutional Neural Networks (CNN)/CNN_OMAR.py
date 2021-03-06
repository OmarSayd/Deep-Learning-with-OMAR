# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 17:40:40 2020

@author: CNN - Image calssification
"""

                                # PART 1
# Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step-1 Convolution
classifier.add(Convolution2D(filters=32, kernel_size=[3,3],
                             padding='valid', activation='relu',
                             input_shape = (64, 64, 3)))

# Step-2 Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step-3 Flattening
classifier.add(Flatten())

# Step-4 Fully Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

                               # PART 2
# Fitting the CNN to the IMAGES
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=5,                # nb_epoch = 25
                         validation_data=test_set,
                         nb_val_samples=2000)


                              # PART 3
# Making New Prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64 ,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = classifier.predict(test_image)
training_set.class_indices








                              
                              
                              
                              