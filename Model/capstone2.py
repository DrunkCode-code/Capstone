# -*- coding: utf-8 -*-
"""Capstone2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gjH3MjxcCcSFt4PATZ_5j8zCjt4HuugO
"""

!pip install tensorflow-gpu

import tensorflow as tf
tf.__version__

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = [150, 150]

train_path = '/content/drive/MyDrive/Dataset/Training'
valid_path = '/content/drive/MyDrive/Dataset/Validation'

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in inception.layers:
  layer.trainable = False

folders = glob('/content/drive/MyDrive/Dataset/Training/*')

folders

x = Flatten()(inception.output)

prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=inception.input, outputs=prediction)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = validation_datagen.flow_from_directory(valid_path,
                                                        target_size = (150, 150),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')

r = model.fit_generator(
    training_set,
    validation_data=validation_set,
    epochs=10,
    steps_per_epoch=len(training_set),
    validation_steps=len(validation_set)
)

import matplotlib.pyplot as plt

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

validation_set.class_indices

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

uploaded = files.upload()

for fn in uploaded.keys():

  # predicting images
  path = fn
  img = image.load_img(path, target_size=(150, 150))
  imgplot = plt.imshow(img)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])/255
  classes = model.predict(images, batch_size=10)
  labels = ['Anggur','Brokoli','Gandum','Jagung','Kapas','Kubis','Padi','Rami','Tebu','Zaitun']

  print(fn)
  i = np.argmax(classes)
  print(classes)
  print(labels[i])

from tensorflow.keras.models import load_model

# Save the entire model as a SavedModel.
!mkdir -p saved_model
model.save('saved_model/my_model')