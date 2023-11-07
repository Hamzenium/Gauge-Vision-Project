import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PySimpleGUI as sg
from keras.utils import load_img, img_to_array
import os

train_dir = os.path.join('/Users/muhammadhamzasohail/Desktop/display_train')
train__dir = os.path.join('/Users/muhammadhamzasohail/Desktop/display')


img = mpimg.imread(path_img)
imgplot = plt.imshow(img)
plt.show()

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
e_pix = [os.path.join(train_horse_dir, fname) 
                for fname in segments[pic_index-8:pic_index]]
next__pix = [os.path.join(train_human_dir, fname) 
                for fname in segments[pic_index-8:pic_index]]


model = tf.keras.models.Sequential([

    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])



from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255

path = os.path.join('/Users/muhammadhamzasohail/Desktop/horse-or-human')

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        path,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 300x300
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=5,
      verbose=1)




img = load_img(path_img, target_size=(300, 300))
x = img_to_array(img)
x /= 255
x = np.expand_dims(x, axis=0)
classes = model.predict(x)

        
