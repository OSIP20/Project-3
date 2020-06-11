import numpy as np
import pandas as pd
import matplotlib as mpl
%matplotlib inline


import tensorflow as tf
from tensorflow import keras

test_dir =r"C:\Users\Hp\Desktop\datasets\test"
validation_dir=r"C:\Users\Hp\Desktop\datasets\validation"
train_dir=r"C:\Users\Hp\Desktop\datasets\train"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)                            

rain_generator = train_datagen.flow_from_directory(train_dir,target_size=(90,90),batch_size = 5,class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(90,90),batch_size = 5,class_mode='binary')


from tensorflow.keras import layers
from tensorflow.keras import models


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(90,90,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))                        
                       
model.summary()

from tensorflow.keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

history = model.fit_generator(
        train_generator,
        steps_per_epoch = 5,
        epochs=20,
        validation_data = validation_generator,validation_steps=5)



                        

      