
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd


# In[17]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D,Dropout
from keras.layers import Flatten
from keras.layers import Dense
from PIL import Image

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(64, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(128, 3, 3, input_shape = (150, 150, 3), activation ='relu'))
classifier.add(Dropout(0.1))
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[18]:


classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 6, activation = 'sigmoid'))

classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])


# In[19]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'E:/JBM/trying/data/train_61326',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(
        'E:/JBM/trying/data/validation_61326',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

classifier.fit_generator(training_set,
                         steps_per_epoch=32,
                         epochs=3,
                         validation_data=validation_generator,
                         validation_steps=10)


# In[15]:


from keras.models import Model
# serialize weights to HDF5
classifier.save_weights("modelConv2d.h5")
classifier.save('model.h5')
print("Saved model to disk")

