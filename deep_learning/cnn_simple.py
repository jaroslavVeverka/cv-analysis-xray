# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 13:48:31 2021

@author: jveverka
"""

import numpy as np
import pandas as pd
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers, Sequential
from tensorflow.math import confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory


from data.data_loader import prepare_images

# path to data
data_dir = '../data/chest_xray'
img_size=300

print('[STATUS] Start of cnn')

image_dataset_train = image_dataset_from_directory(directory = data_dir,
                                             color_mode="grayscale",
                                             image_size=(256, 256),
                                             labels="inferred",
                                             label_mode='int',
                                             class_names=['NORMAL', 'PNEUMONIA'],
                                             validation_split= 0.2,
                                             subset='training',
                                             seed=123)

image_dataset_test = image_dataset_from_directory(directory = data_dir,
                                             color_mode="grayscale",
                                             image_size=(256, 256),
                                             labels="inferred",
                                             label_mode='int',
                                             class_names=['NORMAL', 'PNEUMONIA'],
                                             validation_split= 0.2,
                                             subset='validation',
                                             seed=123)

class_names = image_dataset_train.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in image_dataset_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(cv.cvtColor(images[i].numpy().astype("uint8"), cv.COLOR_BGR2RGB))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    
num_classes = 2

model = models.Sequential()
model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 1)))
# feature (maps) extraction
model.add(layers.Conv2D(32, (8, 8), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (8, 8), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (8, 8), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# create 1D vector representation
model.add(layers.Flatten())

# classification 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


epochs=10

history = model.fit(
  image_dataset_train,
  validation_data=image_dataset_test,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

test_eval = model.evaluate(image_dataset_test, verbose=0)

# predictions = np.array([])
# labels =  np.array([])
# for x, y in image_dataset_test:
#   predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
#   labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
#   print(labels)
  

# con_matrix = confusion_matrix(labels=labels, predictions=predictions).numpy()
# print(con_matrix)


print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

    
    
    
    
    