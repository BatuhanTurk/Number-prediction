!nvidia-smi

import keras
import tensorflow
import matplotlib.pyplot as plt
from keras import layers
from keras import models
import numpy as np

tensorflow.__version__

keras.__version__

(train_images , train_labels) , (test_images , test_labels  ) = keras.datasets.mnist.load_data()

temp_digit = train_images[8]
plt.imshow(temp_digit,cmap = plt.cm.binary)
plt.show()

train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)
train_images = np.expand_dims(train_images, -1)
original_testimages = test_images.copy()
test_images = np.expand_dims(test_images, -1)
train_images[0].shape

modela = models.Sequential()

modela.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)))
modela.add(layers.MaxPooling2D((2,2)))
modela.add(layers.Conv2D(128, (3,3), activation="relu"))
modela.add(layers.MaxPooling2D((2,2)))
modela.add(layers.Conv2D(128, (3,3), activation="relu"))
modela.add(layers.Flatten())
modela.add(layers.Dense(64, activation="relu"))
modela.add(layers.Dense(10, activation="softmax"))

modela.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

modela.fit(train_images, train_labels, epochs=6, batch_size=256)

images_test0 = original_testimages[5]
plt.imshow(images_test0,cmap = plt.cm.binary)
plt.show()

a = modela.predict_classes(test_images)

temp = 0
images_test0 = original_testimages[temp]
plt.imshow(images_test0,cmap = plt.cm.binary)
plt.show()
a[temp]