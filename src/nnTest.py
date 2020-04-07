# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#download mnist data and split into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#plot the first image in the dataset
plt.imshow(X_train[0])
print(X_train[0].shape)

#reshape data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(y_train[0])

model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

predicitons = []
#Create array of predicted categories
for x in model.predict(X_test[:100]):
    m = max(x)
    mi = [i for i, j in enumerate(x) if j == m]
    predictions.append(mi)

actual = []
for x in y_test[:4]:
    actual.append(np.where(x == 1)[0])

print(np.array_equal(predicted, actual))


