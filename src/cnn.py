# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, BatchNormalization, UpSampling2D
from tensorflow.keras.regularizers import l2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

l2_reg = l2(1e-3)
img_rows = 64
img_cols = 64
kernel = 3
num_classes = 313

input_tensor = Input(shape=(img_rows, img_cols, 3))

x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(input_tensor)
x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
x = BatchNormalization()(x)

x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv8_3', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
x = BatchNormalization()(x)

outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')(x)

model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
