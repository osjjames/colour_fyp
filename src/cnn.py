# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, BatchNormalization, UpSampling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import subprocess

from cnn_data_gen import train_gen, valid_gen
from config import img_rows, img_cols, kernel, num_classes, num_train_samples, num_valid_samples, batch_size, epochs, patience

l2_reg = l2(1e-3)

checkpoint_models_path = '/src/data/models/'
gcs_models_path = 'gs://osjjames-aiplatform/models/'

# Callbacks
tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping('val_loss', patience=patience)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)


class MyCbk(keras.callbacks.Callback):
    def __init__(self, model):
        keras.callbacks.Callback.__init__(self)
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        filename = 'model.%02d-%.4f.hdf5' % (epoch, logs['val_loss'])
        fmt = checkpoint_models_path + filename
        self.model_to_save.save(fmt) # Save to container filesystem
        subprocess.run(["gsutil", "-m", "cp", "-r", fmt, gcs_models_path + filename]) # Save to Google Cloud Storage

def create():
  input_tensor = Input(shape=(img_rows, img_cols, 3))

  x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(input_tensor)
  x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv1_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(128, (kernel, kernel), activation='relu', padding='same', name='conv2_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
  x = Conv2D(128, (kernel, kernel), activation='relu', padding='valid', name='conv2_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
  x = BatchNormalization()(x)

  x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
  x = Conv2D(256, (kernel, kernel), activation='relu', padding='same', name='conv3_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
  x = BatchNormalization()(x)

  x = UpSampling2D(size=(2, 2))(x)
  x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv8_1', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
  x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv8_2', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
  x = Conv2D(64, (kernel, kernel), activation='relu', padding='same', name='conv8_3', kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
  x = BatchNormalization()(x)

  outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same', name='pred')(x)

  model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
  # rewrite the callback: saving through the original model and not the multi-gpu model.
  model_checkpoint = MyCbk(model)


  sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')

  print(model.summary())

  # Final callbacks
  callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

  # Start Fine-tuning
  model.fit(train_gen(),
              steps_per_epoch=num_train_samples // batch_size,
              validation_data=valid_gen(),
              validation_steps=num_valid_samples // batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks,
              use_multiprocessing=True,
              workers=8,
              shuffle=False
              )