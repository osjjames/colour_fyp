# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Flatten, BatchNormalization, UpSampling2D, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage.interpolation as sni

from cnn_data_gen import train_gen, valid_gen, read_lab
from config import img_rows, img_cols, kernel, num_classes, num_train_samples, num_valid_samples, batch_size, epochs, patience, num_classes

l2_reg = l2(1e-3)

checkpoint_models_path = '/src/data/models/'

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
        fmt = checkpoint_models_path + 'model.%02d-%.4f.hdf5'
        self.model_to_save.save(fmt % (epoch, logs['val_loss']))

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


  sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)
  model.compile(optimizer=sgd, loss='categorical_crossentropy')

  print(model.summary())

  return model

def train(model):

  # Final callbacks
  callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

  # Start Fine-tuning
  model.fit_generator(train_gen(),
                          steps_per_epoch=num_train_samples // batch_size,
                          validation_data=valid_gen(),
                          validation_steps=num_valid_samples // batch_size,
                          epochs=epochs,
                          verbose=1,
                          callbacks=callbacks,
                          use_multiprocessing=True,
                          workers=8
                          )

  return model

def predict_lab_from_file(model, image_path):
  orig_image = read_lab(image_path)
  return predict_lab(model, orig_image)
  
def predict_lab(model, orig_image):
  (orig_h, orig_w, channels) = orig_image.shape

  image = cv2.resize(orig_image, (img_rows, img_cols))

  h, w = img_rows // 4, img_cols // 4

  q_ab = np.load('/src/data/pts_in_hull.npy')
  nb_q = q_ab.shape[0]

  image_batch = np.empty((1, img_rows, img_cols, 3), dtype=np.float32)
  image_batch[0] = image

  output_ab = model.predict(image_batch)
  output_ab = output_ab.reshape((h * w, nb_q))

  output_ab = np.exp(np.log(output_ab + 1e-8) / 0.35)
  output_ab = output_ab / np.sum(output_ab, 1)[:, np.newaxis]

  q_a = q_ab[:, 0].reshape((1, num_classes))
  q_b = q_ab[:, 1].reshape((1, num_classes))

  X_a = np.sum(output_ab * q_a, 1).reshape((h, w))
  X_b = np.sum(output_ab * q_b, 1).reshape((h, w))

  X_a = cv2.resize(X_a, (orig_w, orig_h), cv2.INTER_CUBIC)
  X_b = cv2.resize(X_b, (orig_w, orig_h), cv2.INTER_CUBIC)

  X_a += 128
  X_b += 128

  # plt.imshow(X_a)
  # plt.imshow(X_b)

  # X_a = sni.zoom(X_a,(1.*orig_h/img_rows,1.*orig_w/img_cols,1))
  # X_b = sni.zoom(X_b,(1.*orig_h/img_rows,1.*orig_w/img_cols,1))

  out_lab = np.zeros((orig_h, orig_w, 3), dtype=np.int32)
  out_lab[:, :, 0] = orig_image[:, :, 0]
  out_lab[:, :, 1] = X_a
  out_lab[:, :, 2] = X_b

  out_lab = out_lab.astype(np.uint8)
  return out_lab