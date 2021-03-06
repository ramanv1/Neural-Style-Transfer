# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 2019

@author: Vinay Raman, PhD
"""

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

import tensorflow as tf
import time

#*****************************************************************************80        

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#*****************************************************************************80        

def load_img(path_to_img):
  # reference: tensorflow examples

  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

#*****************************************************************************80        

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
#*****************************************************************************80        

def getModelIntermediateOutputs(model, layer_names):
  
  model.trainable = False
  outputs = [model.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([model.input], outputs)
  return model
#*****************************************************************************80        


