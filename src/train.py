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
def compute_style_cost_layer(a_S, a_G):
    """
        description: calculates the style cost/loss using gram-matrices
        inputs: 
               - a_S: style layer activations of style image 
               - a_G: style layer activations of generated image
    """
    m, n_H, n_W, n_C = a_S.get_shape().as_list() # m = 1 here, only 1 image
    
    # unrolling the 4-d matrices
    aS = tf.transpose(tf.reshape(a_S, shape=(-1, n_C)), perm=[1, 0])
    aG = tf.transpose(tf.reshape(a_G, shape=(-1, n_C)), perm=[1, 0])
    
    # compute gram-matrices
    gS = tf.matmul(aS, tf.transpose(aS))
    gG = tf.matmul(aG, tf.transpose(aG))
    
    stylCost = 1./(4*(n_H*n_W*n_C)**2)*tf.reduce_sum(tf.square(tf.subtract(gS,
                                                                           gG)))
    return stylCost
#*****************************************************************************80

def compute_content_cost_layer(a_C, a_G):
    
    """
        description: calculates the content cost/loss
        inputs: 
              - a_C: content layer activations of content image 
              - a_G: content layer activations of generated image
    """
    # get dimensions of activations
    m, nH, nW, nC = a_G.get_shape().as_list()
    
    aC = tf.reshape(a_C, shape=[m, -1, nC])
    aG = tf.reshape(a_G, shape=[m, -1, nC])
    
    return 1./(4.*nH*nW*nC)* tf.reduce_sum(tf.square(tf.subtract(aC, aG)))

#*****************************************************************************80
def compute_cost(gen_image, 
                 content_layers,
                 content_outputs,
                 content_extractor,
                 style_layers,
                 style_outputs,
                 style_extractor,
                 wt_content = 10.,
                 wt_style = 40.,
                 wt_variation=10):
    """
        description: Computes total cost/loss  
                     = style_cost 
                       + content_cost 
                       + total_variation_cost (makes output images smoother)
    """
    
    # style cost
    gen_style_outputs = style_extractor(gen_image*255)
    ns = len(style_layers)
    coeff = 1./ns # weights for individual style layers
    J_style_cost = 0
    for a_S, a_G in zip(style_outputs, gen_style_outputs):
        J_style_cost += coeff*compute_style_cost_layer(a_S, a_G)
        
    # content cost
    if (len(content_layers)>1):
        nc = len(content_layers)
        coeff = 1./nc # weights for individual content layers
        for a_C, a_G in zip(content_outputs, gen_content_outputs):
            J_content_cost += coeff*compute_content_cost_layer(a_C, a_G)
    else:
        gen_content_outputs = content_extractor(gen_image*255)
        J_content_cost = compute_content_cost_layer(content_outputs, 
                                                    gen_content_outputs) 
    
    J_total = wt_content*J_content_cost + wt_style*J_style_cost 
    + wt_variation*tf.image.total_variation(gen_image)
        
    return J_total
#*****************************************************************************80
def clip_image(image):
  """
      description: clips images between 0. to 1.
  """
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
#*****************************************************************************80

#@tf.function() # helps in performance
def train_step(image, ntf, wt_content =10., wt_style=40., wt_variation =10.):
    """
        description: performs training using GradientTape functionality
        reference: https://www.tensorflow.org/api_docs/python/tf/GradientTape
        inputs: 
              - image: image to be transformed
              - ntf: NeuralStyleTransfer object 
              - wt_content: coefficient for content loss
              - wt_style: coefficient for style loss
              - wt_variation: coefficent for variation loss
    """
    with tf.GradientTape() as tape:
        loss = compute_cost(image, 
                            ntf.content_layers,
                            ntf.content_outputs,
                            ntf.content_extractor,
                            ntf.style_layers,
                            ntf.style_outputs,
                            ntf.style_extractor,
                            wt_content = wt_content,
                            wt_style = wt_style,
                            wt_variation = wt_variation)
        
    grad = tape.gradient(loss, image)
    ntf.optimizer.apply_gradients([(grad, image)])
    image.assign(clip_image(image))
    
#*****************************************************************************80

def train(image, ntf, wt_content = 10., wt_style=40., wt_variation = 10, 
          epochs=1000):
    
    for epoch in range(epochs):
        start_time = time.time()
        train_step(image, ntf, wt_content, wt_style, wt_variation)
        print ('Epoch {:d} completed, time taken (s) = {:.2f}'.
               format(epoch+1, time.time()-start_time))
    return image

#*****************************************************************************80


