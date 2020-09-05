# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 2019

@author: Vinay Raman, PhD
"""
#*****************************************************************************80
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import time
import functools

import tensorflow as tf
from utils import *
import os
#*****************************************************************************80
class NeuralStyleTransfer:
    
    
    def __init__(self, content_filename, style_filename):
        
        """
            description: constructor for creating NeuralStyleTransfer object
            inputs:
                   - content_filename - image file that provides the content
                   - style_filename - image file that provides the style
                     preferrably in jpg format
    
        """

        # data members set to defaults
        self.content_image = load_img(content_filename)
        self.style_image = load_img(style_filename)
        
        self.model_name = 'vgg19' # default model chosen as VGG19
        self.model = tf.keras.applications.VGG19(include_top=False, 
                                                weights='imagenet')
        self.optimizer_name= 'Adam' # optimizer
        self.optimizer_lr = 2e-2 # optimizer learning rate
        self.optimizer = tf.keras.optimizers.Adam(lr = self.optimizer_lr)

        # style layers used to contruct the gram-matrices
        self.style_layers = ['block1_conv1','block2_conv1', 'block3_conv1',
                             'block4_conv1','block5_conv1']
        self.content_layers = ['block5_conv2'] # content layer
        self.extract_image_style_content() # extract style and content outputs

#*****************************************************************************80        
    def set_model(self, model_name):
        """
            description: setting model type for performing style transfer
            inputs: 
                   - model_name : model name
                   options: vgg19, vgg16, resnet50
        """
        if model_name not in ['vgg19',
                              'vgg16']:
            raise Exception("""Model not available currently, 
            choose between 1. vgg19, 2. vgg16""")
        else:
            if (model_name == 'vgg19'):
                self.model_name = model_name
                self.model = tf.keras.applications.VGG19(include_top=False,
                                                         weights='imagenet')
                print ('Using VGG19')
            if model_name == 'vgg16':
                self.model_name = model_name
                self.model = tf.keras.applications.VGG16(include_top=False,
                                                         weights='imagenet')
                print ('Using VGG16')
                
#             if model_name == 'resnet50':
#                 self.model_name = model_name
#                 self.model = tf.keras.applications.Resnet50(include_top=False,
#                                                          weights='imagenet')    
#                 print ('Using ResNet50')
                
        return self
#******************************************************************************80    
    def set_optimizer(self, opt_name, 
                      learning_rate = 2e-2,
                      decay = 0):
        """
            description: choose the optimizer 
            inputs: optimizer name
                    options: 'Adam','SGD','RMSProp'
        """
        if opt_name not in ['Adam',
                            'RMSProp',
                            'SGD']:
            raise Exception("""Optimizer not available currently, 
            choose between 1. Adam, 2. RMSProp and 3. SGD""")
        else:
            
            self.optimizer_name = opt_name
            if opt_name=='Adam':
                self.optimizer = tf.keras.optimizers.Adam(lr = learning_rate,
                                                         decay = decay)
            elif opt_name=='RMSProp':
                self.optimizer = tf.keras.optimizers.RMSprop(lr = learning_rate,
                                                            decay = decay)
            elif opt_name =='SGD':
                self.optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
        
        print ('Using {} optimizer with learning rate {:.4f}'.format(opt_name,
                                                                     learning_rate))
        return self
#******************************************************************************80   

#******************************************************************************80   
    def set_style_layers(self, style_layers):
        """
            description: choose style layers for performing style transfer
            inputs: 
                   -style_layers: list containing names of layers
        """
        model_layers = [layer.name for layer in self.model.layers]
        for styleLayer in style_layers:
            if (styleLayer not in model_layers):
                if self.model_name =='vgg19':
                    raise Exception(""" Incorrect layers chosen for style
                    choose from: 
                    block1_conv1, block1_conv2, 
                    block2_conv1, block2_conv2, 
                    block3_conv1, block3_conv2, block3_conv3,
                    block3_conv4, block4_conv1, block4_conv2,
                    block4_conv3, block4_conv4, block4_pool,
                    block5_conv1, block5_conv2, block5_conv3,
                    block5_conv4""")
                elif self.model_name == 'vgg16':
                    raise Exception ("""Incorrect layers chosen for style
                    choose from: 
                    block1_conv1, block1_conv2, block1_pool,
                    block2_conv1, block2_conv2, block2_pool,
                    block3_conv1, block3_conv2, block3_conv3,
                    block3_pool, block4_conv1, block4_conv2,
                    block4_conv3, block4_pool, block5_conv1,
                    block5_conv2, block5_conv3, block5_pool                   
                    """)
        self.style_layers = style_layers
        return self
#******************************************************************************80    
    def set_content_layers(self, content_layers):
        """
            description: choose content layers
            inputs: 
                   - content_layers: list containing layer names
        """
        model_layers = [layer.name for layer in self.model.layers]
        
        for contentLayer in content_layers:
            if (contentLayer not in model_layers):
                if (self.model_name=='vgg19'):
                    raise Exception(""" Incorrect layers chosen for content 
                    choose from: 
                    block1_conv1, block1_conv2, 
                    block2_conv1, block2_conv2, 
                    block3_conv1, block3_conv2, block3_conv3,
                    block3_conv4, block4_conv1, block4_conv2,
                    block4_conv3, block4_conv4, block4_pool,
                    block5_conv1, block5_conv2, block5_conv3,
                    block5_conv4""")
                elif self.model_name == 'vgg16':
                    raise Exception("""Incorrect layers chosen for content 
                    choose from: 
                    block1_conv1, block1_conv2, block1_pool,
                    block2_conv1, block2_conv2, block2_pool,
                    block3_conv1, block3_conv2, block3_conv3,
                    block3_pool, block4_conv1, block4_conv2,
                    block4_conv3, block4_pool, block5_conv1,
                    block5_conv2, block5_conv3, block5_pool""")
            
        self.content_layers = content_layers
        return self
#*****************************************************************************80    
    def extract_image_style_content(self):
        
        """
            description: this function extracts intermediate model outputs
                         required for calculation of style loss and 
                         content loss
        """
        print ('Extracting image style and content')
        print ("Using {} model".format(self.model_name))
        print ("Style layers: ", self.style_layers)
        print ("Content layers: ", self.content_layers)

        
        #extract style outputs
        self.style_extractor = getModelIntermediateOutputs(self.model, 
                                                      self.style_layers)
        self.style_outputs = self.style_extractor(self.style_image*255)
           
        #extract content outputs
        self.content_extractor = getModelIntermediateOutputs(self.model, 
                                                        self.content_layers)
        self.content_outputs = self.content_extractor(self.content_image*255)
        
        return self
#*****************************************************************************80        

    def plot_images(self, figure_size=(10, 15)):
        """
           description: simple in-built utility to plot the images 
        """
        plt.figure(figsize=figure_size)
        plt.subplot(1, 2, 1)
        plt.imshow(self.content_image[0])
        plt.title('Content Image')

        plt.subplot(1,2,2)
        plt.imshow(self.style_image[0])
        plt.title('Style image')
        plt.tight_layout()
        plt.show()

#*****************************************************************************80        
   
    
