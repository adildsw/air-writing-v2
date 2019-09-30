"""
Optical character recognition in air-writing.
Created on Fri Apr 19 17:00:00 2019
Author: Adil Rahman

"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import cv2
import numpy
import tensorflow as tf
from keras import backend

backend.set_image_dim_ordering('tf')

class Recognizer(object):
    
    def __init__(self):
        self._i_shape = (1, 28, 28)
        self._b_shape = (1, 22, 22)
        
        self._min_size = 150
        self._d_kernel = (3, 3)
        
        self._opencv_version = int(cv2.__version__.split('.')[0])
        
#        self._model_mnist = tf.keras.models.load_model('models/lstm_modified_mnist.model')
        self._model_mnist = tf.keras.models.load_model('models/lstm_mnist_air_new.model')
        self._model_binary = tf.keras.models.load_model('models/lstm_modified_noclass_binary.model')
        
        return
    
    def _resize(self, image):
        w = image.shape[1]
        h = image.shape[0]
        dst_w = self._i_shape[1]
        dst_h = self._i_shape[2]
        box_w = self._b_shape[1]
        box_h = self._b_shape[2]
        
        if w >= h:
            new_h = h * box_w // w
            image = cv2.resize(image, (box_w, new_h), interpolation=cv2.INTER_AREA)
            pad_w = (dst_w - box_w) // 2
            pad_h = (dst_h - new_h) // 2
            pad_l = numpy.zeros((new_h, pad_w), dtype='uint8')
            pad_r = numpy.zeros((new_h, pad_w), dtype='uint8')
            pad_t = numpy.zeros((pad_h, dst_w), dtype='uint8')
            pad_b = numpy.zeros((dst_h-new_h-pad_h, dst_w), dtype='uint8')
            image = numpy.hstack((pad_l, image, pad_r))
            image = numpy.vstack((pad_t, image, pad_b))
        else:
            new_w = w * box_h // h
            image = cv2.resize(image, (new_w, box_h), interpolation=cv2.INTER_AREA)
            pad_w = (dst_w - new_w) // 2
            pad_h = (dst_h - box_h) // 2
            pad_l = numpy.zeros((box_h, pad_w), dtype='uint8')
            pad_r = numpy.zeros((box_h, dst_w-new_w-pad_w), dtype='uint8')
            pad_t = numpy.zeros((pad_h, dst_w), dtype='uint8')
            pad_b = numpy.zeros((pad_h, dst_w), dtype='uint8')
            image = numpy.hstack((pad_l, image, pad_r))
            image = numpy.vstack((pad_t, image, pad_b))
        
        return image
    
    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)
        
        contours, heirarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        predprobas = []
        ncp = 0
        
        bn_rects = []
        for cntr in contours:
            bn_rects.append(cv2.boundingRect(cntr))
        bn_rects.sort(key=lambda x: x[0])
        
        for rect in bn_rects:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            
            if h < self._min_size:
                continue
            
            image = image[y:y+h, x:x+w]
            image = self._resize(image)
            image = cv2.dilate(image, self._d_kernel)
            image = image.astype('float64').reshape(1, self._i_shape[2], self._i_shape[1]) / 255
            
            noclassprob = self._model_binary.predict(image)
            ncp = noclassprob[0][1]
            if ncp < 0.05:
                prob = self._model_mnist.predict(image)
                probmax = numpy.round(numpy.max(prob), 4)
                if probmax > 0.97:
                    predprobas.append(prob[0])
                    
        if predprobas:
            predicted_value = numpy.argmax(predprobas[0])
            confidence = numpy.round(numpy.max(predprobas[0]), 4)
            return [predprobas[0]], predicted_value, confidence, ncp
        return [], [], [], []