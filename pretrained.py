#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:10:42 2018

@author: Simmetria01
"""


import numpy as np
from skimage.io import imread
from skimage.transform import resize

image = imread('laptop.jpeg')
lowres_image = resize(image, (50, 50), mode='reflect')


from keras.applications.resnet50 import preprocess_input, ResNet50

model = ResNet50(weights='imagenet')


import keras.backend as K

K.image_data_format()

image = imread('laptop.jpeg')
image_224 = resize(image, (224, 224), preserve_range=True, mode='reflect')

image_224_batch = np.expand_dims(image_224, axis=0)
x = preprocess_input(image_224_batch.copy())
preds = model.predict(x)



from keras.applications.resnet50 import decode_predictions

decode_predictions(preds, top=5)


print('Predicted image labels:')
class_names, confidences = [], []
for class_id, class_name, confidence in decode_predictions(preds, top=5)[0]:
    print("    {} (synset: {}): {:0.3f}".format(class_name, class_id, confidence))
