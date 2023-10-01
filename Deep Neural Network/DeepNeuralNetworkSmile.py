from PIL import Image
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

directory = 'training_set/'

pixel_intensities = []

# one-hot encoding -> happy (1,0) and sad (0,1)

labels = []

for filename in os.listdir(directory):
    # print(filename)
    image = Image.open(directory + filename).convert('1')
    # print(image)
    # print(image.getdata())
    pixel_intensities.append(list(image.getdata()))
    if filename[0:5] == 'happy':
        labels.append([1, 0])
    elif filename[0:3] == 'sad':
        labels.append([0, 1])

# for pixel_intensity in pixel_intensities:
#     print(pixel_intensity)




