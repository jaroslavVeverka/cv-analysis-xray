# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:53:01 2021

@author: jveverka
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mahotas
import h5py

bins = 8

#feature-descriptor-0: Color Channel Descriptor
def fd_color_channel_desc(image):
    (means, stds) = cv2.meanStdDev(image)
    feature = np.concatenate([means, stds]).flatten()
    return feature

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0], None, [20], [0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()



def extract_global_features(images):
    labeled_featured_images = []
    print('[STATUS] extracting global featured from', len(images), 'images')
    for i, image in enumerate(images):
        #resized_arr = cv2.resize(image, (img_size, img_size))

        fv_color_channel = fd_color_channel_desc(image)
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)

        labeled_featured_images.append(np.hstack([fv_color_channel, fv_hu_moments, fv_haralick, fv_histogram]))
        

        if (i + 1) % 100 == 0:
            print('[STATUS]', i + 1, 'images processed')

    print('[STATUS] feature extraction of', i + 1, 'images processed')
    return labeled_featured_images