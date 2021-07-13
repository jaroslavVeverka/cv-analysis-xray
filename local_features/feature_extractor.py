# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:12:21 2021

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
from scipy.cluster.vq import kmeans, vq

def extract_local_features(images, img_size=300):
    #brisk = cv2.BRISK_create()
    sift = cv2.SIFT_create(edgeThreshold = 12)
    #surf = cv2.SURF(400)
    labeled_featured_images = []
    print('[STATUS] extracting local featured from', len(images), 'images')
    for i, image in enumerate(images):
        resized_arr = cv2.resize(image, (img_size, img_size))
        
        kpts, des = sift.detectAndCompute(resized_arr, None)
        
        # create picture with detected kpts
        if (i == 0):
            print(len(kpts))
            print(sift.descriptorSize())
            img = cv2.drawKeypoints(resized_arr, kpts, resized_arr,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('brisk_keypoints.jpg',img)


        labeled_featured_images.append(des)

        if (i + 1) % 100 == 0:
            print('[STATUS]', i + 1, 'images processed')

    print('[STATUS] feature extraction of', i + 1, 'images processed')
    return labeled_featured_images


def fit_transform_bovw(data, k = 100):
    # split all arrays into one
    descriptors = np.vstack(data)
    descriptors = descriptors.astype(float)
    
    voc, variance = kmeans(descriptors, k, 1)
    
    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    features = np.zeros((len(data), k), "float32")
    for i in range(len(data)):
        words, distance = vq(data[i],voc)
        for w in words:
            features[i][w] += 1
    
    return features, voc


def transform_bovw(data, fitted_kmeans, k = 100):
    # split all arrays into one
    descriptors = np.vstack(data)
    descriptors = descriptors.astype(float)
    
    voc, variance = kmeans(descriptors, k, 1)
    
    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    features = np.zeros((len(data), k), "float32")
    for i in range(len(data)):
        words, distance = vq(data[i],fitted_kmeans)
        for w in words:
            features[i][w] += 1
        
    return features