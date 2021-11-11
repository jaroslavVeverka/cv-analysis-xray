# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 13:13:53 2021

@author: jveverka
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import custom functions for global features extraction
from local_features.feature_extractor import extract_local_features, fit_transform_bovw, transform_bovw
from data.data_loader import prepare_images

# path to data
train_data_dir = '../data/preprocessed_chest_xray/train'
test_data_dir = '../data/preprocessed_chest_xray/test'

# get labeled images
train_labeled_images = prepare_images(train_data_dir)
print('[STATUS] data size: ', np.array(train_labeled_images).shape)

test_labeled_images = prepare_images(test_data_dir)
print('[STATUS] data size: ', np.array(test_labeled_images).shape)


train_images = [image[1] for image in train_labeled_images]
test_images = [image[1] for image in test_labeled_images]
# get Y
train_labels = [image[0] for image in train_labeled_images]
test_labels = [image[0] for image in test_labeled_images]


# get extracted features of images: X
train_extracted_features = extract_local_features(train_images)
test_extracted_features = extract_local_features(test_images)

# X_train, X_test, y_train, y_test = train_test_split(np.array(extracted_features),
#                                                     np.array(labels),
#                                                     test_size=0.2, stratify=np.array(labels), random_state=42)

print('[INFO] train X dim: ', np.array(train_extracted_features).shape)
print('[INFO] test X dim: ', np.array(test_extracted_features).shape)
print('[INFO] train Y dim: ', np.array(train_labels).shape)
print('[INFO] test Y dim: ', np.array(test_labels).shape)

k_range = [10, 25, 50, 100]

for k in k_range:
    
    print('[INFO] Start of BOVW for k = ', k)
    X_train_trans_bovw, fitted_kmeans = fit_transform_bovw(train_extracted_features, k = k)
    X_test_trans_bovw = transform_bovw(test_extracted_features, fitted_kmeans, k = k)
    
    print('[INFO] X_train_trans_bovw dim: ', X_train_trans_bovw.shape)
    print('[INFO] X_test_trans_bovw dim: ', X_test_trans_bovw.shape)
    
    X_train_trans_bovw = pd.DataFrame(X_train_trans_bovw)
    X_test_trans_bovw = pd.DataFrame(X_test_trans_bovw)
    
    train_trans_bovw = pd.concat([pd.DataFrame(train_labels), X_train_trans_bovw], axis=1)
    test_trans_bovw = pd.concat([pd.DataFrame(test_labels), X_test_trans_bovw], axis=1)
    
    train_path = './data/train_orb_' + str(k) + '.csv'
    test_path = './data/test_orb_' + str(k) + '.csv'
    
    train_trans_bovw.to_csv(train_path, index=False)
    test_trans_bovw.to_csv(test_path, index=False)
    
