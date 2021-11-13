# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:59:02 2021

@author: jveverka
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import custom functions for global features extraction
from global_features.feature_extractor import extract_global_features
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
train_extracted_features = extract_global_features(train_images)
test_extracted_features = extract_global_features(test_images)


print('[INFO] train X dim: ', np.array(train_extracted_features).shape)
print('[INFO] test X dim: ', np.array(test_extracted_features).shape)
print('[INFO] train Y dim: ', np.array(train_labels).shape)
print('[INFO] test Y dim: ', np.array(test_labels).shape)


X_train = pd.DataFrame(train_extracted_features)
X_test = pd.DataFrame(test_extracted_features)

train_data = pd.concat([pd.DataFrame(train_labels), X_train], axis=1)
test_data = pd.concat([pd.DataFrame(test_labels), X_test], axis=1)

train_data.to_csv('train_data_full.csv', index=False)
test_data.to_csv('test_data_full.csv', index=False)