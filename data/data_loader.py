# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:27:29 2021

@author: jveverka
"""

import os
import cv2

def prepare_images(data_dir):
    labels = os.listdir(data_dir)
    labels.sort()
    print(labels)
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), 0)
               # print(img_arr.shape)
                data.append([class_num, img_arr])
            except Exception as e:
                print(e)
        print(f'[STATUS] images from ', path, ' prepared')
        print(f'[STATUS] ', class_num)
    return data
