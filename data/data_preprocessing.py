# -*- coding: utf-8 -*-

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random

def perform_basic_preprocessing(data_dir, size = 300):
    labels = os.listdir(data_dir)
    labels.sort()
    print(labels)
    
    for label in labels:
        dir_end = data_dir + '/../preprocessed_chest_xray/' + label
        os.makedirs(dir_end)
    
        print('hier')
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
         
        for img in os.listdir(path):
            try:
                image = cv2.imread(os.path.join(path, img))
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                resized_image = cv2.resize(gray_image, (size, size))
                equalized_image = cv2.equalizeHist(resized_image)
                
                
                
                cv2.imwrite(os.path.join(dir_end, img), equalized_image)
                 
            except Exception as e:
                print(e)

        print(f'[STATUS] images from ', path, ' prepared')
        print(f'[STATUS] ', class_num)
        

def split_dataset(data_dir, classes_dir = ['Normal', 'Pneumonia'],
                  test_ratio = 0.2):

    for cls in classes_dir:
        os.makedirs(data_dir +'/train/' + cls)
        os.makedirs(data_dir +'/test/' + cls)
    
        src = data_dir + '/' + cls
    
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)
        train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                                  [int(len(allFileNames)* (1 - test_ratio))])
    
    
        train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
        test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]
    
        print("***************************")
        print('Total images: ', len(allFileNames))
        print('Training: ', len(train_FileNames))
        print('Testing: ', len(test_FileNames))
        print("***************************")
    
    
    
        for name in train_FileNames:
                shutil.copy(name, data_dir +'/train/' + cls)
    
        for name in test_FileNames:
                shutil.copy(name, data_dir +'/test/' + cls)
        print("Copying Done!")
        
import numpy as np
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage   
import random     
        
        
def train_data_argumentation(train_data_dir):
    
    for img in os.listdir(train_data_dir):
        image = cv2.imread(os.path.join(train_data_dir, img))
        
        random_value = random.random()
        #if random_value < 0.5:
        argumented_image1 = rotate(image, angle=random.randint(-10, 10)) + (random.randint(-20, 20)/255)
        argumented_umage2 = rotate(image, angle=-random.randint(-10, 10)) + (random.randint(-20, 20)/255)
        #else:
        #rotate_plus = rotate(image, angle=random.randint(1, 10)) - (random.randint(1, 20)/255)
        #rotate_minus = rotate(image, angle=-random.randint(1, 10)) + (random.randint(1, 20)/255)
    
    
        #image rotation using skimage.transformation.rotate
        fig = plt.figure(tight_layout='auto', figsize=(10, 7))
        fig.add_subplot(131)
        plt.title('Rotate 5 and random contrast')
        plt.imshow(image)
        fig.add_subplot(132)
        plt.title('Rotate -5 and random contrast')
        plt.imshow(argumented_umage2)
        fig.add_subplot(133)
        plt.title('Rotate -5 and random contrast')
        plt.imshow(argumented_umage2)
        plt.show()
        
        break
   
        
        name_argumented_image1 = 'arg1' + img
        name_argumented_image2 = 'arg2' + img

        cv2.imwrite(os.path.join(train_data_dir, name_argumented_image1), 255*argumented_image1)
        cv2.imwrite(os.path.join(train_data_dir, name_argumented_image2), 255*argumented_umage2)


        
        
data_dir = '../data/preprocessed_chest_xray'

#perform_basic_preprocessing(data_dir)
#split_dataset(data_dir)
train_data_argumentation('../data/preprocessed_chest_xray/train/Normal')

