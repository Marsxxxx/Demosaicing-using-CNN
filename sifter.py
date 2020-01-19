import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, exists
from os import listdir, mkdir, remove
from shutil import copy2
from PIL import Image
import cv2
import numpy as np



training_path = "filtered_dataset/training"
origin_training_path = "origin_filtered_dataset/training"
val_path = "filtered_dataset/validation"
origin_val_path = "origin_filtered_dataset/validation"
mypath = "patches_dataset"
origin_patches_path = "origin_patches_dataset"


dirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

training_patches = []
val_patches = []

counter = 0
for dir in dirs:
    onlyfiles = [f for f in listdir(join(mypath, dir)) if isfile(join(mypath, dir, f))]
    for f in onlyfiles:
        counter += 1
        if counter % 100 == 0:
            im = cv2.imread(join(mypath, dir, f))
            original_im = cv2.imread(join(origin_patches_path, dir, f))
            if(dir == 'training'):
                path = training_path
                origin_path = origin_training_path
            else:
                path = val_path
                origin_path = origin_val_path
            cv2.imwrite(join(path, f), im)
            cv2.imwrite(join(origin_path, f), original_im)
            counter = 0



