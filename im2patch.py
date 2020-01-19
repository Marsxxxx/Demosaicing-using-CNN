import matplotlib.pyplot as plt
from os.path import join, isfile, isdir, exists
from os import listdir, mkdir, remove
from shutil import copy2
from PIL import Image
import cv2
import numpy as np

PATCH_SIZE = 33


mypath = "dataset"
newpath = "patches_dataset"
origin_patches_path = "origin_patches_dataset"
dirs = [f for f in listdir(mypath) if isdir(join(mypath, f))]

def masks_CFA_Bayer(img):
    row = img.shape[0]
    col = img.shape[1]
    green = np.zeros((row, col), dtype=bool)
    red = np.zeros((row, col), dtype=bool)
    blue = np.zeros((row, col), dtype=bool)
    for i in range(row):
        for j in range(col):
            if (i*row+j)%2==1:
                green[i][j] = True
            elif i%2 == 0:
                red[i][j] = True
            else:
                blue[i][j] = True
    return red, green, blue

def color_split(img):
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    return red, green, blue

def mosaic(RGB):
    RGB = np.array(RGB, dtype=float)
    R, G, B = color_split(RGB)
    R_m, G_m, B_m = masks_CFA_Bayer(RGB)

    CFA = R * R_m + G * G_m + B * B_m
    return CFA

def show_image(img, title):
    plt.title(title)
    plt.imshow(img)
    plt.show()

def create_patches(img, patch_size, step_size):
    array_for_patches = []
    for i in range(0, int(img.shape[0] - patch_size), step_size):
        for j in range(0, int(img.shape[1] - patch_size), step_size):
            curr_patch = img[i:i + patch_size, j:j + patch_size]
            array_for_patches.append(curr_patch)
    return np.array(array_for_patches)

def save_patches(patches, path, dir, f):
    i = 0
    for patch in patches:
        cv2.imwrite(join(path, dir, str(i) + f), patch)
        i += 1


for dir in dirs:
    onlyfiles = [f for f in listdir(join(mypath, dir)) if isfile(join(mypath, dir, f))]
    for f in onlyfiles:
        im = cv2.cvtColor(cv2.imread(join(mypath, dir, f), 1), cv2.COLOR_BGR2RGB)
        #show_image(im, "original image")
        mosaiced_image = mosaic(im)
        #show_image(mosaiced_image, "mosaiced image")
        patches = create_patches(mosaiced_image, PATCH_SIZE, PATCH_SIZE)
        origin_patches = create_patches(im, PATCH_SIZE, PATCH_SIZE)
        if not isdir(newpath):
            mkdir(newpath)
        if not isdir(join(newpath, dir)):
            mkdir(join(newpath, dir))
        save_patches(patches, newpath, dir, f)
        if not isdir(origin_patches_path):
            mkdir(origin_patches_path)
        if not isdir(join(origin_patches_path, dir)):
            mkdir(join(origin_patches_path, dir))
        save_patches(origin_patches, origin_patches_path, dir, f)
