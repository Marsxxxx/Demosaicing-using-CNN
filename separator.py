from skimage import color
from skimage.filters import sobel
from skimage.io import imread, imsave
import matplotlib as plt
from os.path import join, isfile, isdir, exists
from os import listdir, mkdir, remove
from shutil import copy2
from PIL import Image


mypath = "Flickr500"
training_path = "dataset/training"
validation_path = "dataset/validation"
#s = [f for f in list(mypath) if is(join(mypath, f))]


counter = 0
#for  in s:
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in onlyfiles:
    path = training_path
    if counter % 7 == 0:
        path = validation_path
        counter = 0
    if not exists(path):
        mkdir(path)
    if not exists(path):
        mkdir(path)
    copy2(join(mypath, f), path)
    if f.split('.')[-1] != "png":
        im = Image.open(join(path, f))
        im.save(join(path, f) + ".png")
        remove(join(path, f))
    counter += 1


