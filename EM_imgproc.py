#%%
%load_ext autoreload
%autoreload 2

#%%
import os, sys, re
import numpy as np
import cv2
from PIL import Image
import pprint
from shutil import copyfile
from core.fileop import DirCheck, ListFiles, SplitAll, SortByFolder
from core.img import openimage, AlignImages, AlignImagesStack, Equalize, Standardize


# functions ============================================


# ========================================================

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
raw_dir = 'alignment_opencv_01'
sub_dir = 'KO_2_reordered_rename'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

op_dir = 'alignment_opencv_01_std'
oppath = os.path.join(path, op_dir, sub_dir)

DirCheck(oppath)

#%%
filename, filepath = ListFiles(ippath, '.tif')
print(filepath)

#%%
inten_min = 0.0
inten_max = 0.0
for i in range(len(filename)):
    ipfilepath = os.path.join(ippath, filename[i] + '.tif')
    opfilepath = os.path.join(oppath, filename[i] + '.tif')
    
    ROI = [4352, 3088, 4096, 4096]
    im = Image.open(ipfilepath)
    im_std = Standardize(im, ROI)
    
    if inten_min > np.amin(im_std):
        inten_min = np.amin(im_std)

    if inten_max < np.amax(im_std):
        inten_max = np.amax(im_std)

    imarray_std_PIL = Image.fromarray(im_std)
    imarray_std_PIL.save(opfilepath)

    im.close()
    imarray_std_PIL.close()
print(inten_min, inten_max)

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
raw_dir = 'alignment_opencv_01_std'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

op_dir = 'alignment_opencv_01_std_8bit'
oppath = os.path.join(path, op_dir, sub_dir)

DirCheck(oppath)

filename, filepath = ListFiles(ippath, '.tif')
print(filepath)

#%%
for i in range(len(filename)):
    print(i)
    ipfilepath = os.path.join(ippath, filename[i] + '.tif')
    opfilepath = os.path.join(oppath, filename[i] + '.tif')
    
    im = Image.open(ipfilepath)
    imarray = np.array(im)
    imarray = (imarray - inten_min) / (inten_max - inten_min) 
    imarray = imarray * 255
    imarray_8bit = imarray.astype(np.uint8)
    imarray_8bit_PIL = Image.fromarray(imarray_8bit)
    imarray_8bit_PIL.save(opfilepath)
    
    im.close()
    imarray_8bit_PIL.close()
    