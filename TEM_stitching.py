#%%
%load_ext autoreload
%autoreload 2

#%%
import os, sys, re
import numpy as np
import pandas as pd
import argparse
import cv2
from PIL import Image, ImageEnhance, ImageOps
import pprint
from shutil import copyfile
from core.fileop import DirCheck, ListFiles, ListFolders, SplitAll, SortByFolder
from core.imgstitching import IdxConverter
from matplotlib import pyplot as plt

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
wd_dir = 'TEM_stitching_test'
ip_dir = 'test_opencv'
im1_path = os.path.join(path, wd_dir, ip_dir, 'img_0011_0034.tif')
im2_path = os.path.join(path, wd_dir, ip_dir, 'img_0011_0035.tif')
print(im1_path)
print(im2_path)

im1_array = cv2.imread(im1_path, -1) - 32768
im2_array = cv2.imread(im2_path, -1) - 32768

im1_array = im1_array / (np.amax(im1_array) - np.amin(im1_array)) * 255
im2_array = im2_array / (np.amax(im1_array) - np.amin(im1_array)) * 255
im1_array = im1_array.astype('uint8')
im2_array = im2_array.astype('uint8')

factor = 1/2
im1_array = cv2.resize(im1_array, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)
im2_array = cv2.resize(im2_array, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)

print(im1_array)
fig = plt.figure(figsize=(10, 10))
plt.imshow(im1_array, cmap='gray', vmin=0, vmax=255)

# cv2.imshow("im1_array", im1_array)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()

#%%
MAX_FEATURES = 500
orb = cv2.ORB_create(nfeatures = MAX_FEATURES, patchSize = 700)
keypoints1, descriptors1 = orb.detectAndCompute(im1_array, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_array, None)

imkp = cv2.drawKeypoints(im1_array, keypoints1, None)
fig = plt.figure(figsize=(30, 30))
plt.imshow(imkp, cmap='gray')

#%%
surf = cv2.xfeatures2d.SURF_create()
keypoints1, descriptors1 = surf.detectAndCompute(im1_array, None)
keypoints2, descriptors2 = surf.detectAndCompute(im2_array, None)

imkp = cv2.drawKeypoints(im1_array, keypoints1, None)
fig = plt.figure(figsize=(30, 30))
plt.imshow(imkp, cmap='gray')