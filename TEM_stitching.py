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
im2_array = im2_array / (np.amax(im2_array) - np.amin(im2_array)) * 255
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
'''
MAX_FEATURES = 500
orb = cv2.ORB_create(nfeatures = MAX_FEATURES, patchSize = 700)
keypoints1, descriptors1 = orb.detectAndCompute(im1_array, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2_array, None)

imkp = cv2.drawKeypoints(im1_array, keypoints1, None)
fig = plt.figure(figsize=(30, 30))
plt.imshow(imkp, cmap='gray')
'''

#%%
surf = cv2.xfeatures2d.SURF_create()
keypoints1, descriptors1 = surf.detectAndCompute(im1_array, None)
keypoints2, descriptors2 = surf.detectAndCompute(im2_array, None)

imkp = cv2.drawKeypoints(im1_array, keypoints1, None)
fig = plt.figure(figsize=(30, 30))
plt.imshow(imkp, cmap='gray')


imkp = cv2.drawKeypoints(im2_array, keypoints2, None)
fig = plt.figure(figsize=(30, 30))
plt.imshow(imkp, cmap='gray')


#%%
print(len(matches))
print(keypoints1[0].pt[0])
print(dir(keypoints1[0].pt))
print(dir(descriptors1[0]))

#%%
# FLANN matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
match = cv2.FlannBasedMatcher(index_params, search_params)
matches = match.knnMatch(descriptors1, descriptors2, k=2)

#%%
good = []
for m, n in matches:
    if m.distance < 0.25 * n.distance:
        good.append(m)
print(len(good))

#%%
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   flags = 2)
img3 = cv2.drawMatches(im1_array, keypoints1, im2_array, keypoints2, good, None, **draw_params)
# cv2.imshow("original_image_drawMatches.jpg", img3)
fig = plt.figure(figsize=(30, 30))
plt.imshow(img3, cmap='gray')

#%%
src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

height, weight = im1_array.shape
pts = np.float32([[0, 0], [0, height-1], 
        [weight-1, height-1], [weight-1, 0]])
print(pts)
pts = pts.reshape(-1,1,2)
print(pts)

#%%
dst = cv2.perspectiveTransform(pts, H)
print(dst)
#%%
# img2 = cv2.polylines(im2_array,[np.int32(dst)],True, 255, 3, cv2.LINE_AA)
fig = plt.figure(figsize=(30, 30))
plt.imshow(img2, cmap='gray')

#%%
#dst = cv2.warpPerspective(im2_array, H, (im1_array.shape[1] + im2_array.shape[1], im1_array.shape[0]))
dst = cv2.warpPerspective(im1_array, H, (im2_array.shape[1], im2_array.shape[0] + im1_array.shape[0]))
dst[0:im2_array.shape[0], 0:im2_array.shape[1]] = im2_array


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:]) 
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])    
    return frame


fig = plt.figure(figsize=(30, 30))
plt.imshow(trim(dst), cmap='gray')