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
from core.img import openimage, alignImages


# functions ============================================


# ========================================================

#%%
path = '/Volumes/LaCie_DataStorage/em_alignment_test'
raw_dir = 'raw'
sub_dir = 'KO'
raw_path = os.path.join(path, raw_dir, sub_dir)
print(raw_path)

op_dir = 'output'
oppath = os.path.join(path, op_dir, sub_dir)

DirCheck(oppath)

#%%
filepath = ListFiles(raw_path, '.tif')['fileabslist']
print(filepath)

#%%
cleaned_filelist = SortByFolder(filepath, 'Tile_')
print([item.filename for item in cleaned_filelist])

#%%
'''
# copy files to the new output folder
for item in cleaned_filelist:
    print(item.filename)
    copyfile(item.filepath, os.path.join(oppath, 'image_' + str(item.index).zfill(4) + '.tif'))
'''
#%%
ippath_1 = os.path.join(oppath, 'image_' + str(2).zfill(4) + '.tif')
a = openimage(ippath_1)

#%%
ippath_1 = os.path.join(oppath, 'image_' + str(2).zfill(4) + '.tif')
ippath_2 = os.path.join(oppath, 'image_' + str(3).zfill(4) + '.tif')

im1 = Image.open(ippath_1)
im2 = Image.open(ippath_2)

#%%
imip_1, imip_2, imReg, h = alignImages(im1, im2)

#%%
imip_1_pil = Image.fromarray(imip_1)
imip_1_pil.show(title = 'imip_1')
imip_2_pil = Image.fromarray(imip_2)
imip_2_pil.show(title = 'imip_2')
imReg_pil = Image.fromarray(imReg)
imReg_pil.show(title = 'imip_1_reg')


#%%
im1array = np.array(im1)
MAX_FEATURES = 500
orb = cv2.ORB_create(MAX_FEATURES)
imip = cv2.resize(im1array, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
keypoints, descriptors = orb.detectAndCompute(imip, None)
print("number of keypoints: {}".format(len(keypoints)))

image1_key = cv2.drawKeypoints(imip, keypoints, flags = 0, outImage = None)
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
'''
cv2.imshow('name', image1_key)
cv2.waitKey()
cv2.destroyAllWindows()
'''
image1_key_pil = Image.fromarray(image1_key)
image1_key_pil.show()



