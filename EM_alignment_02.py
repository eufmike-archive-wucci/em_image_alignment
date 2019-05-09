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
from core.img import openimage, AlignImages, AlignImagesStack, AlignImages_Affine, AlignImagesStack_Affine


# functions ============================================


# ========================================================

#%%
'''
path = '/Volumes/LaCie_DataStorage/em_alignment_test'
raw_dir = 'output'
sub_dir = 'KO'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

op_dir = 'alignment_opencv'
oppath = os.path.join(path, op_dir, sub_dir)

DirCheck(oppath)
'''

path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
raw_dir = 'alignment_opencv_01_std_8bit'
sub_dir = 'rescue'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

dircheck = []
op_dir = 'alignment_opencv_02'
opmatch_dir = 'alignment_match_opencv_02'
oppath = os.path.join(path, op_dir, sub_dir)
oppath_match = os.path.join(path, opmatch_dir, sub_dir)
dircheck.append(oppath)
dircheck.append(oppath_match)

DirCheck(dircheck)

#%%
factor = 1/16
center = 50

AlignImagesStack(ippath, oppath, oppath_match, factor, centerimg = center, ext = '.tif')


#%%
factor = 1/16
center = 61
startimg = 60
endimg = 68
ROI = [5153, 4618, 2048, 2048]
AlignImagesStack(ippath, oppath, oppath_match, factor, ROI = None,
                    centerimg = center, startimg = startimg, endimg = endimg, ext = '.tif', GOOD_MATCH_PERCENT = 0.15)