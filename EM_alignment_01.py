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

path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
raw_dir = 'output'
sub_dir = 'KO_2_reordered_rename'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

dircheck = []
op_dir = 'alignment_opencv_01'
opmatch_dir = 'alignment_match_opencv_01'
oppath = os.path.join(path, op_dir, sub_dir)
oppath_match = os.path.join(path, opmatch_dir, sub_dir)
dircheck.append(oppath)
dircheck.append(oppath_match)

DirCheck(dircheck)

#%%
factor = 1/16
center = 50
AlignImagesStack_Affine(ippath, oppath, oppath_match, factor, centerimg = center, ext = '.tif')
