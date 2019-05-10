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
from core.fileop import DirCheck, ListFiles, ListFolders, SplitAll, SortByFolder

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
raw_dir = 'alignment_opencv_01_std_8bit'
sub_dir = 'KO_2_reordered_rename'
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

#