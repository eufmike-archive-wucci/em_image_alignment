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

# copy files to the new output folder
for item in cleaned_filelist:
    print(item.filename)
    copyfile(item.filepath, os.path.join(oppath, 'image_' + str(item.index).zfill(4) + '.tif'))
