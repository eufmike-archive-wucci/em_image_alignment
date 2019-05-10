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
raw_dir = 'output'
sub_dir = 'KO_2_reordered'
ippath = os.path.join(path, raw_dir, sub_dir)
print(ippath)

dircheck = []
op_dir = 'KO_2_reordered_rename'
oppath = os.path.join(path, raw_dir, op_dir)
dircheck.append(oppath)
DirCheck(dircheck)

#%%
filename, filepath = ListFiles(ippath, '.tif')
#print(filepath)

newname = []
newpath = []
for i in range(len(filename)):
    ind = str(i + 1)
    newname_tmp = 'image_' + ind.zfill(4) + '.tif'
    newpath_tmp = os.path.join(oppath, newname_tmp)
    newname.append(newname_tmp)
    newpath.append(newpath_tmp)

print(newname)
print(newpath)

#%%
for i in range(len(filename)):
    print(i)
    copyfile(filepath[i], newpath[i])