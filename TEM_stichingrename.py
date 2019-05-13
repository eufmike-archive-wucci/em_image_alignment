#%%
%load_ext autoreload
%autoreload 2

#%%
import os, sys, re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pprint
from shutil import copyfile
from core.fileop import DirCheck, ListFiles, ListFolders, SplitAll, SortByFolder
from core.imgstitching import IdxConverter

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
wd_dir = 'TEM_stitching_test'
ip_dir = 'output'
ippath = os.path.join(path, wd_dir, ip_dir)
print(ippath)

dircheck = []
op_dir = 'ouput_reordered'
oppath = os.path.join(path, wd_dir, op_dir)
dircheck.append(oppath)
DirCheck(dircheck)

#%%
# load par
par_dir = 'par'
par_path = os.path.join(path, wd_dir, par_dir, 'tile_dimension.csv')
DimData = pd.read_csv(par_path)
display(DimData)
DimData['filename'] = DimData['filename'].astype(str)
DimData.dtypes
#%%
for i in ListFolders(ippath):
    ippath_tmp = os.path.join(ippath, i)
    oppath_tmp = os.path.join(oppath, i)
    DirCheck(oppath_tmp)

    dimdata = DimData[DimData['group'] == i]
    display(dimdata)
    
    for j in ListFolders(ippath_tmp):
        ippath_img_tmp = os.path.join(ippath_tmp, j)
        oppath_img_tmp = os.path.join(oppath_tmp, j)
        DirCheck(oppath_img_tmp)
        print(j)
        m = re.search('^img_(0*[1-9][0-9]*)$', j)
        if m:
            found = m.group(1)
            found = found.strip('0')
            print(found)
        filename_list = list(dimdata['filename'])
        idx = filename_list.index(found)
        print(idx)
        
        tile_dim = (dimdata['x'][idx], dimdata['y'][idx])
        box_dim = (3, 3)
        
        filelist, fileabslist = ListFiles(ippath_img_tmp, '.tif') 
        print(filelist)

        for idx in range(len(filelist)):
            ip_filename = j + '_' + str(idx).zfill(4) + '.tif'
            op_filename = j + '_' + str(IdxConverter(idx, tile_dim, box_dim)).zfill(4) + '.tif'
            print(ip_filename)
            ip_path = os.path.join(ippath, i, j, ip_filename)
            print(op_filename)
            op_path = os.path.join(oppath, i, j, op_filename)
            copyfile(ip_path, op_path)