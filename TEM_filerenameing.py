#%%


#####Halt




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

#%%
path = '/Volumes/LaCie_DataStorage/Asensio_Lab'
project_dir = 'TEM_stitching'
ip_dir = 'data'
op_dir = 'data_renamed'
ippath = os.path.join(path, project_dir, ip_dir)
print(ippath)
oppath = os.path.join(path, project_dir, op_dir)
print(oppath)

#%%
foldernames = ListFolders(ippath)

dirchecklist = []
for i in foldernames:
    dirchecklist.append(os.path.join(oppath, i))
DirCheck(dirchecklist)

#%% 
par_path = os.path.join(path, project_dir, 'par', 'tile_dimension.csv')
par_data= pd.read_csv(par_path)
display(par_data)

#%%
filelist, fileabslist = ListFiles(ippath, extension = '.mrc')
# print(filelist)
org_filename = par_data['cellname'] 

class imgfile:
    def __init__(self, filename, filepath, foldername, folderpath):
            self.filename = filename
            self.filepath = filepath
            self.foldername = foldername
            self.folderpath = folderpath

filelistprofile = []
for idx in range(len(fileabslist)):
    path_tmp = fileabslist[idx]
    filename_tmp = filelist[idx]
    folderpath_tmp = os.path.split(path_tmp)[0]
    foldername_tmp = os.path.split(folderpath_tmp)[1]
    filelistprofile.append(imgfile(filename_tmp, path_tmp, foldername_tmp, folderpath_tmp))

for item in 
print([item.filename for item in filelistprofile])