#%%
%load_ext autoreload
%autoreload 2

#%%
import os, sys, re
import numpy as np
import cv2
from PIL import Image
import pprint


#%%
order = list(range(25))
print(order)
f_x = 5
f_y = 5
box_x = 3
box_y = 3

def idxconverter(idx, f_x = f_x, f_y = f_y, box_x = box_x, box_y = box_y):
    x_com_box = f_x//box_x
    x_rem = f_x%box_x
    if  x_rem > 0:
        x_incom_box = 1
    else:
        x_incom_box = 0
    
    print(x_com_box, x_incom_box, x_rem)
    
    

    '''
    return idx_new
    '''
idx_new = idxconverter(idx, f_x = f_x, f_y = f_y, box_x = box_x, box_y = box_y)