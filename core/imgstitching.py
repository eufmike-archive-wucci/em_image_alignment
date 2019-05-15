#%%
import os, sys, re
import numpy as np
import cv2
from PIL import Image
import pprint

def IdxConverter(idx, tile_dim, box_dim):
    f_x, f_y = tile_dim
    tile_size = f_x * f_y
    box_x, box_y = box_dim
    box_size = box_x * box_y

    # print(tile_size)
    x_com_box = f_x//box_x
    x_rem = f_x%box_x
    if x_rem > 0:
        x_incom_box = 1
    else:
        x_incom_box = 0

    y_com_box = f_y//box_y
    y_rem = f_y%box_y
    if y_rem > 0:
        y_incom_box = 1
    else:
        y_incom_box = 0
    
    com_box = x_com_box * y_com_box 
    vert_incom_box = x_incom_box * y_com_box
    hori_incom_box = y_incom_box * x_com_box
    xy_rem = x_incom_box * y_incom_box
    
    print('x_com_box: {}'.format(x_com_box))
    print('x_incom_box: {}'.format(x_incom_box))
    print('y_com_box: {}'.format(y_com_box))
    print('y_incom_box: {}'.format(y_incom_box))
    print('x_rem: {}'.format(x_rem))
    print('y_rem: {}'.format(y_rem))
    print('Amount of boxes')
    print('com_box: {}'.format(com_box))
    print('vert_incom_box: {}'.format(vert_incom_box))
    print('hori_incom_box: {}'.format(hori_incom_box))
    print('xy_rem: {}'.format(xy_rem))
    
    area_com = x_com_box * y_com_box * box_size
    area_incom_hori = hori_incom_box * y_rem * box_x
    area_incom_vert = vert_incom_box * x_rem * box_y

    if idx > (area_com + area_incom_hori + area_incom_vert - 1):
        # print('area_rem')
        box_local = idx - (area_com + area_incom_hori + area_incom_vert)
        idx_absidx_x = box_local // y_rem + x_com_box * box_x
        idx_absidx_y = box_local % y_rem + y_com_box * box_y

    elif idx > (area_com + area_incom_hori - 1 ):
        # print('area_vert')
        box_idx = idx // (x_rem * box_y)
        idx_in_area_vert =  idx - (area_com + area_incom_hori)
        box_local = idx_in_area_vert % (x_rem * box_y)
        # print(box_local)
        idx_absidx_x = box_local // box_y + x_com_box * box_x
        idx_absidx_y = box_local % box_y + (idx_in_area_vert // (x_rem * box_y)) * box_y

    else:
        # print('area_vert or com')
        box_idx_x = idx // (box_size * y_com_box + y_rem * box_x)
        box_idx_y = idx % (box_size * y_com_box + y_rem * box_x) // box_size
        # print(box_idx_x)
        # print(box_idx_y) 

        if box_idx_y == y_com_box:
            # print('box_idx_y == y_com_box')
            box_local = idx % (box_size * y_com_box + y_rem * box_x) % box_size
            # print(box_local)
            idx_absidx_x = box_local // y_rem + box_idx_x * box_x
            idx_absidx_y = box_local % y_rem + box_idx_y * box_y
        else:
            # print('box_idx_y != y_com_box')
            box_local = idx % (box_size * y_com_box + y_rem * box_x) % box_size
            # print(box_local)
            idx_absidx_x = box_local // box_y + box_idx_x * box_x
            idx_absidx_y = box_local % box_y + box_idx_y * box_y
    
    # print('idx_absidx_x: {}'.format(idx_absidx_x))
    # print('idx_absidx_y: {}'.format(idx_absidx_y))
    
    idx_new = idx_absidx_x * f_y + idx_absidx_y
    return idx_new

    
def ScanStitching():