import os, sys, re
import numpy as np
import cv2
from PIL import Image
from shutil import copyfile
from core.fileop import DirCheck, ListFiles, SplitAll, SortByFolder
from scipy.ndimage import gaussian_filter

def openimage(path):
    print(path)
    im = Image.open(path)
    #im.show()
    print(type(im))
    imarray = np.array(im)
    print(imarray)

def AlignImages(im1_path, im2_path, factor, ROI, MAX_FEATURES = 1000, GOOD_MATCH_PERCENT = 0.1):

    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    im1array = np.array(im1)
    im2array = np.array(im2)
    # im1array = gaussian_filter(im1array, sigma = 2)
    im1array_roi = im1array
    # im2array = gaussian_filter(im2array, sigma = 2)
    im2array_roi = im2array

    if ROI is not None:
        print(ROI)
        x = ROI[0]
        y = ROI[1]
        h = ROI[2]
        w = ROI[3]
        ROI_fitler = np.zeros(im1array.shape)
        ROI_fitler[y:y+h, x:x+w] = 1
        print(ROI_fitler)
        im1array_roi = im1array_roi * ROI_fitler.astype(np.uint8)
        im2array_roi = im2array_roi * ROI_fitler.astype(np.uint8)
        print(im1array_roi.shape)
        print(im2array_roi.shape)
    
    #im1array_pil = Image.fromarray(im1array)
    #im1array_pil.show()

    # Detect ORB features and compute descriptors.
    imip_1 = cv2.resize(im1array_roi, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)
    imip_2 = cv2.resize(im2array_roi, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)
    
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(imip_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imip_2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    good = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(imip_1, keypoints1, imip_2, keypoints2, good, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    # correct the homography matrix with scale factor
    new_h = h
    new_h[0, 2] = h[0, 2]/factor 
    new_h[1, 2] = h[1, 2]/factor
    new_h[2, 0] = h[2, 0]*factor
    new_h[2, 1] = h[2, 1]*factor
    # print(new_h)
    
    # Use homography
    height, width = im1array.shape
    im2Reg = cv2.warpPerspective(im2array, new_h, (width, height))

    return im1array, im2array, im2Reg, new_h, imMatches

def AlignImages_Affine(im1_path, im2_path, factor, MAX_FEATURES = 500, GOOD_MATCH_PERCENT = 0.1):

    im1 = Image.open(im1_path)
    im2 = Image.open(im2_path)

    im1array = np.array(im1)
    im2array = np.array(im2)


    # Detect ORB features and compute descriptors.
    imip_1 = cv2.resize(im1array, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)
    imip_2 = cv2.resize(im2array, dsize = None, fx = factor, fy = factor, interpolation=cv2.INTER_CUBIC)
    
    orb = cv2.ORB_create(nfeatures = MAX_FEATURES, patchSize = 100)
    keypoints1, descriptors1 = orb.detectAndCompute(imip_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imip_2, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    good = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(imip_1, keypoints1, imip_2, keypoints2, good, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask= cv2.estimateAffinePartial2D(points2, points1, cv2.RANSAC)
    
    # correct the homography matrix with scale factor
    new_h = h
    new_h[0, 2] = h[0, 2]/factor 
    new_h[1, 2] = h[1, 2]/factor
    print(new_h)
    
    # Use homography
    height, width = im1array.shape
    # im2Reg = cv2.warpPerspective(im2array, new_h, (width, height))
    im2Reg = cv2.warpAffine(im2array, new_h, (width, height))

    return im1array, im2array, im2Reg, new_h, imMatches

def AlignImagesStack(ippath, oppath, oppath_match = None, factor = 1, ROI = None, warp_mode = cv2.MOTION_HOMOGRAPHY, 
                            centerimg = None, startimg = None, endimg = None, ext = '.tif', GOOD_MATCH_PERCENT = 0.1):
    
    # configure parameters

    filename, filepath = ListFiles(ippath, ext)
    # print(filepath)

    if startimg is None:
        startimg_idx = 0   
    else:
        startimg_idx = startimg - 1
    
    print("The First Image of the Stack: Slice No.{}(idx {})".format(startimg, startimg_idx))

    if endimg is None: 
        endimg_idx = len(filename)
    else:
        endimg_idx = endimg - 1

    print("The Last Image of the Stack: Slice No.{}(idx {})".format(endimg, endimg_idx))

    if centerimg is None:
        center_idx = startimg_idx
    else: 
        center_idx = centerimg - 1
    
    print("Image registration starting from: Slice No.{}(idx {})".format(centerimg, center_idx))

    copyfile(filepath[center_idx], os.path.join(oppath, filename[center_idx] + ext))

    # center to the end
    for i in range(center_idx, endimg_idx):
        print(i)
        
        if filename[i] == filename[-1]:
            break
        
        im1_path = os.path.join(oppath, filename[i] + ext)
        im2_path = filepath[i+1]
        opfilename = filename[i+1]

        imip_1, imip_2, im2Reg, h, imMatches = AlignImages(im1_path, im2_path, factor = factor, ROI = ROI, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)

        im2Reg_pil = Image.fromarray(im2Reg)
        im2Reg_pil.save(os.path.join(oppath, opfilename + ext))
        
        if oppath_match is not None: 
            cv2.imwrite(os.path.join(oppath_match, opfilename + '.png'), imMatches)
        
    # ceter to the start
    for i in reversed(range(startimg_idx + 1, center_idx+1)):
        print(i)
        
        im1_path = os.path.join(oppath, filename[i] + ext)
        im2_path = filepath[i-1]
        opfilename = filename[i-1]

        imip_1, imip_2, im2Reg, h, imMatches = AlignImages(im1_path, im2_path, factor = factor, ROI = ROI, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)
    
        im2Reg_pil = Image.fromarray(im2Reg)
        im2Reg_pil.save(os.path.join(oppath, opfilename + ext))
        
        if oppath_match is not None: 
            cv2.imwrite(os.path.join(oppath_match, opfilename + '.png'), imMatches)
        

def AlignImagesStack_Affine(ippath, oppath, oppath_match = None, factor = 1, warp_mode = cv2.MOTION_HOMOGRAPHY, 
                            centerimg = None, startimg = None, endimg = None, ext = '.tif', GOOD_MATCH_PERCENT = 0.1):
    
    # configure parameters

    filename, filepath = ListFiles(ippath, ext)
    # print(filepath)

    if startimg is None:
        startimg_idx = 0   
    else:
        startimg_idx = startimg - 1
    
    print("The First Image of the Stack: Slice No.{}(idx {})".format(startimg, startimg_idx))

    if endimg is None: 
        endimg_idx = len(filename)
    else:
        endimg_idx = endimg - 1

    print("The Last Image of the Stack: Slice No.{}(idx {})".format(endimg, endimg_idx))

    if centerimg is None:
        center_idx = startimg_idx
    else: 
        center_idx = centerimg - 1
    
    print("Image registration starting from: Slice No.{}(idx {})".format(centerimg, center_idx))

    copyfile(filepath[center_idx], os.path.join(oppath, filename[center_idx] + ext))

    # center to the end
    for i in range(center_idx, endimg_idx):
        print(i)
        
        if filename[i] == filename[-1]:
            break
        
        im1_path = os.path.join(oppath, filename[i] + ext)
        im2_path = filepath[i+1]
        opfilename = filename[i+1]

        imip_1, imip_2, im2Reg, h, imMatches = AlignImages_Affine(im1_path, im2_path, factor = factor, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)

        im2Reg_pil = Image.fromarray(im2Reg)
        im2Reg_pil.save(os.path.join(oppath, opfilename + ext))
        
        if oppath_match is not None: 
            cv2.imwrite(os.path.join(oppath_match, opfilename + '.png'), imMatches)

    # ceter to the start
    for i in reversed(range(startimg_idx + 1, center_idx+1)):
        print(i)
        
        im1_path = os.path.join(oppath, filename[i] + ext)
        im2_path = filepath[i-1]
        opfilename = filename[i-1]

        imip_1, imip_2, im2Reg, h, imMatches = AlignImages_Affine(im1_path, im2_path, factor = factor, GOOD_MATCH_PERCENT = GOOD_MATCH_PERCENT)
    
        im2Reg_pil = Image.fromarray(im2Reg)
        im2Reg_pil.save(os.path.join(oppath, opfilename + ext))
        
        if oppath_match is not None: 
            cv2.imwrite(os.path.join(oppath_match, opfilename + '.png'), imMatches)

import operator
from functools import reduce

def Equalize(im):
    h = im.convert("L").histogram()
    lut = []
    for b in range(0, len(h), 256):
        # step size
        step = reduce(operator.add, h[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n = n + h[i+b]
    # map image through lookup table
    return im.point(lut)

def Standardize(im, cropROI = None):
    imarray = np.array(im)
    
    if cropROI is None:
        x = 0
        y = 0
        w = imarray.shape[1]
        h = imarray.shape[0]
    else:
        x = cropROI[0]
        y = cropROI[1]
        w = cropROI[2]
        h = cropROI[3]
    
    imarray_crop = imarray[y:y+h, x:x+w]
    
    mean = np.mean(imarray_crop)
    std = np.std(imarray_crop)
    imarray_std = (imarray - mean) / std

    # imarray_std = imarray_std.astype(np.uint8)

    
    return imarray_std
