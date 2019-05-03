import os, sys, re
import numpy as np
import cv2
from PIL import Image

def openimage(path):
    print(path)
    im = Image.open(path)
    #im.show()
    print(type(im))
    imarray = np.array(im)
    print(imarray)

def alignImages(im1, im2):

    # im1.show()
    # im2.show()

    im1array = np.array(im1)
    im2array = np.array(im2)

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.1
    
    # Detect ORB features and compute descriptors.
    height = 256
    width = 256

    imip_1 = cv2.resize(im1array, dsize=(height, width), interpolation=cv2.INTER_CUBIC)
    imip_2 = cv2.resize(im2array, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

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
    imMatches = cv2.drawMatches(imip_1, keypoints1, imip_2, keypoints2, good, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    
    # Use homography
    height, width = imip_2.shape
    im1Reg = cv2.warpPerspective(imip_1, h, (width, height))
    
    return imip_1, imip_2, im1Reg, h
    '''

     # Use homography
    height, width = im2array.shape
    im1Reg = cv2.warpPerspective(im1array, h, (width, height))
    
    return im1array, im2array, im1Reg, h
    '''
