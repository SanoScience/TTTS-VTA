import cv2 as cv
import numpy as np
import os

def BGR2G(img):
    green_channel = img[:, :, 1]
    return  green_channel

def medianBlur(img):
    kernel_size = 5
    median_blur = cv.medianBlur(img, kernel_size)
    return median_blur

def CLAHE(img):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clache_img = clahe.apply(img)
    return clache_img

def adaptiveThreashold(img, isGauss):
    method = cv.ADAPTIVE_THRESH_MEAN_C
    if isGauss:
        method = cv.ADAPTIVE_THRESH_GAUSSIAN_C

    theashold_img = cv.adaptiveThreshold(img,255,method, cv.THRESH_BINARY,11,2)
    return theashold_img

def speckleFilter(img):
    speckless_img = cv.filterSpeckles(img, 255, 900, 100)
    return speckless_img[0]

def morphologicClosing(img):
    kernel = np.ones((5,5),np.uint8)
    closed_img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    return closed_img

def removeBorder(img):
    mask = cv.imread('data/mask/mask.png')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

    masked_img = cv.bitwise_and(img, img, mask=mask)
    extracted_img = cv.bitwise_not(masked_img, masked_img, mask=mask)
    inverted_img = cv.bitwise_not(extracted_img)
    return inverted_img

def processImg(img):
    green_channel_img = BGR2G(img)
    cv.imshow('Green channel', green_channel_img)

    median_blur_img = medianBlur(green_channel_img)
    cv.imshow('Median blur', median_blur_img)

    clahe_img = CLAHE(median_blur_img)
    cv.imshow('CLAHE', clahe_img)

    threashold_img = adaptiveThreashold(clahe_img, isGauss = False)
    cv.imshow('Adaptive threashold', threashold_img)

    speckless_img = speckleFilter(threashold_img)
    cv.imshow('Speckless Mean', speckless_img)

    closed_img = morphologicClosing(speckless_img)
    cv.imshow('Morphologic closing', closed_img)

    borderless_img = removeBorder(closed_img)
    cv.imshow('Borderless', borderless_img)

    # cv.imwrite('data/processed/2.png', borderless_img)


def cropImg(img):
    # img = cv.imread(img_path)
    mask = cv.imread('data/mask/mask.png')
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    masked_img = cv.bitwise_and(img, img, mask=mask)

    gray_img = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)
    _,alpha = cv.threshold(gray_img,0,255,cv.THRESH_BINARY)

    masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2BGRA)
    masked_img[:, :, 3] = alpha
    cv.imwrite('data/cropped/1.png', masked_img)


def loadImgs(dir_path):
    image_paths =  [ dir_path + '/' + f  for f  in sorted(os.listdir(dir_path))]
    return image_paths


def mosaicingImg():
    # Load the images
    cimage1 = cv.imread('data/cropped/cropped.png')
    cimage2 = cv.imread('data/cropped/cropped2.png')


    image1 = cv.imread('data/processed/1.png')
    image2 = cv.imread('data/processed/2.png')


    # Initialize the SIFT feature detector and extractor
    sift = cv.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Draw keypoints on the images
    image1_keypoints = cv.drawKeypoints(image1, keypoints1, None)
    image2_keypoints = cv.drawKeypoints(image2, keypoints2, None)

    # Initialize the feature matcher using FLANN matching
    num_matches = 50

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Match the descriptors using FLANN matching
    matches_flann = flann.match(descriptors1, descriptors2)

    # Sort the matches by distance (lower is better)
    matches_flann = sorted(matches_flann, key=lambda x: x.distance)

    # Draw the top N matches
    image_matches_flann = cv.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches], None)


    # Extract the matched keypoints
    src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches_flann]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches_flann]).reshape(-1, 1, 2)

    # Estimate the homography matrix using RANSAC
    homography, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)

    # Print the estimated homography matrix
    print("Estimated Homography Matrix:")
    print(homography)

    # Display the images with matches
    cv.imshow('FLANN Matching', image_matches_flann)

    # Warp the first image using the homography
    # result = cv.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))
    result = cv.warpPerspective(cimage1, homography, (cimage2.shape[1], cimage2.shape[0]))


    # Blending the warped image with the second image using alpha blending
    alpha = 0.5  # blending factor
    # blended_image = cv.addWeighted(result, alpha, image2, 1 - alpha, 0)
    blended_image = cv.addWeighted(result, alpha, cimage2, 1 - alpha, 0)

    # Display the blended image
    cv.imshow('Blended Image', blended_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    img = cv.imread('data/sample/test2.png')
    processImg(img)
