import cv2 as cv
import numpy as np
import os


class VTA:

    def __init__(self):
        self.mask = self.loadMask("data/mask/mask.png")
        self.shape = self.mask.shape[:2]

    def BGR2G(self, img):
        green_channel = img[:, :, 1]
        return green_channel

    def medianBlur(self, img):
        kernel_size = 11  # 5, 7
        median_blur = cv.medianBlur(img, kernel_size)
        return median_blur

    def CLAHE(self, img):
        clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
        clache_img = clahe.apply(img)
        return clache_img

    def adaptiveThreashold(self, img, isGauss: bool):
        method = cv.ADAPTIVE_THRESH_MEAN_C
        if isGauss:
            method = cv.ADAPTIVE_THRESH_GAUSSIAN_C

        theashold_img = cv.adaptiveThreshold(img, 255, method, cv.THRESH_BINARY, 11, 2)
        return theashold_img

    def speckleFilter(self, img):
        speckless_img = cv.filterSpeckles(img, 255, 900, 100)
        return speckless_img[0]

    def morphologicClosing(self, img):
        kernel = np.ones((2, 2), np.uint8)
        closed_img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
        return closed_img

    def removeBorder(self, img, mask):
        masked_img = cv.bitwise_and(img, img, mask=mask)
        extracted_img = cv.bitwise_not(masked_img, masked_img, mask=mask)
        inverted_img = cv.bitwise_not(extracted_img)
        return inverted_img

    def cropImg(self, img, img_name, mask):
        img = cv.resize(img, (470, 470))

        masked_img = cv.bitwise_and(img, img, mask=mask)

        gray_img = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)
        _, alpha = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY)

        masked_img = cv.cvtColor(masked_img, cv.COLOR_BGR2BGRA)
        masked_img[:, :, 3] = alpha

        # cv.imwrite("data/cropped/C_" + img_name, masked_img)
        return masked_img

    def loadImgs(self, dir_path):
        image_names = [f for f in sorted(os.listdir(dir_path))]
        return image_names

    def loadMask(self, mask_path):
        mask = cv.imread(mask_path)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        return mask

    def mosaicingImgs(img1, img2, crop1, crop2):

        # Initialize the SIFT feature detector and extractor
        # sift = cv.SIFT_create()
        sift = cv.ORB_create()

        # Detect keypoints and compute descriptors for both images
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Initialize the feature matcher using FLANN matching
        num_matches = 20

        # SIFT
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)

        # ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1,
        )  # 2

        flann = cv.FlannBasedMatcher(index_params, search_params)

        # Match the descriptors using FLANN matching
        matches_flann = flann.match(descriptors1, descriptors2)

        # Sort the matches by distance (lower is better)
        matches_flann = sorted(matches_flann, key=lambda x: x.distance)

        # Draw the top N matches
        image_matches_flann = cv.drawMatches(
            img1, keypoints1, img2, keypoints2, matches_flann[:num_matches], None
        )

        # Extract the matched keypoints
        src_points = np.float32(
            [keypoints1[m.queryIdx].pt for m in matches_flann]
        ).reshape(-1, 1, 2)
        dst_points = np.float32(
            [keypoints2[m.trainIdx].pt for m in matches_flann]
        ).reshape(-1, 1, 2)

        # Estimate the homography matrix using RANSAC
        homography, mask = cv.findHomography(src_points, dst_points, cv.RANSAC, 5.0)

        # # Print the estimated homography matrix
        # print("Estimated Homography Matrix:")
        # print(homography)

        # Display the images with matches
        cv.imshow("FLANN Matching", image_matches_flann)

        # Warp the first image using the homography
        # result = cv.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))
        result = cv.warpPerspective(crop1, homography, (crop2.shape[1], crop2.shape[0]))

        # Blending the warped image with the second image using alpha blending
        alpha = 0.5  # blending factor
        # blended_image = cv.addWeighted(result, alpha, img2, 1 - alpha, 0)
        blended_image = cv.addWeighted(result, alpha, crop2, 1 - alpha, 0)

        # Display the blended image
        cv.imshow("Blended Image", blended_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        cv.imwrite("data/mosaic.png", blended_image)

    def processImg(self, img, img_name, imShow: bool):

        init_shape = img.shape[:2]

        img = cv.resize(img, self.shape)
        green_channel_img = self.BGR2G(img)

        mask = np.where(green_channel_img < 110, 0, 1).astype(np.uint8)


        median_blur_img = self.medianBlur(green_channel_img)
        clahe_img = self.CLAHE(median_blur_img)
        threashold_img = self.adaptiveThreashold(clahe_img, isGauss=False)
        speckless_img = self.speckleFilter(threashold_img)
        closed_img = self.morphologicClosing(speckless_img)
        # borderless_img = self.removeBorder(closed_img, self.mask)
        resized_img = cv.resize(closed_img, init_shape)
        inverted_image = 255 - resized_img

        inverted_image *=mask

        if imShow:
            cv.imshow("Green channel", green_channel_img)
            cv.imshow("Median blur", median_blur_img)
            cv.imshow("CLAHE", clahe_img)
            cv.imshow("Adaptive threashold", threashold_img)
            cv.imshow("Speckless Mean", speckless_img)
            cv.imshow("Morphologic closing", closed_img)
            # cv.imshow("Borderless", borderless_img)
            # cv.imshow("Resized", resized_img)
            cv.imshow("Inverted", inverted_image)

            cv.waitKey(0)
            cv.destroyAllWindows()

        # cv.imwrite("data/processed/P_" + img_name, inverted_image)
        return inverted_image

    def segment(self, img):
        img = self.processImg(img, "a", True)
        return img


if __name__ == "__main__":
    vta = VTA()

    img = cv.imread("1.png")
    img2 = vta.segment(img)

    threshold = 100
    img2 = np.where(img > threshold, 1, 0).astype(np.uint8)

    cv.imwrite("2.png", img2)
