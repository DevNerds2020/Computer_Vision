import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import gabor
from skimage import exposure
from skimage.feature import hog

def load_images(folder_path, colorType = cv2.IMREAD_GRAYSCALE):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, colorType)
        
        images.append(img)
    return images

def apply_prewitt(image):
    prewitt_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    return magnitude

def apply_kirsch(image):
    kernel = np.array([
        [-3, -3, 5],
        [-3, 0, 5],
        [-3, -3, 5]
    ])
    kirsch_image = cv2.filter2D(image, cv2.CV_64F, kernel)
    return np.abs(kirsch_image)

def apply_marr_hildreth(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    marr_hildreth = cv2.Laplacian(gaussian, cv2.CV_64F)
    return np.abs(marr_hildreth)

def apply_canny(image):
    return cv2.Canny(image, 100, 200)

def display_results(images, titles, cmap='gray'):
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.xticks([]), plt.yticks([])

    plt.show()

def color_segmentation(image, n_colors):
    image_flat = image.reshape((-1, 3))
    
    print("**cluster with kmeans**")
    kmeans = KMeans(n_clusters=n_colors, init='k-means++', max_iter=50, n_init=10)
    kmeans.fit(image_flat)
    print("**cluster done**")

    segmented_image = kmeans.cluster_centers_.astype(int)[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)  # Convert to uint8

    return segmented_image

def apply_gabor_filter(image, ksize, sigma, theta, lambd, gamma):
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image

def segment_image(image_path, ksize, sigma, theta, lambd, gamma):
    original_image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gabor filter
    filtered_image = apply_gabor_filter(gray_image, ksize, sigma, theta, lambd, gamma)
    
    # Threshold the filtered image
    _, segmented_image = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return original_image, segmented_image

def compute_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    hog_features, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2-Hys', visualize=True)
    return hog_features, hog_image

def sift_stitch(img1, img2, ratio_threshold=0.75, min_match_count=4):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and compute key points and descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Perform matching using BFMatcher between key points of img1 and img2
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Filter matches based on Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < ratio_threshold * n.distance]

    # Check if enough good matches are found for image stitching
    if len(good_matches) >= min_match_count:
        # Obtain corresponding points in img1 and img2 for homography estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Estimate homography between img1 and img2 using RANSAC
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp img1 to match perspective with img2
        result = cv2.warpPerspective(img1, H, (img2.shape[1] + img1.shape[1], img2.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        return result, H
    else:
        return None, None

def stitch_images(images, ratio_threshold=0.75, min_match_count=4):
    # Initialize the stitched result with the first image
    stitched_result = images[0]

    for i in range(1, len(images)):
        # Perform stitching for the current image and the existing stitched result
        current_image = images[i]
        stitched_result, _ = sift_stitch(stitched_result, current_image, ratio_threshold, min_match_count)

        if stitched_result is None:
            print(f"Insufficient matches for stitching image {i+1}.")
            return

    return stitched_result