# feature_matching.py
# Educational Python script demonstrating SIFT, SURF, and ORB feature detection,
# description, and matching between two images using OpenCV in Python.
#
# Important notes (as of 2025–2026):
#   • SIFT  → fully available in main OpenCV (patent expired ~2020–2021)
#   • ORB   → always free and patent-free, very fast
#   • SURF  → still patented / non-free in most distributions
#             Usually requires compiling OpenCV with OPENCV_ENABLE_NONFREE=ON
#             and including opencv_contrib modules.
#             → Many pre-built pip packages (opencv-python) do NOT include SURF.
#             → If SURF fails → comment it out or use ORB/SIFT instead for teaching.
#
# Requirements:
#   pip install opencv-python matplotlib numpy

import cv2                          # Main OpenCV library
import numpy as np                  # For array operations
import matplotlib.pyplot as plt     # For nice visualization


def load_and_prepare_image(image_path):
    """
    Load color image and convert to grayscale.
    Many feature detectors work on grayscale images.
    """
    img_color = cv2.imread(image_path)                      # Read image in BGR format
    if img_color is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # Convert BGR → Grayscale
    return img_color, img_gray                               # Return both versions


def detect_and_compute_sift(gray_img):
    """
    Detect keypoints and compute SIFT descriptors.
    SIFT is scale & rotation invariant, 128-float descriptor.
    """
    sift = cv2.SIFT_create()                            # Create SIFT detector object (OpenCV ≥4.4)
    # You can tune parameters: nfeatures, nOctaveLayers, contrastThreshold, etc.
    # sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04)
    
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    # detectAndCompute() does both detection + description in one call
    # None = no mask (use whole image)
    
    return keypoints, descriptors


def detect_and_compute_orb(gray_img, nfeatures=1500):
    """
    Detect keypoints and compute ORB (binary) descriptors.
    ORB is very fast, rotation-aware, binary (Hamming distance matching).
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)           # Limit max number of features
    # Other useful params: scaleFactor=1.2, nlevels=8, edgeThreshold=31, scoreType=cv2.ORB_HARRIS_SCORE
    
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    # Returns: list of KeyPoint objects + binary descriptor array (uint8)
    
    return keypoints, descriptors


def detect_and_compute_surf(gray_img):
    """
    Attempt to use SURF (Speeded-Up Robust Features).
    WARNING: Often raises exception in standard pip installs because patented.
    Only works if OpenCV was built with OPENCV_ENABLE_NONFREE=ON + contrib modules.
    """
    try:
        surf = cv2.xfeatures2d.SURF_create()            # SURF is in xfeatures2d module
        # surf.setExtended(1)                           # Optional: 128-dim instead of 64
        # surf.setHessianThreshold(400)
        
        keypoints, descriptors = surf.detectAndCompute(gray_img, None)
        return keypoints, descriptors
    except AttributeError:
        print("SURF is not available in this OpenCV build (patented / non-free).")
        print("Use SIFT or ORB instead for this demonstration.")
        return [], None
    except cv2.error as e:
        print("SURF error:", e)
        print("Likely need to compile OpenCV with non-free flag.")
        return [], None


def draw_keypoints(img_color, keypoints, title="Keypoints"):
    """
    Draw detected keypoints on the color image for visualization.
    """
    img_with_kp = cv2.drawKeypoints(
        img_color,
        keypoints,
        None,                                   # output image (None = create new)
        color=(0, 255, 0),                      # green circles
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        # shows orientation & scale
    )
    
    # Show using matplotlib (handles RGB conversion)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def match_features_bf(des1, des2, descriptor_type='SIFT', ratio_thresh=0.75):
    """
    Match features using Brute-Force Matcher with ratio test (Lowe's ratio test).
    
    Args:
        des1: Descriptors from image 1
        des2: Descriptors from image 2
        descriptor_type: 'SIFT'/'SURF' (L2 norm) or 'ORB' (Hamming distance)
        ratio_thresh: Threshold for ratio test (default 0.75)
    
    Returns:
        good_matches: List of good DMatch objects after ratio test
    """
    if des1 is None or des2 is None:
        print("One or both descriptor sets are None. Cannot perform matching.")
        return []
    
    # Select distance metric based on descriptor type
    if descriptor_type == 'ORB':
        # ORB uses binary descriptors → Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        # SIFT/SURF use float descriptors → L2 norm (Euclidean distance)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Find k=2 nearest matches for each descriptor (for ratio test)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        # Some keypoints might have only 1 match (edge case)
        if len(match_pair) == 2:
            m, n = match_pair
            # Keep match if best match is significantly better than second-best
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
    
    return good_matches


def draw_matches_visualization(img1_color, kp1, img2_color, kp2, matches, title="Feature Matches"):
    """
    Draw matched keypoints between two images side-by-side.
    
    Args:
        img1_color: First image (BGR)
        kp1: Keypoints from image 1
        img2_color: Second image (BGR)
        kp2: Keypoints from image 2
        matches: List of DMatch objects (good matches)
        title: Plot title
    """
    # Draw matches with random colors for each match
    img_matches = cv2.drawMatches(
        img1_color, kp1,                        # Image 1 and its keypoints
        img2_color, kp2,                        # Image 2 and its keypoints
        matches,                                 # Good matches to draw
        None,                                    # Output image (None = create new)
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        # Don't draw unmatched keypoints
    )
    
    # Display using matplotlib
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({len(matches)} good matches)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # ────────────────────────────────────────────────
    # Change these paths to your test images
    # Good choices: same object/scene from different angles or scales
    # Example: two photos of same building, book cover, logo, etc.
    
    image_path_1 = "Feature_matching/cat.jpg"               # ← Replace with real path (reference image)
    image_path_2 = "Feature_matching/cat.jpg"               # ← Replace with real path (query image)
    
    # # Alternative examples:
    # image_path_1 = "book_cover_1.jpg"
    # image_path_2 = "book_cover_2.jpg"
    
    try:
        color_img1, gray_img1 = load_and_prepare_image(image_path_1)
        color_img2, gray_img2 = load_and_prepare_image(image_path_2)
    except FileNotFoundError as e:
        print(e)
        print("Please update 'image_path_1' and 'image_path_2' variables with valid image files.")
        return
    
    print(f"Image 1 loaded: {gray_img1.shape}")
    print(f"Image 2 loaded: {gray_img2.shape}")
    
    # ────────────── SIFT Feature Detection & Matching ─────────────────────────────
    print("\n" + "="*60)
    print("Computing SIFT features...")
    print("="*60)
    
    kp1_sift, des1_sift = detect_and_compute_sift(gray_img1)
    kp2_sift, des2_sift = detect_and_compute_sift(gray_img2)
    
    print(f"SIFT → Image 1: {len(kp1_sift)} keypoints")
    print(f"SIFT → Image 2: {len(kp2_sift)} keypoints")
    
    if kp1_sift and kp2_sift:
        # Optional: visualize keypoints on individual images
        # draw_keypoints(color_img1, kp1_sift, "SIFT Keypoints - Image 1")
        # draw_keypoints(color_img2, kp2_sift, "SIFT Keypoints - Image 2")
        
        # Perform feature matching with ratio test
        good_matches_sift = match_features_bf(des1_sift, des2_sift, descriptor_type='SIFT', ratio_thresh=0.75)
        print(f"SIFT → {len(good_matches_sift)} good matches after ratio test")
        
        # Visualize matches
        if good_matches_sift:
            draw_matches_visualization(color_img1, kp1_sift, color_img2, kp2_sift, 
                                      good_matches_sift, "SIFT Feature Matches")
    
    # ────────────── ORB Feature Detection & Matching ──────────────────────────────
    print("\n" + "="*60)
    print("Computing ORB features...")
    print("="*60)
    
    kp1_orb, des1_orb = detect_and_compute_orb(gray_img1, nfeatures=2000)
    kp2_orb, des2_orb = detect_and_compute_orb(gray_img2, nfeatures=2000)
    
    print(f"ORB → Image 1: {len(kp1_orb)} keypoints")
    print(f"ORB → Image 2: {len(kp2_orb)} keypoints")
    
    if kp1_orb and kp2_orb:
        # Optional: visualize keypoints on individual images
        # draw_keypoints(color_img1, kp1_orb, "ORB Keypoints - Image 1")
        # draw_keypoints(color_img2, kp2_orb, "ORB Keypoints - Image 2")
        
        # Perform feature matching with ratio test
        # Note: ORB uses Hamming distance (binary descriptors)
        good_matches_orb = match_features_bf(des1_orb, des2_orb, descriptor_type='ORB', ratio_thresh=0.75)
        print(f"ORB → {len(good_matches_orb)} good matches after ratio test")
        
        # Visualize matches
        if good_matches_orb:
            draw_matches_visualization(color_img1, kp1_orb, color_img2, kp2_orb, 
                                      good_matches_orb, "ORB Feature Matches")
    
    # ────────────── SURF Feature Detection & Matching (may fail) ──────────────────
    print("\n" + "="*60)
    print("Trying SURF features (may not be available)...")
    print("="*60)
    
    kp1_surf, des1_surf = detect_and_compute_surf(gray_img1)
    kp2_surf, des2_surf = detect_and_compute_surf(gray_img2)
    
    if kp1_surf and kp2_surf:
        print(f"SURF → Image 1: {len(kp1_surf)} keypoints")
        print(f"SURF → Image 2: {len(kp2_surf)} keypoints")
        
        # Perform feature matching with ratio test
        good_matches_surf = match_features_bf(des1_surf, des2_surf, descriptor_type='SURF', ratio_thresh=0.75)
        print(f"SURF → {len(good_matches_surf)} good matches after ratio test")
        
        # Visualize matches
        if good_matches_surf:
            draw_matches_visualization(color_img1, kp1_surf, color_img2, kp2_surf, 
                                      good_matches_surf, "SURF Feature Matches")
    else:
        print("Skipping SURF matching (not available).")


if __name__ == "__main__":
    main()


# ────────────────────────────────────────────────────────────────
# Quick usage guide for students:
#
# 1. Install dependencies:
#    pip install opencv-python matplotlib numpy
#
# 2. Prepare TWO images of the same object/scene:
#    - Different angles, scales, or lighting
#    - Examples: same book from 2 angles, building from different positions
#    - Update image_path_1 and image_path_2 in the code
#
# 3. Run: python feature_matching.py
#
# 4. Observe:
#    - Number of keypoints each detector finds in both images
#    - Number of good matches after ratio test
#    - Visual connections between matched features
#    - SIFT usually gives more stable matches
#    - ORB is faster but may have more outliers
#    - Ratio test (0.75) filters ambiguous matches
#
# 5. Experiment:
#    - Try different ratio_thresh values (0.7, 0.8, 0.9)
#    - Adjust nfeatures for ORB
#    - Use images with rotation, scale changes, occlusion
#
# Next step: Homography estimation using RANSAC with matched features
# ────────────────────────────────────────────────────────────────
