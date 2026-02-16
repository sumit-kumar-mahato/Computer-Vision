"""
FEATURE DETECTORS: Harris Corner Detector & FAST Detector
==========================================================

Educational implementation with detailed line-by-line comments for learning.
Demonstrates both classic algorithms from scratch using NumPy.

Author: Computer Vision Learning Module
Date: 2026
Purpose: Educational understanding of corner detection algorithms
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter, maximum_filter
import time


# =============================================================================
#                      HARRIS CORNER DETECTOR
#                    (Classic Implementation from Scratch)
# =============================================================================

class HarrisDetectorEducational:
    """
    Harris Corner Detection algorithm implemented from scratch.

    Mathematical Foundation:
    ========================
    The Harris detector analyzes the local auto-correlation function to find corners.

    1. Compute gradients: Ix, Iy using Sobel operator
    2. Form structure tensor M at each pixel:
       M = [Σ(Ix²)    Σ(Ix·Iy)]
           [Σ(Ix·Iy)  Σ(Iy²)  ]
    3. Compute corner response: R = det(M) - k·trace(M)²
    4. Find local maxima and threshold

    Interpretation:
    ===============
    - R >> 0:  Corner (large eigenvalues)
    - R ~ 0:   Edge or flat region
    - R << 0:  Edge (negative eigenvalue)
    """

    def __init__(self, block_size=3, k=0.04, sigma=1.0):
        """
        Initialize Harris detector parameters.

        Args:
            block_size (int): Window size for structure tensor computation (3, 5, 7, ...)
            k (float): Harris constant, typically 0.04-0.06
                      Controls sensitivity to cornerness vs edgeness
            sigma (float): Gaussian smoothing parameter for noise reduction
        """
        self.block_size = block_size
        self.k = k
        self.sigma = sigma

    def detect(self, gray_img, threshold_ratio=0.01, nms_radius=1):
        """
        Detect corners in grayscale image.

        Args:
            gray_img (np.ndarray): Input grayscale image (float32 recommended)
            threshold_ratio (float): Corners = R > threshold_ratio × max(R)
            nms_radius (int): Non-maximum suppression radius (1, 2, 3, ...)

        Returns:
            np.ndarray: Corners as (N, 2) array of [y, x] coordinates
        """

        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Compute Gradients Using Sobel Operator
        # ─────────────────────────────────────────────────────────────────
        # The Sobel operator computes approximate derivatives
        # Ix = ∂I/∂x (horizontal changes) → detects vertical edges
        # Iy = ∂I/∂y (vertical changes)   → detects horizontal edges

        Ix = sobel(gray_img, axis=1, mode='constant')  # Horizontal gradient
        Iy = sobel(gray_img, axis=0, mode='constant')  # Vertical gradient

        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Compute Products of Gradients (Components of M)
        # ─────────────────────────────────────────────────────────────────
        # These will be summed over neighborhoods to create structure tensor
        # Ixx = (∂I/∂x)²  : Square of horizontal gradient
        # Iyy = (∂I/∂y)²  : Square of vertical gradient
        # Ixy = (∂I/∂x)·(∂I/∂y) : Cross product of gradients

        Ixx = Ix * Ix  # Element-wise squaring
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Gaussian Smoothing
        # ─────────────────────────────────────────────────────────────────
        # Reduces noise and creates smoother response map
        # σ controls the smoothing strength
        #   σ small: Noisy response, responsive to texture
        #   σ large: Smooth response, less responsive

        Ixx_smooth = gaussian_filter(Ixx, sigma=self.sigma)
        Iyy_smooth = gaussian_filter(Iyy, sigma=self.sigma)
        Ixy_smooth = gaussian_filter(Ixy, sigma=self.sigma)

        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Compute Harris Response
        # ─────────────────────────────────────────────────────────────────
        # At each pixel, construct structure tensor M and compute R
        # R = det(M) - k·trace(M)²
        #   = (λ₁·λ₂) - k·(λ₁ + λ₂)²
        # where λ₁, λ₂ are eigenvalues of M
        #
        # Eigenvalue interpretation:
        #   λ₁, λ₂ both large → R > 0, R is large → CORNER
        #   One large, one small → R < 0 → EDGE
        #   Both small → R ~ 0 → FLAT REGION

        det_M = Ixx_smooth * Iyy_smooth - Ixy_smooth * Ixy_smooth  # λ₁·λ₂
        trace_M = Ixx_smooth + Iyy_smooth                           # λ₁ + λ₂
        response = det_M - self.k * (trace_M ** 2)                  # Harris measure

        # ─────────────────────────────────────────────────────────────────
        # STEP 5: Thresholding
        # ─────────────────────────────────────────────────────────────────
        # Keep only pixels with strong corner response
        # Using relative threshold: threshold = ratio × max(response)
        # This adapts automatically to image brightness

        max_response = np.max(response)
        threshold = threshold_ratio * max_response
        corner_mask = response > threshold

        # ─────────────────────────────────────────────────────────────────
        # STEP 6: Non-Maximum Suppression (NMS)
        # ─────────────────────────────────────────────────────────────────
        # Remove duplicate corners that are close together
        # Keep only local maxima within a radius
        #
        # Why? Harris response peaks at true corner locations, but due to
        # smoothing, we might get multiple high responses in the same region

        if nms_radius > 0:
            # Create circular footprint for maximum filtering
            size = 2 * nms_radius + 1
            footprint = np.ones((size, size), dtype=bool)

            # Find local maxima
            local_max = maximum_filter(response, footprint=footprint) == response
            nms_mask = corner_mask & local_max
        else:
            nms_mask = corner_mask

        # ─────────────────────────────────────────────────────────────────
        # STEP 7: Extract Corner Coordinates
        # ─────────────────────────────────────────────────────────────────
        # Convert boolean mask to list of corner positions

        corners_y, corners_x = np.where(nms_mask)
        corners = np.column_stack([corners_y, corners_x])

        # ─────────────────────────────────────────────────────────────────
        # STEP 8: Sort by Response Strength (Strongest First)
        # ─────────────────────────────────────────────────────────────────
        # Higher response = stronger, more reliable corner

        if len(corners) > 0:
            response_values = response[corners_y, corners_x]
            sorted_indices = np.argsort(-response_values)  # Descending order
            corners = corners[sorted_indices]

        return corners, response


# =============================================================================
#                           FAST DETECTOR
#                     (Classic Implementation from Scratch)
# =============================================================================

class FASTDetectorEducational:
    """
    FAST (Features from Accelerated Segment Test) Corner Detector.

    Why "FAST"?
    ===========
    1. Segment Test: Only compares pixel intensities (no gradients)
    2. Accelerated: Uses decision tree to quickly reject non-corners
    3. Fast implementation: Minimal computation per pixel

    Algorithm Concept:
    ==================
    For each pixel p:
    1. Draw circle of 16 pixels at radius 3 around p
    2. Select a threshold t (e.g., t=25)
    3. If N (typically 9-12) consecutive pixels are:
       - ALL brighter than p+t, OR
       - ALL darker than p-t
       Then p is a CORNER

    Advantages:
    - Much faster than Harris (no gradients!)
    - Works well for real-time processing
    - Binary decision makes it very efficient

    Disadvantages:
    - Less robust to noise than Harris
    - No corner strength/quality measure (binary)
    - Threshold tuning required
    """

    def __init__(self, threshold=30, n_consecutive=12):
        """
        Initialize FAST detector parameters.

        Args:
            threshold (int): Intensity difference to consider pixel brighter/darker
                           Higher = fewer corners, Lower = more corners
            n_consecutive (int): Min consecutive pixels in circle (9, 12, or 16)
                               9: Faster, less selective
                               12: Balanced
                               16: Most selective
        """
        self.threshold = threshold
        self.n_consecutive = n_consecutive

        # Define Bresenham circle (radius=3, 16 pixels)
        # These are the pixel offsets relative to center
        self.circle = np.array([
            (0, 3),   (1, 3),   (2, 2),   (3, 1),
            (3, 0),   (3, -1),  (2, -2),  (1, -3),
            (0, -3),  (-1, -3), (-2, -2), (-3, -1),
            (-3, 0),  (-3, 1),  (-2, 2),  (-1, 3)
        ])

    def detect(self, gray_img):
        """
        Detect corners using FAST algorithm.

        Args:
            gray_img (np.ndarray): Input grayscale image (uint8 or float)

        Returns:
            np.ndarray: Corners as (N, 2) array of [y, x] coordinates
        """

        if gray_img.dtype != np.uint8:
            gray_img = (gray_img * 255).astype(np.uint8)

        height, width = gray_img.shape
        corners = []

        # ─────────────────────────────────────────────────────────────────
        # Iterate over all interior pixels (radius=3, so skip 3 pixels)
        # ─────────────────────────────────────────────────────────────────

        for y in range(3, height - 3):
            for x in range(3, width - 3):
                # Get center pixel value
                center = int(gray_img[y, x])
                threshold_high = center + self.threshold  # Brighter threshold
                threshold_low = center - self.threshold   # Darker threshold

                # ───────────────────────────────────────────────────────
                # Test corner criterion: N consecutive pixels brighter/darker
                # ───────────────────────────────────────────────────────

                is_corner = False

                # Sample circle pixels
                circle_values = []
                for dx, dy in self.circle:
                    pixel = int(gray_img[y + dy, x + dx])
                    circle_values.append(pixel)

                # Extend to check wraparound (circular list)
                circle_values = circle_values + circle_values[:self.n_consecutive]

                # Check for N consecutive brighter pixels
                for i in range(len(self.circle)):
                    if all(p > threshold_high for p in circle_values[i:i+self.n_consecutive]):
                        is_corner = True
                        break

                # Check for N consecutive darker pixels
                if not is_corner:
                    for i in range(len(self.circle)):
                        if all(p < threshold_low for p in circle_values[i:i+self.n_consecutive]):
                            is_corner = True
                            break

                if is_corner:
                    corners.append([y, x])

        return np.array(corners) if corners else np.array([]).reshape(0, 2)


# =============================================================================
#                        VISUALIZATION UTILITIES
# =============================================================================

def visualize_comparison(image_path, harris_corners, fast_corners, original_img):
    """
    Create side-by-side comparison of Harris and FAST detections.

    Args:
        image_path (str): Path to image file
        harris_corners (np.ndarray): Harris corner coordinates
        fast_corners (np.ndarray): FAST corner coordinates
        original_img (np.ndarray): Original BGR image
    """

    # Create copies for visualization
    img_harris = original_img.copy()
    img_fast = original_img.copy()
    img_both = original_img.copy()

    # Draw Harris corners (RED)
    for y, x in harris_corners:
        cv2.circle(img_harris, (int(x), int(y)), 5, (0, 0, 255), 2)  # Red
        cv2.circle(img_both, (int(x), int(y)), 5, (0, 0, 255), 2)

    # Draw FAST corners (GREEN)
    for y, x in fast_corners:
        cv2.circle(img_fast, (int(x), int(y)), 5, (0, 255, 0), 2)   # Green
        cv2.circle(img_both, (int(x), int(y)), 5, (0, 255, 0), 1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original
    axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    # Harris
    axes[0, 1].imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Harris Corners (n={len(harris_corners)})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    # FAST
    axes[1, 0].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'FAST Corners (n={len(fast_corners)})', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    # Both
    axes[1, 1].imshow(cv2.cvtColor(img_both, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Comparison (Harris=Red, FAST=Green)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def print_statistics(harris_corners, fast_corners, exec_time_harris, exec_time_fast):
    """Print detection statistics and timing."""

    print("\n" + "="*70)
    print("CORNER DETECTION RESULTS")
    print("="*70)
    print(f"\n📊 HARRIS DETECTOR:")
    print(f"   Corners found:     {len(harris_corners)}")
    print(f"   Execution time:    {exec_time_harris*1000:.2f} ms")
    print(f"   Throughput:        {512*512/1e6/(exec_time_harris):.1f} MP/s")

    print(f"\n📊 FAST DETECTOR:")
    print(f"   Corners found:     {len(fast_corners)}")
    print(f"   Execution time:    {exec_time_fast*1000:.2f} ms")
    print(f"   Throughput:        {512*512/1e6/(exec_time_fast):.1f} MP/s")

    print(f"\n⚡ PERFORMANCE:")
    speedup = exec_time_harris / exec_time_fast
    print(f"   FAST is {speedup:.1f}× faster than Harris")
    print("="*70)


# =============================================================================
#                             MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Main demonstration of Harris and FAST corner detectors.
    """

    # Try to load image
    # For testing, you can use:
    # - "chessboard.jpg" - Regular pattern with clear corners
    # - "sudoku.jpg" - Checkered pattern
    # - "building.jpg" - Real-world scene

    image_paths = ["chessboard.jpg", "sudoku.jpg", "building.jpg", "road.jpg"]
    img_path = None

    for path in image_paths:
        try:
            original = cv2.imread(path)
            if original is not None:
                img_path = path
                break
        except:
            pass

    if img_path is None:
        print("⚠️  No test images found. Creating synthetic test image...")
        # Create synthetic image with corners
        original = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.rectangle(original, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(original, (170, 50), (270, 150), (200, 200, 200), -1)
        cv2.circle(original, (150, 250), 40, (100, 100, 255), -1)
    else:
        print(f"✓ Loaded image: {img_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)

    print(f"Image shape: {gray.shape}")
    print("\nRunning corner detectors...\n")

    # ─────────────────────────────────────────────────────────────────
    # HARRIS DETECTOR
    # ─────────────────────────────────────────────────────────────────

    harris_detector = HarrisDetectorEducational(block_size=3, k=0.04, sigma=1.0)

    t0 = time.time()
    harris_corners, harris_response = harris_detector.detect(gray, 
                                                             threshold_ratio=0.01, 
                                                             nms_radius=2)
    exec_time_harris = time.time() - t0

    # ─────────────────────────────────────────────────────────────────
    # FAST DETECTOR
    # ─────────────────────────────────────────────────────────────────

    fast_detector = FASTDetectorEducational(threshold=30, n_consecutive=12)

    t0 = time.time()
    fast_corners = fast_detector.detect(gray)
    exec_time_fast = time.time() - t0

    # ─────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────

    print_statistics(harris_corners, fast_corners, exec_time_harris, exec_time_fast)

    # Visualize
    visualize_comparison(img_path, harris_corners, fast_corners, original)


if __name__ == "__main__":
    main()
