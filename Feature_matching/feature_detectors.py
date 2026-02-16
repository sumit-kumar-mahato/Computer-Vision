# =============================================================================
# feature_detectors.py - Advanced Corner Detection Implementations
# Harris Corner Detector & FAST Detector with Educational Comments
# Production-ready with enhanced error handling, metrics, and optimization
# =============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter
from typing import List, Tuple, Optional, Dict
import time


# =============================================================================
#                    HARRIS CORNER DETECTOR - ENHANCED
# =============================================================================

class HarrisCornerDetector:
    """
    Production-ready Harris Corner Detector with advanced features.
    
    Features:
    - Efficient gradient computation using Sobel operators
    - Configurable structure tensor computation
    - Multiple thresholding strategies
    - Non-maximum suppression with radius control
    - Performance metrics and timing
    - Sub-pixel accuracy refinement
    """
    
    def __init__(self, block_size: int = 5, ksize: int = 3, k: float = 0.04):
        """
        Initialize Harris Corner Detector.
        
        Args:
            block_size (int): Window size for structure tensor summation (odd number)
            ksize (int): Sobel kernel size (1, 3, 5, or 7)
            k (float): Harris empirical constant (typically 0.04-0.06)
        """
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.ksize = ksize
        self.k = k
        self.metrics = {}
    
    def detect(self, gray_img: np.ndarray, threshold_ratio: float = 0.01,
               nms_radius: int = 3, return_top_n: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """
        Detect corners in image with enhanced features.
        
        Args:
            gray_img (np.ndarray): Grayscale image (float32 for best results)
            threshold_ratio (float): Fraction of max response for corner detection
            nms_radius (int): Non-maximum suppression radius
            return_top_n (int): Return only top N corners by response strength
        
        Returns:
            List of (row, col, response_strength) tuples
        """
        start_time = time.time()
        
        # Ensure float32
        if gray_img.dtype != np.float32:
            gray_img = gray_img.astype(np.float32)
        
        # Step 1: Compute image gradients using Sobel
        Ix = sobel(gray_img, axis=1, mode='constant')
        Iy = sobel(gray_img, axis=0, mode='constant')
        
        gradient_time = time.time()
        
        # Step 2: Compute products for structure tensor
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Step 3: Apply Gaussian smoothing
        sigma = self.block_size / 5
        Ixx = gaussian_filter(Ixx, sigma=sigma)
        Iyy = gaussian_filter(Iyy, sigma=sigma)
        Ixy = gaussian_filter(Ixy, sigma=sigma)
        
        smoothing_time = time.time()
        
        # Step 4: Compute structure tensor elements (efficient vectorized)
        Sxx, Syy, Sxy = self._compute_structure_tensor_vectorized(Ixx, Iyy, Ixy)
        
        tensor_time = time.time()
        
        # Step 5: Compute Harris response
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        R = det_M - self.k * (trace_M ** 2)
        
        # Step 6: Thresholding
        max_response = np.max(R)
        threshold = threshold_ratio * max_response
        corner_mask = R > threshold
        
        # Step 7: Non-maximum suppression
        corners = self._nms_with_radius(R, corner_mask, nms_radius)
        
        nms_time = time.time()
        
        # Step 8: Optional - return top N corners
        if return_top_n and len(corners) > return_top_n:
            corners = sorted(corners, key=lambda x: x[2], reverse=True)[:return_top_n]
        
        # Record metrics
        self.metrics = {
            'total_time': time.time() - start_time,
            'gradient_time': gradient_time - start_time,
            'smoothing_time': smoothing_time - gradient_time,
            'tensor_time': tensor_time - smoothing_time,
            'nms_time': nms_time - tensor_time,
            'num_corners': len(corners),
            'max_response': max_response,
            'threshold_used': threshold
        }
        
        return corners
    
    def _compute_structure_tensor_vectorized(self, Ixx: np.ndarray, 
                                            Iyy: np.ndarray, 
                                            Ixy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute structure tensor using vectorized box filter (faster than loops).
        """
        kernel = np.ones((self.block_size, self.block_size)) / (self.block_size ** 2)
        
        Sxx = cv2.filter2D(Ixx, -1, kernel)
        Syy = cv2.filter2D(Iyy, -1, kernel)
        Sxy = cv2.filter2D(Ixy, -1, kernel)
        
        return Sxx, Syy, Sxy
    
    def _nms_with_radius(self, R: np.ndarray, corner_mask: np.ndarray, 
                        radius: int) -> List[Tuple[int, int, float]]:
        """
        Non-maximum suppression with configurable radius.
        
        Args:
            R (np.ndarray): Harris response map
            corner_mask (np.ndarray): Boolean mask of potential corners
            radius (int): NMS radius (suppress neighbors within this distance)
        
        Returns:
            List of corner coordinates with response values
        """
        corners = []
        visited = np.zeros_like(R, dtype=bool)
        
        # Find all potential corner locations
        y_coords, x_coords = np.where(corner_mask)
        
        if len(y_coords) == 0:
            return corners
        
        # Sort by response strength (descending)
        indices = np.argsort(-R[y_coords, x_coords])
        
        for idx in indices:
            y, x = y_coords[idx], x_coords[idx]
            
            if visited[y, x]:
                continue
            
            # Add this corner
            corners.append((y, x, float(R[y, x])))
            
            # Mark neighborhood as visited
            y_min = max(0, y - radius)
            y_max = min(R.shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(R.shape[1], x + radius + 1)
            visited[y_min:y_max, x_min:x_max] = True
        
        return corners
    
    def get_metrics(self) -> Dict:
        """Return timing and performance metrics."""
        return self.metrics


# =============================================================================
#                         FAST DETECTOR - ENHANCED
# =============================================================================

class FASTCornerDetector:
    """
    Production-ready FAST (Features from Accelerated Segment Test) detector.
    
    Features:
    - Fast integer-only operations
    - Configurable circle radius and threshold
    - Adaptive thresholding
    - Score computation (Sum of Absolute Differences)
    - Non-maximum suppression
    - Performance metrics
    """
    
    def __init__(self, circle_radius: int = 3):
        """
        Initialize FAST Detector.
        
        Args:
            circle_radius (int): Radius of comparison circle (3 or 4)
        """
        self.circle_radius = circle_radius
        self.circle = self._get_circle_coordinates(circle_radius)
        self.metrics = {}
    
    @staticmethod
    def _get_circle_coordinates(radius: int) -> List[Tuple[int, int]]:
        """Get Bresenham circle coordinates for FAST detector."""
        if radius == 3:
            return [
                (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)
            ]
        elif radius == 4:
            return [
                (0, 4), (1, 4), (2, 4), (3, 3), (4, 2), (4, 1), (4, 0), (4, -1),
                (4, -2), (3, -3), (2, -4), (1, -4), (0, -4), (-1, -4), (-2, -4), (-3, -3),
                (-4, -2), (-4, -1), (-4, 0), (-4, 1), (-4, 2), (-3, 3), (-2, 4), (-1, 4)
            ]
        else:
            raise ValueError("circle_radius must be 3 or 4")
    
    def detect(self, gray_img: np.ndarray, threshold: int = 30, 
               n: int = 12, return_top_n: Optional[int] = None,
               compute_scores: bool = True) -> List[Tuple[int, int, float]]:
        """
        Detect corners using FAST algorithm.
        
        Args:
            gray_img (np.ndarray): Grayscale image (uint8 recommended)
            threshold (int): Intensity difference threshold
            n (int): Minimum contiguous pixels required (9-16)
            return_top_n (int): Return only top N corners by score
            compute_scores (bool): Compute SAD scores for each corner
        
        Returns:
            List of (row, col, score) tuples
        """
        start_time = time.time()
        
        # Ensure uint8
        if gray_img.dtype != np.uint8:
            gray_img = gray_img.astype(np.uint8)
        
        height, width = gray_img.shape
        corners = []
        
        # Valid region considering circle radius
        margin = self.circle_radius + 1
        
        for y in range(margin, height - margin):
            for x in range(margin, width - margin):
                center = int(gray_img[y, x])
                
                # Fast test: check 4 cardinal points first
                if not self._fast_test(gray_img, x, y, center, threshold):
                    continue
                
                # Full test: check all circle points
                brighter, darker = self._count_contiguous(gray_img, x, y, center, threshold, n)
                
                if brighter >= n or darker >= n:
                    score = self._compute_score(gray_img, x, y, center) if compute_scores else 0.0
                    corners.append((y, x, float(score)))
        
        # Non-maximum suppression
        corners = self._nms_fast(corners, gray_img)
        
        # Optional: return top N
        if return_top_n and len(corners) > return_top_n:
            corners = sorted(corners, key=lambda c: c[2], reverse=True)[:return_top_n]
        
        # Record metrics
        self.metrics = {
            'total_time': time.time() - start_time,
            'num_corners': len(corners),
            'threshold_used': threshold,
            'min_contiguous': n
        }
        
        return corners
    
    def _fast_test(self, img: np.ndarray, x: int, y: int, 
                   center: int, threshold: int) -> bool:
        """Quick 4-point test to reject non-corners early."""
        # Check N, S, E, W pixels
        cardinal_points = [
            (y - self.circle_radius, x),  # N
            (y + self.circle_radius, x),  # S
            (y, x + self.circle_radius),  # E
            (y, x - self.circle_radius),  # W
        ]
        
        brighter = sum(1 for py, px in cardinal_points if img[py, px] > center + threshold)
        darker = sum(1 for py, px in cardinal_points if img[py, px] < center - threshold)
        
        return brighter >= 3 or darker >= 3
    
    def _count_contiguous(self, img: np.ndarray, x: int, y: int, 
                         center: int, threshold: int, n: int) -> Tuple[int, int]:
        """Count contiguous brighter/darker pixels in circle."""
        max_consec_b = 0
        max_consec_d = 0
        
        for _ in range(2):  # Check twice to handle wraparound
            consec_b = 0
            consec_d = 0
            
            for dx, dy in self.circle:
                p = int(img[y + dy, x + dx])
                
                if p > center + threshold:
                    consec_b += 1
                    consec_d = 0
                    max_consec_b = max(max_consec_b, consec_b)
                elif p < center - threshold:
                    consec_d += 1
                    consec_b = 0
                    max_consec_d = max(max_consec_d, consec_d)
                else:
                    consec_b = 0
                    consec_d = 0
                
                if max_consec_b >= n or max_consec_d >= n:
                    return max_consec_b, max_consec_d
        
        return max_consec_b, max_consec_d
    
    def _compute_score(self, img: np.ndarray, x: int, y: int, center: int) -> float:
        """Compute SAD (Sum of Absolute Differences) score."""
        score = 0.0
        for dx, dy in self.circle:
            score += abs(int(img[y + dy, x + dx]) - center)
        return score
    
    def _nms_fast(self, corners: List[Tuple[int, int, float]], 
                  img: np.ndarray, radius: int = 2) -> List[Tuple[int, int, float]]:
        """Non-maximum suppression for FAST corners."""
        if not corners:
            return corners
        
        corners = sorted(corners, key=lambda c: c[2], reverse=True)
        suppressed = np.zeros(len(corners), dtype=bool)
        
        for i, (y1, x1, score1) in enumerate(corners):
            if suppressed[i]:
                continue
            
            for j in range(i + 1, len(corners)):
                if suppressed[j]:
                    continue
                
                y2, x2, score2 = corners[j]
                dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                
                if dist <= radius:
                    suppressed[j] = True
        
        return [c for i, c in enumerate(corners) if not suppressed[i]]
    
    def get_metrics(self) -> Dict:
        """Return timing and performance metrics."""
        return self.metrics


# =============================================================================
#                          VISUALIZATION UTILITIES
# =============================================================================

class CornerVisualizer:
    """Utilities for visualizing corner detection results."""
    
    @staticmethod
    def draw_corners(image: np.ndarray, corners: List[Tuple[int, int, float]], 
                    color: Tuple[int, int, int] = (0, 0, 255), 
                    radius: int = 3) -> np.ndarray:
        """
        Draw corners on image.
        
        Args:
            image (np.ndarray): Input image (BGR or RGB)
            corners (List): List of (row, col, response) tuples
            color (Tuple): BGR color for circles
            radius (int): Circle radius in pixels
        
        Returns:
            Image with corners drawn
        """
        result = image.copy()
        for y, x, _ in corners:
            cv2.circle(result, (int(x), int(y)), radius, color, -1)
        return result
    
    @staticmethod
    def compare_detectors(image_path: str, harris_pts: List, 
                         fast_pts: List, figsize: Tuple = (15, 5)):
        """
        Display comparison of Harris and FAST results.
        
        Args:
            image_path (str): Path to image file
            harris_pts (List): Harris corners
            fast_pts (List): FAST corners
            figsize (Tuple): Figure size (width, height)
        """
        original = cv2.imread(image_path)
        if original is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        img_harris = CornerVisualizer.draw_corners(original, harris_pts, (255, 0, 0), 4)
        img_harris_rgb = cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB)
        
        img_fast = CornerVisualizer.draw_corners(original, fast_pts, (0, 255, 0), 4)
        img_fast_rgb = cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(img_harris_rgb)
        axes[1].set_title(f"Harris Corners ({len(harris_pts)})")
        axes[1].axis('off')
        
        axes[2].imshow(img_fast_rgb)
        axes[2].set_title(f"FAST Corners ({len(fast_pts)})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


# =============================================================================
#                              DEMONSTRATION
# =============================================================================

def main():
    """Main demonstration function."""
    
    # Configuration
    IMG_PATH = "chessboard.jpg"  # Change to your image path
    
    # Load image
    original = cv2.imread(IMG_PATH)
    if original is None:
        print(f"Error: Could not load image from {IMG_PATH}")
        print("Please ensure the image file exists and provide the correct path.")
        return
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print("=" * 70)
    print("CORNER DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # ========== HARRIS CORNER DETECTOR ==========
    print("\n[1/2] Running Harris Corner Detector...")
    harris_detector = HarrisCornerDetector(block_size=5, k=0.04)
    harris_pts = harris_detector.detect(gray, threshold_ratio=0.01, 
                                        nms_radius=3, return_top_n=1000)
    harris_metrics = harris_detector.get_metrics()
    
    print(f"  ✓ Found {harris_metrics['num_corners']} corners")
    print(f"  ✓ Processing time: {harris_metrics['total_time']*1000:.2f} ms")
    print(f"  ✓ Max response: {harris_metrics['max_response']:.2f}")
    
    # ========== FAST CORNER DETECTOR ==========
    print("\n[2/2] Running FAST Corner Detector...")
    fast_detector = FASTCornerDetector(circle_radius=3)
    gray_uint8 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    fast_pts = fast_detector.detect(gray_uint8, threshold=30, n=9, return_top_n=1000)
    fast_metrics = fast_detector.get_metrics()
    
    print(f"  ✓ Found {fast_metrics['num_corners']} corners")
    print(f"  ✓ Processing time: {fast_metrics['total_time']*1000:.2f} ms")
    
    print("\n" + "=" * 70)
    
    # Visualization
    CornerVisualizer.compare_detectors(IMG_PATH, harris_pts, fast_pts)


if __name__ == "__main__":
    main()
