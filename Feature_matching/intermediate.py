# =============================================================================
# advanced_corner_detection.py
# Comprehensive Corner Detection with 5 Advanced Features
# - Scale-space FAST (multi-scale detection)
# - ML-accelerated FAST (decision tree optimization)
# - Rotation invariance testing
# - KLT (Shi-Tomasi) corner response
# - Temporal consistency analysis
# =============================================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import sobel, gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.tree import DecisionTreeClassifier
from typing import List, Tuple, Optional, Dict
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
#                    1. HARRIS CORNER DETECTOR (BASELINE)
# =============================================================================

class HarrisCornerDetector:
    """Harris Corner Detector with KLT variant support."""
    
    def __init__(self, block_size: int = 5, ksize: int = 3, k: float = 0.04, use_klt: bool = False):
        """
        Initialize Harris or KLT corner detector.
        
        Args:
            block_size (int): Window size for structure tensor
            ksize (int): Sobel kernel size
            k (float): Harris constant (0.04-0.06)
            use_klt (bool): Use KLT (Shi-Tomasi) response instead of Harris
        """
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.ksize = ksize
        self.k = k
        self.use_klt = use_klt
        self.metrics = {}
    
    def detect(self, gray_img: np.ndarray, threshold_ratio: float = 0.01,
               nms_radius: int = 3, return_top_n: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """
        Detect corners using Harris or KLT response.
        
        Args:
            gray_img (np.ndarray): Grayscale image (float32)
            threshold_ratio (float): Fraction of max response for detection
            nms_radius (int): Non-maximum suppression radius
            return_top_n (int): Return only top N corners
        
        Returns:
            List of (row, col, response_strength) tuples
        """
        start_time = time.time()
        
        if gray_img.dtype != np.float32:
            gray_img = gray_img.astype(np.float32)
        
        # Compute gradients
        Ix = sobel(gray_img, axis=1, mode='constant')
        Iy = sobel(gray_img, axis=0, mode='constant')
        
        # Compute structure tensor elements
        Ixx = gaussian_filter(Ix * Ix, sigma=self.block_size / 5)
        Iyy = gaussian_filter(Iy * Iy, sigma=self.block_size / 5)
        Ixy = gaussian_filter(Ix * Iy, sigma=self.block_size / 5)
        
        # Vectorized structure tensor computation
        kernel = np.ones((self.block_size, self.block_size)) / (self.block_size ** 2)
        Sxx = cv2.filter2D(Ixx, -1, kernel)
        Syy = cv2.filter2D(Iyy, -1, kernel)
        Sxy = cv2.filter2D(Ixy, -1, kernel)
        
        # Compute corner response
        det_M = Sxx * Syy - Sxy * Sxy
        trace_M = Sxx + Syy
        
        if self.use_klt:
            # KLT (Shi-Tomasi) response: min(λ₁, λ₂)
            # Approximation: det(M) / (1 + trace(M))
            # Better: use eigenvalues directly
            R = np.zeros_like(det_M)
            for i in range(det_M.shape[0]):
                for j in range(det_M.shape[1]):
                    M = np.array([[Sxx[i, j], Sxy[i, j]], [Sxy[i, j], Syy[i, j]]])
                    eigenvalues = np.linalg.eigvalsh(M)
                    R[i, j] = np.min(eigenvalues)  # min(λ₁, λ₂)
        else:
            # Harris response: det(M) - k*trace(M)²
            R = det_M - self.k * (trace_M ** 2)
        
        # Thresholding and NMS
        max_response = np.max(R)
        threshold = threshold_ratio * max_response
        corner_mask = R > threshold
        
        corners = self._nms_with_radius(R, corner_mask, nms_radius)
        
        if return_top_n and len(corners) > return_top_n:
            corners = sorted(corners, key=lambda x: x[2], reverse=True)[:return_top_n]
        
        self.metrics = {
            'total_time': time.time() - start_time,
            'num_corners': len(corners),
            'detector_type': 'KLT (Shi-Tomasi)' if self.use_klt else 'Harris',
            'max_response': max_response
        }
        
        return corners
    
    def _nms_with_radius(self, R: np.ndarray, corner_mask: np.ndarray, 
                        radius: int) -> List[Tuple[int, int, float]]:
        """Non-maximum suppression with configurable radius."""
        corners = []
        visited = np.zeros_like(R, dtype=bool)
        
        y_coords, x_coords = np.where(corner_mask)
        if len(y_coords) == 0:
            return corners
        
        indices = np.argsort(-R[y_coords, x_coords])
        
        for idx in indices:
            y, x = y_coords[idx], x_coords[idx]
            
            if visited[y, x]:
                continue
            
            corners.append((y, x, float(R[y, x])))
            
            y_min, y_max = max(0, y - radius), min(R.shape[0], y + radius + 1)
            x_min, x_max = max(0, x - radius), min(R.shape[1], x + radius + 1)
            visited[y_min:y_max, x_min:x_max] = True
        
        return corners


# =============================================================================
#              2. FAST DETECTOR WITH SCALE-SPACE SELECTION
# =============================================================================

class FASTScaleSpace:
    """
    FAST detector with multi-scale selection.
    Runs FAST on multiple downsampled versions and returns scale-aware keypoints.
    """
    
    def __init__(self, circle_radius: int = 3):
        """Initialize FAST scale-space detector."""
        self.circle_radius = circle_radius
        self.circle = self._get_circle_coordinates(circle_radius)
        self.metrics = {}
    
    @staticmethod
    def _get_circle_coordinates(radius: int) -> List[Tuple[int, int]]:
        """Get Bresenham circle coordinates."""
        if radius == 3:
            return [
                (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)
            ]
        else:
            raise ValueError("circle_radius must be 3")
    
    def detect_multiscale(self, gray_img: np.ndarray, scales: List[float] = None,
                         threshold: int = 30, n: int = 12, 
                         return_top_n: Optional[int] = None) -> List[Tuple[int, int, float, float]]:
        """
        Detect corners at multiple scales.
        
        Args:
            gray_img (np.ndarray): Grayscale image
            scales (List[float]): Scale factors [0.5, 0.75, 1.0, 1.5, 2.0]
            threshold (int): Intensity threshold
            n (int): Minimum contiguous pixels
            return_top_n (int): Return top N corners
        
        Returns:
            List of (row, col, response_score, scale) tuples
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        
        start_time = time.time()
        all_corners = []
        
        for scale_idx, scale in enumerate(scales):
            if scale == 1.0:
                scaled_img = gray_img.copy()
            else:
                h, w = gray_img.shape
                scaled_img = cv2.resize(gray_img, (int(w * scale), int(h * scale)))
            
            if scaled_img.dtype != np.uint8:
                scaled_img = scaled_img.astype(np.uint8)
            
            height, width = scaled_img.shape
            margin = self.circle_radius + 1
            
            # Detect at this scale
            for y in range(margin, height - margin):
                for x in range(margin, width - margin):
                    center = int(scaled_img[y, x])
                    
                    if not self._fast_test(scaled_img, x, y, center, threshold):
                        continue
                    
                    brighter, darker = self._count_contiguous(scaled_img, x, y, center, threshold, n)
                    
                    if brighter >= n or darker >= n:
                        # Scale back to original coordinates
                        orig_y = int(y / scale) if scale != 1.0 else y
                        orig_x = int(x / scale) if scale != 1.0 else x
                        score = self._compute_score(scaled_img, x, y, center)
                        all_corners.append((orig_y, orig_x, float(score), float(scale)))
        
        # Remove duplicates and sort by score
        unique_corners = self._remove_scale_duplicates(all_corners)
        unique_corners = sorted(unique_corners, key=lambda c: c[2], reverse=True)
        
        if return_top_n and len(unique_corners) > return_top_n:
            unique_corners = unique_corners[:return_top_n]
        
        self.metrics = {
            'total_time': time.time() - start_time,
            'num_scales': len(scales),
            'num_corners': len(unique_corners),
            'scales_used': scales
        }
        
        return unique_corners
    
    def _remove_scale_duplicates(self, corners: List, distance_threshold: int = 5) -> List:
        """Remove corners detected at multiple scales within distance threshold."""
        if not corners:
            return corners
        
        filtered = []
        used = set()
        
        for i, (y1, x1, score1, scale1) in enumerate(corners):
            if i in used:
                continue
            
            filtered.append((y1, x1, score1, scale1))
            
            for j in range(i + 1, len(corners)):
                if j in used:
                    continue
                
                y2, x2, score2, scale2 = corners[j]
                dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                
                if dist <= distance_threshold:
                    used.add(j)
        
        return filtered
    
    def _fast_test(self, img: np.ndarray, x: int, y: int, center: int, threshold: int) -> bool:
        """Quick 4-point cardinal test."""
        cardinal = [
            (y - self.circle_radius, x),
            (y + self.circle_radius, x),
            (y, x + self.circle_radius),
            (y, x - self.circle_radius),
        ]
        brighter = sum(1 for py, px in cardinal if img[py, px] > center + threshold)
        darker = sum(1 for py, px in cardinal if img[py, px] < center - threshold)
        return brighter >= 3 or darker >= 3
    
    def _count_contiguous(self, img: np.ndarray, x: int, y: int, 
                         center: int, threshold: int, n: int) -> Tuple[int, int]:
        """Count contiguous brighter/darker pixels."""
        max_consec_b = 0
        max_consec_d = 0
        
        for _ in range(2):
            consec_b = consec_d = 0
            
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
                    consec_b = consec_d = 0
                
                if max_consec_b >= n or max_consec_d >= n:
                    return max_consec_b, max_consec_d
        
        return max_consec_b, max_consec_d
    
    def _compute_score(self, img: np.ndarray, x: int, y: int, center: int) -> float:
        """Compute SAD score."""
        return sum(abs(int(img[y + dy, x + dx]) - center) for dx, dy in self.circle)


# =============================================================================
#         3. ML-ACCELERATED FAST (DECISION TREE OPTIMIZATION)
# =============================================================================

class MLAcceleratedFAST:
    """
    FAST with machine learning acceleration.
    Trains decision tree to predict corner vs non-corner using fewer checks.
    Based on original FAST paper's decision tree approach.
    """
    
    def __init__(self, circle_radius: int = 3):
        """Initialize ML-accelerated FAST."""
        self.circle_radius = circle_radius
        self.circle = self._get_circle_coordinates(circle_radius)
        self.decision_tree = None
        self.metrics = {}
    
    @staticmethod
    def _get_circle_coordinates(radius: int) -> List[Tuple[int, int]]:
        """Get Bresenham circle coordinates."""
        if radius == 3:
            return [
                (0, 3), (1, 3), (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                (0, -3), (-1, -3), (-2, -2), (-3, -1), (-3, 0), (-3, 1), (-2, 2), (-1, 3)
            ]
        else:
            raise ValueError("circle_radius must be 3")
    
    def train(self, training_images: List[np.ndarray], threshold: int = 30, n: int = 12):
        """
        Train decision tree on training images.
        
        Args:
            training_images (List[np.ndarray]): List of grayscale images
            threshold (int): Intensity threshold
            n (int): Minimum contiguous pixels
        """
        print(f"  Training ML-accelerated FAST on {len(training_images)} images...")
        
        X = []  # Features (subset of circle pixel differences)
        y = []  # Labels (corner: 1, non-corner: 0)
        
        # Use 4 cardinal points + 4 diagonal points = 8 checks
        check_indices = [1, 5, 9, 13, 2, 4, 10, 14]  # Cardinal + diagonal
        
        for img_idx, img in enumerate(training_images):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            h, w = img.shape
            margin = self.circle_radius + 1
            
            # Sample points
            for y_pos in range(margin, h - margin, 10):
                for x_pos in range(margin, w - margin, 10):
                    center = int(img[y_pos, x_pos])
                    
                    # Extract features for selected checks
                    features = []
                    for check_idx in check_indices:
                        dx, dy = self.circle[check_idx]
                        diff = int(img[y_pos + dy, x_pos + dx]) - center
                        features.append(diff)
                    
                    # Full FAST test for label
                    brighter, darker = self._count_contiguous(img, x_pos, y_pos, center, threshold, n)
                    is_corner = 1 if (brighter >= n or darker >= n) else 0
                    
                    X.append(features)
                    y.append(is_corner)
        
        # Train decision tree
        self.decision_tree = DecisionTreeClassifier(max_depth=8, random_state=42)
        self.decision_tree.fit(X, y)
        
        print(f"  Decision tree trained with {len(X)} samples")
        print(f"  Tree depth: {self.decision_tree.get_depth()}")
    
    def detect(self, gray_img: np.ndarray, threshold: int = 30, n: int = 12,
               return_top_n: Optional[int] = None) -> List[Tuple[int, int, float]]:
        """
        Detect corners using ML-accelerated approach.
        
        Args:
            gray_img (np.ndarray): Grayscale image
            threshold (int): Intensity threshold
            n (int): Minimum contiguous pixels
            return_top_n (int): Return top N corners
        
        Returns:
            List of (row, col, score) tuples
        """
        start_time = time.time()
        
        if gray_img.dtype != np.uint8:
            gray_img = gray_img.astype(np.uint8)
        
        h, w = gray_img.shape
        margin = self.circle_radius + 1
        corners = []
        
        check_indices = [1, 5, 9, 13, 2, 4, 10, 14]  # 8 checks
        
        for y_pos in range(margin, h - margin):
            for x_pos in range(margin, w - margin):
                center = int(gray_img[y_pos, x_pos])
                
                # Extract 8 features
                features = []
                for check_idx in check_indices:
                    dx, dy = self.circle[check_idx]
                    diff = int(gray_img[y_pos + dy, x_pos + dx]) - center
                    features.append(diff)
                
                # ML prediction (fast)
                if self.decision_tree is None:
                    # Fallback: use standard FAST test
                    brighter, darker = self._count_contiguous(gray_img, x_pos, y_pos, center, threshold, n)
                    if brighter >= n or darker >= n:
                        score = self._compute_score(gray_img, x_pos, y_pos, center)
                        corners.append((y_pos, x_pos, float(score)))
                else:
                    prediction = self.decision_tree.predict([features])[0]
                    
                    if prediction == 1:
                        # Predicted corner - verify with full test
                        brighter, darker = self._count_contiguous(gray_img, x_pos, y_pos, center, threshold, n)
                        if brighter >= n or darker >= n:
                            score = self._compute_score(gray_img, x_pos, y_pos, center)
                            corners.append((y_pos, x_pos, float(score)))
        
        if return_top_n and len(corners) > return_top_n:
            corners = sorted(corners, key=lambda c: c[2], reverse=True)[:return_top_n]
        
        self.metrics = {
            'total_time': time.time() - start_time,
            'num_corners': len(corners),
            'ml_accelerated': self.decision_tree is not None
        }
        
        return corners
    
    def _count_contiguous(self, img: np.ndarray, x: int, y: int,
                         center: int, threshold: int, n: int) -> Tuple[int, int]:
        """Count contiguous brighter/darker pixels."""
        max_consec_b = 0
        max_consec_d = 0
        
        for _ in range(2):
            consec_b = consec_d = 0
            
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
                    consec_b = consec_d = 0
                
                if max_consec_b >= n or max_consec_d >= n:
                    return max_consec_b, max_consec_d
        
        return max_consec_b, max_consec_d
    
    def _compute_score(self, img: np.ndarray, x: int, y: int, center: int) -> float:
        """Compute SAD score."""
        return sum(abs(int(img[y + dy, x + dx]) - center) for dx, dy in self.circle)


# =============================================================================
#            4. ROTATION INVARIANCE TESTING
# =============================================================================

class RotationInvarianceTest:
    """Test corner detector robustness to rotation."""
    
    @staticmethod
    def test_rotation_invariance(detector, gray_img: np.ndarray,
                                 angles: List[float] = None,
                                 distance_threshold: float = 5.0) -> Dict:
        """
        Test rotation invariance by rotating image and checking corner stability.
        
        Args:
            detector: Corner detector instance
            gray_img (np.ndarray): Grayscale image
            angles (List[float]): Rotation angles in degrees [5, 15, 30, 45]
            distance_threshold (float): Distance threshold for matching corners
        
        Returns:
            Dictionary with repeatability metrics
        """
        if angles is None:
            angles = [5, 15, 30, 45]
        
        print("\n" + "="*60)
        print("ROTATION INVARIANCE TEST")
        print("="*60)
        
        h, w = gray_img.shape
        center = (w // 2, h // 2)
        
        # Get baseline corners from original image
        if hasattr(detector, 'detect_multiscale'):
            baseline_corners = detector.detect_multiscale(gray_img)
        else:
            baseline_corners = detector.detect(gray_img.astype(np.float32), return_top_n=100)
        
        baseline_pts = np.array([(c[0], c[1]) for c in baseline_corners])
        
        results = {
            'baseline_corners': len(baseline_corners),
            'angles_tested': angles,
            'repeatability_by_angle': {}
        }
        
        print(f"\nBaseline (0°): {len(baseline_corners)} corners detected")
        print(f"\nTesting rotations: {angles}°")
        print("-" * 60)
        
        for angle in angles:
            # Rotate image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_img = cv2.warpAffine(gray_img, rotation_matrix, (w, h))
            
            # Detect corners
            if hasattr(detector, 'detect_multiscale'):
                rotated_corners = detector.detect_multiscale(rotated_img)
            else:
                rotated_corners = detector.detect(rotated_img.astype(np.float32), return_top_n=100)
            
            rotated_pts = np.array([(c[0], c[1]) for c in rotated_corners])
            
            # Match corners
            if len(baseline_pts) > 0 and len(rotated_pts) > 0:
                distances = cdist(baseline_pts, rotated_pts)
                min_distances = np.min(distances, axis=1)
                matched = np.sum(min_distances <= distance_threshold)
                repeatability = (matched / len(baseline_pts)) * 100
            else:
                repeatability = 0.0
            
            results['repeatability_by_angle'][angle] = {
                'detected': len(rotated_corners),
                'matched': matched if len(baseline_pts) > 0 else 0,
                'repeatability_pct': repeatability
            }
            
            print(f"  {angle:3d}°: {len(rotated_corners):3d} corners, "
                  f"{matched}/{len(baseline_pts)} matched ({repeatability:.1f}% repeatability)")
        
        print("-" * 60)
        avg_repeatability = np.mean([v['repeatability_pct'] 
                                    for v in results['repeatability_by_angle'].values()])
        print(f"Average repeatability: {avg_repeatability:.1f}%")
        print("="*60 + "\n")
        
        return results


# =============================================================================
#            5. TEMPORAL CONSISTENCY TEST
# =============================================================================

class TemporalConsistencyTest:
    """Test corner stability across consecutive video frames."""
    
    @staticmethod
    def test_temporal_consistency(detector, video_path: str, 
                                 num_frames: int = 5,
                                 distance_threshold: float = 5.0,
                                 skip_frames: int = 0) -> Dict:
        """
        Test temporal consistency on video frames.
        
        Args:
            detector: Corner detector instance
            video_path (str): Path to video file
            num_frames (int): Number of consecutive frames to test
            distance_threshold (float): Distance for corner matching
            skip_frames (int): Frames to skip between analysis
        
        Returns:
            Dictionary with temporal stability metrics
        """
        print("\n" + "="*60)
        print("TEMPORAL CONSISTENCY TEST")
        print("="*60)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {}
        
        frames = []
        frame_corners = []
        
        # Extract frames
        frame_count = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
            
            # Detect corners
            if hasattr(detector, 'detect_multiscale'):
                corners = detector.detect_multiscale(gray)
            else:
                corners = detector.detect(gray.astype(np.float32), return_top_n=100)
            
            frame_corners.append(corners)
            frame_count += 1
        
        cap.release()
        
        print(f"Extracted {len(frames)} frames from video")
        print(f"\nFrame-by-frame corner detection:")
        print("-" * 60)
        
        for i, corners in enumerate(frame_corners):
            print(f"Frame {i}: {len(corners)} corners")
        
        print("-" * 60)
        
        # Compute temporal consistency
        results = {
            'num_frames': len(frames),
            'corners_per_frame': [len(c) for c in frame_corners],
            'frame_to_frame_consistency': {}
        }
        
        if len(frames) > 1:
            print("\nFrame-to-frame stability:")
            print("-" * 60)
            
            for i in range(len(frames) - 1):
                pts1 = np.array([(c[0], c[1]) for c in frame_corners[i]])
                pts2 = np.array([(c[0], c[1]) for c in frame_corners[i + 1]])
                
                if len(pts1) > 0 and len(pts2) > 0:
                    distances = cdist(pts1, pts2)
                    min_distances = np.min(distances, axis=1)
                    stable = np.sum(min_distances <= distance_threshold)
                    stability = (stable / len(pts1)) * 100
                else:
                    stable = 0
                    stability = 0.0
                
                results['frame_to_frame_consistency'][f'frame_{i}_to_{i+1}'] = {
                    'stable_corners': stable,
                    'stability_pct': stability
                }
                
                print(f"Frame {i}→{i+1}: {stable}/{len(pts1)} stable ({stability:.1f}%)")
        
        print("="*60 + "\n")
        
        return results


# =============================================================================
#                          VISUALIZATION SUITE
# =============================================================================

class CornerVisualizerAdvanced:
    """Advanced visualization utilities."""
    
    @staticmethod
    def draw_corners(image: np.ndarray, corners: List, 
                    color: Tuple = (0, 0, 255), radius: int = 3) -> np.ndarray:
        """Draw corners on image."""
        result = image.copy()
        for corner in corners:
            y, x = int(corner[0]), int(corner[1])
            cv2.circle(result, (x, y), radius, color, -1)
        return result
    
    @staticmethod
    def visualize_scale_space(image_path: str, detector):
        """Visualize corners detected at different scales."""
        original = cv2.imread(image_path)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        corners = detector.detect_multiscale(gray)
        
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        
        result = CornerVisualizerAdvanced.draw_corners(original, corners, (0, 255, 0), 4)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        axes.imshow(result_rgb)
        axes.set_title(f"Scale-Space FAST: {len(corners)} corners detected")
        axes.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_rotation_test(image_path: str, results: Dict):
        """Visualize rotation invariance test results."""
        angles = list(results['repeatability_by_angle'].keys())
        repeatabilities = [results['repeatability_by_angle'][a]['repeatability_pct'] 
                          for a in angles]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(angles, repeatabilities, 'o-', linewidth=2, markersize=8, color='#0284c7')
        ax.fill_between(angles, repeatabilities, alpha=0.3, color='#0284c7')
        
        ax.set_xlabel('Rotation Angle (degrees)', fontsize=12)
        ax.set_ylabel('Corner Repeatability (%)', fontsize=12)
        ax.set_title('Rotation Invariance Test Results', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        for angle, rep in zip(angles, repeatabilities):
            ax.text(angle, rep + 2, f'{rep:.1f}%', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
#                       COMPREHENSIVE DEMONSTRATION
# =============================================================================

def main():
    """Main demonstration with all 5 advanced features."""
    
    IMG_PATH = "chessboard.jpg"  # Your image
    VIDEO_PATH = None  # Set to video path for temporal test
    
    # Load image
    original = cv2.imread(IMG_PATH)
    if original is None:
        print(f"Error: Could not load image from {IMG_PATH}")
        return
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print("\n" + "="*70)
    print("ADVANCED CORNER DETECTION SUITE - COMPREHENSIVE DEMONSTRATION")
    print("="*70)
    
    # ==== 1. HARRIS vs KLT ====
    print("\n[1/5] Harris vs KLT (Shi-Tomasi) Detector")
    print("-" * 70)
    
    harris = HarrisCornerDetector(use_klt=False)
    harris_corners = harris.detect(gray, return_top_n=100)
    print(f"Harris: {harris.metrics['num_corners']} corners in {harris.metrics['total_time']*1000:.2f}ms")
    
    klt = HarrisCornerDetector(use_klt=True)
    klt_corners = klt.detect(gray, return_top_n=100)
    print(f"KLT:    {klt.metrics['num_corners']} corners in {klt.metrics['total_time']*1000:.2f}ms")
    
    # ==== 2. SCALE-SPACE FAST ====
    print("\n[2/5] Scale-Space FAST Detection")
    print("-" * 70)
    
    fast_ss = FASTScaleSpace()
    scales_used = [0.5, 0.75, 1.0, 1.5, 2.0]
    fast_ss_corners = fast_ss.detect_multiscale(gray.astype(np.uint8), 
                                                scales=scales_used, return_top_n=100)
    print(f"Scale-Space FAST: {len(fast_ss_corners)} corners")
    print(f"Scales used: {scales_used}")
    print(f"Processing time: {fast_ss.metrics['total_time']*1000:.2f}ms")
    
    # Visualize
    CornerVisualizerAdvanced.visualize_scale_space(IMG_PATH, fast_ss)
    
    # ==== 3. ML-ACCELERATED FAST ====
    print("\n[3/5] ML-Accelerated FAST")
    print("-" * 70)
    
    ml_fast = MLAcceleratedFAST()
    # Train on the image itself (in practice, use diverse training set)
    ml_fast.train([gray.astype(np.uint8)], threshold=30, n=9)
    ml_fast_corners = ml_fast.detect(gray.astype(np.uint8), return_top_n=1000)
    print(f"ML-FAST: {len(ml_fast_corners)} corners in {ml_fast.metrics['total_time']*1000:.2f}ms")
    
    # ==== 4. ROTATION INVARIANCE TEST ====
    print("\n[4/5] Rotation Invariance Test")
    
    rotation_results = RotationInvarianceTest.test_rotation_invariance(
        fast_ss, gray.astype(np.uint8), angles=[5, 15, 30, 45]
    )
    
    CornerVisualizerAdvanced.visualize_rotation_test(IMG_PATH, rotation_results)
    
    # ==== 5. TEMPORAL CONSISTENCY TEST ====
    print("\n[5/5] Temporal Consistency Test")
    print("-" * 70)
    
    if VIDEO_PATH and cv2.VideoCapture(VIDEO_PATH).isOpened():
        temporal_results = TemporalConsistencyTest.test_temporal_consistency(
            fast_ss, VIDEO_PATH, num_frames=5
        )
        print(f"Temporal test completed on {temporal_results.get('num_frames', 0)} frames")
    else:
        print("Video path not provided or file not found.")
        print("To test temporal consistency, provide a valid VIDEO_PATH")
    
    # ==== FINAL COMPARISON ====
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nHarris:         {harris.metrics['num_corners']:3d} corners ({harris.metrics['total_time']*1000:6.2f}ms)")
    print(f"KLT:            {klt.metrics['num_corners']:3d} corners ({klt.metrics['total_time']*1000:6.2f}ms)")
    print(f"Scale-Space:    {len(fast_ss_corners):3d} corners ({fast_ss.metrics['total_time']*1000:6.2f}ms)")
    print(f"ML-FAST:        {len(ml_fast_corners):3d} corners ({ml_fast.metrics['total_time']*1000:6.2f}ms)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
