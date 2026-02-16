# Face and Eye Detection using Haar Cascade Classifiers
# =====================================================
# A complete OpenCV project for detecting faces and eyes in images and real-time video

import cv2
import numpy as np
import os
from pathlib import Path


class FaceEyeDetector:
    """
    A comprehensive face and eye detection system using Haar Cascade classifiers.
    
    Features:
    - Face detection with configurable parameters
    - Eye detection within detected face regions (ROI)
    - Support for both images and video streams
    - Parameter tuning for handling various scenarios
    - Visualization with bounding boxes
    """
    
    def __init__(self, cascade_path_face=None, cascade_path_eye=None):
        """
        Initialize the detector with Haar Cascade XML files.
        
        Args:
            cascade_path_face (str): Path to frontalface_default.xml
            cascade_path_eye (str): Path to haarcascade_eye.xml
        """
        # Use OpenCV's built-in cascades if paths not provided
        if cascade_path_face is None:
            cascade_path_face = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        if cascade_path_eye is None:
            cascade_path_eye = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        # Load cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cascade_path_face)
        self.eye_cascade = cv2.CascadeClassifier(cascade_path_eye)
        
        # Verify cascades loaded successfully
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load face cascade from {cascade_path_face}")
        if self.eye_cascade.empty():
            raise ValueError(f"Failed to load eye cascade from {cascade_path_eye}")
        
        print("✓ Cascades loaded successfully")
    
    def detect_faces_and_eyes(self, image, scale_factor=1.2, min_neighbors=5, 
                             min_face_size=(30, 30), max_face_size=None,
                             draw_faces=True, draw_eyes=True):
        """
        Detect faces and eyes in an image.
        
        Args:
            image (np.ndarray): Input image (BGR format)
            scale_factor (float): Image pyramid scale factor (1.05-1.4, default 1.2)
                                 Lower = more accurate but slower
                                 Higher = faster but less accurate
            min_neighbors (int): Minimum neighbors for filtering (3-6, default 5)
                               Higher = fewer false positives but may miss faces
                               Lower = more detections but more false positives
            min_face_size (tuple): Minimum face size (width, height)
            max_face_size (tuple): Maximum face size or None for no limit
            draw_faces (bool): Draw face rectangles
            draw_eyes (bool): Draw eye circles
            
        Returns:
            dict: Contains detected faces, eyes, and annotated image
        """
        # Convert to grayscale for cascade classifier
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_face_size,
            maxSize=max_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Copy image for drawing
        result_image = image.copy()
        detected_eyes = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            if draw_faces:
                # Draw blue rectangle around face
                cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face region (ROI) for eye detection
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result_image[y:y+h, x:x+w]
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(15, 15)
            )
            
            # Draw eyes if found
            for (ex, ey, ew, eh) in eyes:
                if draw_eyes:
                    # Draw green circle around eye
                    center = (ex + ew//2, ey + eh//2)
                    radius = (ew + eh)//4
                    cv2.circle(roi_color, center, radius, (0, 255, 0), 2)
                
                # Store eye coordinates relative to full image
                detected_eyes.append({
                    'face_id': len(detected_eyes),
                    'x': x + ex,
                    'y': y + ey,
                    'width': ew,
                    'height': eh,
                    'center': (x + ex + ew//2, y + ey + eh//2)
                })
        
        return {
            'faces': faces,
            'eyes': detected_eyes,
            'annotated_image': result_image,
            'num_faces': len(faces),
            'num_eyes': len(detected_eyes)
        }
    
    def detect_in_image(self, image_path, **kwargs):
        """
        Detect faces and eyes in a single image file.
        
        Args:
            image_path (str): Path to image file
            **kwargs: Additional parameters for detect_faces_and_eyes()
            
        Returns:
            dict: Detection results
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        print(f"\nProcessing: {image_path}")
        print(f"Image shape: {image.shape}")
        
        results = self.detect_faces_and_eyes(image, **kwargs)
        print(f"Detected: {results['num_faces']} faces, {results['num_eyes']} eyes")
        
        return results
    
    def detect_in_video(self, video_source=0, scale_factor=1.2, min_neighbors=5,
                       display_fps=True, write_output=False, output_path='output.mp4'):
        """
        Real-time face and eye detection from video or webcam.
        
        Args:
            video_source (int/str): 0 for webcam or path to video file
            scale_factor (float): Cascade scale factor
            min_neighbors (int): Minimum neighbors for detection
            display_fps (bool): Show FPS on output
            write_output (bool): Save output video
            output_path (str): Output video file path
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if requested
        out = None
        if write_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"Output will be saved to: {output_path}")
        
        print("Press 'q' to quit, 's' to save frame")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect faces and eyes
            results = self.detect_faces_and_eyes(
                frame,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors
            )
            
            display_frame = results['annotated_image']
            
            # Add FPS and detection info
            if display_fps:
                fps_text = f"Faces: {results['num_faces']} | Eyes: {results['num_eyes']}"
                cv2.putText(display_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write to output video
            if out:
                out.write(display_frame)
            
            # Display frame
            cv2.imshow('Face & Eye Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"frame_{frame_count}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Frame saved: {filename}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Total frames: {frame_count}")
    
    def detect_and_visualize_image(self, image_path, output_path=None, **kwargs):
        """
        Detect faces/eyes in image and optionally save result.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image (optional)
            **kwargs: Parameters for detection
        """
        results = self.detect_in_image(image_path, **kwargs)
        
        # Display results
        cv2.imshow('Face & Eye Detection', results['annotated_image'])
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save if requested
        if output_path:
            cv2.imwrite(output_path, results['annotated_image'])
            print(f"Annotated image saved to: {output_path}")
        
        return results


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    # Initialize detector
    detector = FaceEyeDetector()
    
    # Example 1: Detect in single image
    # --------------------------------
    # Uncomment and modify path to test with your image
    # results = detector.detect_and_visualize_image(
    #     'photo.jpg',
    #     output_path='detected_faces.jpg',
    #     scale_factor=1.2,    # Adjust: 1.05 (accurate) to 1.4 (fast)
    #     min_neighbors=5      # Adjust: 3-6 (lower = more detections)
    # )
    
    # Example 2: Real-time webcam detection
    # -----------------------------------------------
    # Uncomment to run live detection
    detector.detect_in_video(
        video_source=0,  # 0 for webcam
        scale_factor=1.2,
        min_neighbors=5,
        display_fps=True
    )
    
    # Example 3: Process video file
    # -----------------------------------------------
    # Uncomment to process video file
    # detector.detect_in_video(
    #     video_source='input_video.mp4',
    #     scale_factor=1.2,
    #     min_neighbors=5,
    #     write_output=True,
    #     output_path='output_detected.mp4'
    # )
    
    # Example 4: Test with different parameters (for tuning)
    # -----------------------------------------------
    print("\n" + "="*60)
    print("PARAMETER TUNING GUIDE")
    print("="*60)
    print("""
    scaleFactor: Controls image pyramid scaling
      - 1.05: Very accurate but SLOW (>2x computation)
      - 1.1-1.2: Good balance (RECOMMENDED)
      - 1.3-1.4: FAST but less accurate
    
    minNeighbors: Filters false positives
      - 3-4: RELAXED, more detections (may have false positives)
      - 5-6: STRICT, fewer false positives (may miss some faces)
    
    Quick Tests:
      - Group photo: scaleFactor=1.3, minNeighbors=3 (detect all)
      - Single face: scaleFactor=1.1, minNeighbors=6 (high precision)
      - Rotated face: May not work with default cascade
      - Poor lighting: Consider preprocessing (histogram equalization)
    """)
    
    print("\nTo get started:")
    print("1. Prepare your image/video file")
    print("2. Uncomment an example above")
    print("3. Run: python face_detection_project.py")
