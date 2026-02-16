
"""
YOLOv8 Real-Time Object Detection System
Supports: Webcam, Video Files, Images, and Sample Dataset
No manual modification needed - just run!

FIXED VERSION - Resolves common issues:
- Webcam access errors
- cv2.imshow not responding
- Platform compatibility (Windows/Mac/Linux)
- Better error handling and diagnostics
"""

from ultralytics import YOLO
import cv2
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import platform

# =============================================================================
# CONFIGURATION - Modify these if needed
# =============================================================================

class Config:
    # Model selection: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
    MODEL = "yolov8n.pt"  # nano = fastest for real-time
    
    # Detection mode: 'webcam', 'video', 'image', 'dataset'
    MODE = "webcam"
    
    # Video/Image paths (used if MODE is 'video' or 'image')
    VIDEO_PATH = "sample_video.mp4"
    IMAGE_PATH = "sample_image.jpg"
    
    # Webcam settings
    WEBCAM_ID = 0  # 0 = default camera, 1 = second camera, etc.
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5  # 0.0 to 1.0
    IOU_THRESHOLD = 0.45  # Non-maximum suppression threshold
    
    # Display settings
    SHOW_LABELS = True
    SHOW_CONFIDENCE = True
    LINE_THICKNESS = 2
    
    # Output settings
    SAVE_OUTPUT = False  # Save processed video/image
    OUTPUT_DIR = "output"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def download_sample_dataset():
    """Download COCO8 sample dataset automatically"""
    dataset_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
    dataset_dir = Path("datasets/coco8")
    
    if dataset_dir.exists():
        print(f"✓ Dataset already exists at {dataset_dir}")
        return dataset_dir
    
    print("📥 Downloading COCO8 sample dataset...")
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    
    zip_path = "datasets/coco8.zip"
    urllib.request.urlretrieve(dataset_url, zip_path)
    
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets")
    
    os.remove(zip_path)
    print(f"✓ Dataset ready at {dataset_dir}")
    return dataset_dir

def create_sample_video():
    """Create a simple test video if no video file exists"""
    if os.path.exists(Config.VIDEO_PATH):
        return Config.VIDEO_PATH
    
    print("📹 No video file found. Use your own video or download from:")
    print("   https://www.pexels.com/search/videos/people/")
    return None

def load_model(model_path):
    """Load YOLOv8 model (downloads automatically if needed)"""
    print(f"🔄 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    return model

# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_webcam(model):
    """Real-time detection on webcam feed"""
    print(f"🎥 Starting webcam detection (Camera {Config.WEBCAM_ID})")
    print("   Press 'q' to quit, 's' to save screenshot")
    
    # Try different backends for better compatibility
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_AVFOUNDATION]
    cap = None
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(Config.WEBCAM_ID, backend)
            if cap.isOpened():
                print(f"   ✓ Webcam opened successfully")
                break
        except:
            continue
    
    if cap is None or not cap.isOpened():
        print("❌ Error: Cannot open webcam")
        print("   Troubleshooting:")
        print("   1. Check if camera is connected and not used by another app")
        print("   2. Try changing Config.WEBCAM_ID to 1 or 2")
        print("   3. Run: pip install --upgrade opencv-python")
        return
    
    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    
    # Create window before loop (fixes Mac/Linux issues)
    cv2.namedWindow("YOLOv8 Webcam Detection", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Verify frame is valid
        if frame is None or frame.size == 0:
            continue
        
        # Run YOLOv8 detection
        results = model(
            frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            verbose=False
        )
        
        # Annotate frame with detections
        annotated_frame = results[0].plot(
            line_width=Config.LINE_THICKNESS,
            labels=Config.SHOW_LABELS,
            conf=Config.SHOW_CONFIDENCE
        )
        
        # Display FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Show frame (with error handling)
        try:
            cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)
        except cv2.error as e:
            print(f"⚠️  Display error: {e}")
            print("   Continuing without display...")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    # Extra cleanup for Mac/Linux
    cv2.waitKey(1)
    print(f"✓ Processed {frame_count} frames")

def detect_video(model, video_path):
    """Detection on video file"""
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"📹 Processing video: {video_path}")
    print("   Press 'q' to quit early")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video file")
        print("   Check if file format is supported (MP4, AVI, MOV)")
        return
    
    # Create window
    cv2.namedWindow("YOLOv8 Video Detection", cv2.WINDOW_NORMAL)
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Prepare output video writer if saving
    if Config.SAVE_OUTPUT:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            Config.OUTPUT_DIR,
            f"detected_{os.path.basename(video_path)}"
        )
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(
            frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            verbose=False
        )
        
        annotated_frame = results[0].plot(
            line_width=Config.LINE_THICKNESS,
            labels=Config.SHOW_LABELS,
            conf=Config.SHOW_CONFIDENCE
        )
        
        # Show progress
        progress = (frame_count / total_frames) * 100
        cv2.putText(
            annotated_frame,
            f"Progress: {progress:.1f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        try:
            cv2.imshow("YOLOv8 Video Detection", annotated_frame)
        except cv2.error:
            pass  # Continue processing even if display fails
        
        if Config.SAVE_OUTPUT:
            out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    cap.release()
    if Config.SAVE_OUTPUT:
        out.release()
        print(f"✓ Output saved to: {output_path}")
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Extra cleanup
    print(f"✓ Processed {frame_count}/{total_frames} frames")

def detect_image(model, image_path):
    """Detection on single image"""
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        print(f"   Create or download a test image first")
        print(f"   Example: https://ultralytics.com/images/bus.jpg")
        return
    
    print(f"🖼️  Processing image: {image_path}")
    
    # Run detection
    results = model(
        image_path,
        conf=Config.CONFIDENCE_THRESHOLD,
        iou=Config.IOU_THRESHOLD
    )
    
    # Verify results
    if not results or len(results) == 0:
        print("⚠️  No detections or processing error")
        return
    
    # Get annotated image
    annotated_img = results[0].plot(
        line_width=Config.LINE_THICKNESS,
        labels=Config.SHOW_LABELS,
        conf=Config.SHOW_CONFIDENCE
    )
    
    # Save output
    if Config.SAVE_OUTPUT:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(
            Config.OUTPUT_DIR,
            f"detected_{os.path.basename(image_path)}"
        )
        cv2.imwrite(output_path, annotated_img)
        print(f"✓ Output saved to: {output_path}")
    
    # Display result
    try:
        cv2.namedWindow("YOLOv8 Image Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("YOLOv8 Image Detection", annotated_img)
        print("Press any key to close...")
        cv2.waitKey(0)
    except cv2.error as e:
        print(f"⚠️  Cannot display image: {e}")
        print("   Image saved but display failed (common in some environments)")
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    
    # Print detection results
    print("\nDetected objects:")
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[cls]
            print(f"  - {name}: {conf:.2f}")

def detect_dataset(model):
    """Run detection on sample dataset"""
    dataset_dir = download_sample_dataset()
    
    # Get all images from dataset
    image_dir = dataset_dir / "images" / "val"
    
    if not image_dir.exists():
        print(f"❌ Dataset images not found at {image_dir}")
        return
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No images found in dataset")
        return
    
    print(f"📊 Processing {len(image_files)} images from dataset")
    print("   Press any key to see next image, 'q' to quit")
    
    # Create window once
    cv2.namedWindow("YOLOv8 Dataset Detection", cv2.WINDOW_NORMAL)
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {img_path.name}")
        
        # Run detection
        results = model(
            str(img_path),
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD
        )
        
        annotated_img = results[0].plot(
            line_width=Config.LINE_THICKNESS,
            labels=Config.SHOW_LABELS,
            conf=Config.SHOW_CONFIDENCE
        )
        
        # Print detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names[cls]
                print(f"  ✓ {name}: {conf:.2f}")
        
        # Display image
        try:
            cv2.imshow("YOLOv8 Dataset Detection", annotated_img)
            key = cv2.waitKey(0)
            
            if key == ord('q') or key == 27:  # q or ESC
                break
        except cv2.error:
            print("   Display error, continuing...")
            continue
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print(f"\n✓ Dataset processing complete")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("   YOLOv8 Real-Time Object Detection System")
    print("=" * 70)
    print(f"   Platform: {platform.system()} | Python: {sys.version.split()[0]}")
    print("=" * 70)
    
    # Load model
    try:
        model = load_model(Config.MODEL)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("   Try: pip install --upgrade ultralytics")
        return
    
    # Run detection based on selected mode
    print(f"\n🎯 Mode: {Config.MODE.upper()}")
    
    if Config.MODE == "webcam":
        detect_webcam(model)
    
    elif Config.MODE == "video":
        video_path = create_sample_video()
        if video_path:
            detect_video(model, video_path)
    
    elif Config.MODE == "image":
        detect_image(model, Config.IMAGE_PATH)
    
    elif Config.MODE == "dataset":
        detect_dataset(model)
    
    else:
        print(f"❌ Invalid mode: {Config.MODE}")
        print("   Valid modes: webcam, video, image, dataset")
    
    print("\n✓ Detection complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
                