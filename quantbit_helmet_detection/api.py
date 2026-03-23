import frappe
import base64
import cv2
import numpy as np
import io
from pathlib import Path
import tempfile
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Color palette & labels
COLORS = {
    "helmet":    (0, 200, 0),       # Green  – helmet detected
    "no_helmet": (0, 0, 220),       # Red    – no helmet
    "person":    (255, 165, 0),     # Orange – person (generic)
    "head":      (0, 165, 255),     # Yellow – head without helmet
}

# Map class names from the pre-trained model to our labels
CLASS_MAP = {
    "helmet":       "helmet",
    "hard hat":     "helmet",
    "hardhat":      "helmet",
    "safety helmet":"helmet",
    "no helmet":    "no_helmet",
    "no-helmet":    "no_helmet",
    "no hardhat":   "no_helmet",
    "person":       "person",
    "head":         "head",
    "worker":       "person",
}

class HelmetDetector:
    """YOLOv8-based construction helmet detector."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.4):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package required. pip install ultralytics")

        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names
        self.stats = {"helmet": 0, "no_helmet": 0, "frames": 0}

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run inference on a single frame. Returns list of detection dicts."""
        results = self.model(frame, conf=self.conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_name = self.class_names[cls_id].lower()
            label    = CLASS_MAP.get(raw_name, raw_name)
            detections.append({
                "label": label,
                "raw_name": raw_name,
                "conf": conf_val,
                "box": (x1, y1, x2, y2),
            })
        return detections

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Annotate frame with bounding boxes and labels."""
        overlay = frame.copy()
        helmet_count    = sum(1 for d in detections if d["label"] == "helmet")
        no_helmet_count = sum(1 for d in detections if d["label"] == "no_helmet")

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            color = COLORS.get(det["label"], (200, 200, 200))
            label_text = f"{det['raw_name']} {det['conf']:.0%}"

            # Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(overlay, label_text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Status banner
        h, w = frame.shape[:2]
        banner_h = 40
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 30, 30), -1)

        status = "✓ All helmets on" if no_helmet_count == 0 and helmet_count > 0 else \
                 "⚠ HELMET MISSING!" if no_helmet_count > 0 else "Scanning..."
        color  = (0, 220, 0) if no_helmet_count == 0 and helmet_count > 0 else \
                 (0, 0, 220) if no_helmet_count > 0 else (200, 200, 200)

        cv2.putText(overlay, f"Helmets: {helmet_count}  No-helmet: {no_helmet_count}  |  {status}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Semi-transparent blend
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

@frappe.whitelist()
def detect_helmet(image_data, confidence=0.4, model="yolov8n.pt"):
    """
    Detect helmets in uploaded image
    Args:
        image_data: Base64 encoded image
        confidence: Confidence threshold (0-1)
        model: YOLO model to use
    Returns:
        dict with detection results and processed image
    """
    try:
        if not YOLO_AVAILABLE:
            return {
                "success": False,
                "error": "YOLO library not available. Please install ultralytics package."
            }

        # Decode base64 image
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "success": False,
                "error": "Could not decode image"
            }

        # Initialize detector
        detector = HelmetDetector(model_path=model, conf=float(confidence))
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw results
        result_frame = detector.draw(frame, detections)
        
        # Convert result back to base64
        _, buffer = cv2.imencode('.jpg', result_frame)
        result_image = base64.b64encode(buffer).decode('utf-8')
        
        # Count results
        helmet_count = sum(1 for d in detections if d["label"] == "helmet")
        no_helmet_count = sum(1 for d in detections if d["label"] == "no_helmet")
        
        return {
            "success": True,
            "detections": detections,
            "result_image": f"data:image/jpeg;base64,{result_image}",
            "helmet_count": helmet_count,
            "no_helmet_count": no_helmet_count,
            "total_detections": len(detections)
        }
        
    except Exception as e:
        frappe.log_error(f"Helmet detection error: {str(e)}", "Helmet Detection")
        return {
            "success": False,
            "error": str(e)
        }

@frappe.whitelist()
def get_camera_devices():
    """
    Get available camera devices
    Returns:
        dict with list of available cameras
    """
    try:
        import cv2
        
        # Test camera devices (typically 0-3)
        available_cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                available_cameras.append({
                    "id": i,
                    "name": f"Camera {i}",
                    "resolution": f"{width}x{height}",
                    "fps": fps if fps > 0 else "Unknown"
                })
                cap.release()
        
        return {
            "success": True,
            "cameras": available_cameras
        }
        
    except Exception as e:
        frappe.log_error(f"Camera detection error: {str(e)}", "Camera Detection")
        return {
            "success": False,
            "error": str(e)
        }
