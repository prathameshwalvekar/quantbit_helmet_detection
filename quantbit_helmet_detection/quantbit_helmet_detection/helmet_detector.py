"""
Construction Helmet Detection using YOLOv8
==========================================
Detects whether workers are wearing safety helmets (hard hats) in real-time.
Supports: webcam, video files, and static images.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import argparse
import sys

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")


# ─────────────────────────────────────────────
# Color palette & labels
# ─────────────────────────────────────────────
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
        """
        Args:
            model_path: Path to YOLOv8 weights.
                        Use a custom helmet-trained model for best results.
                        Falls back to general YOLOv8 (detects 'person' only).
            conf:       Confidence threshold (0–1).
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics package required. pip install ultralytics")

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf
        self.class_names = self.model.names  # {id: name}
        print(f"[INFO] Model loaded. Classes: {list(self.class_names.values())[:10]}...")
        self.stats = {"helmet": 0, "no_helmet": 0, "frames": 0}

    # ── Core detection ────────────────────────────────────────────────────────

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

    # ── Drawing ───────────────────────────────────────────────────────────────

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

    # ── Input modes ───────────────────────────────────────────────────────────

    def process_image(self, path: str, save: bool = True) -> str:
        """Detect helmets in a static image."""
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        dets   = self.detect(img)
        result = self.draw(img, dets)
        out    = Path(path).stem + "_detected.jpg"
        if save:
            cv2.imwrite(out, result)
            print(f"[INFO] Saved: {out}")
        self._print_summary(dets)
        return out

    def process_video(self, path: str, save: bool = True) -> str:
        """Detect helmets in a video file."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        fps  = cap.get(cv2.CAP_PROP_FPS) or 25
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(path).stem + "_detected.mp4"
        writer   = None
        if save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        print("[INFO] Processing video... (press Q to stop preview)")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets   = self.detect(frame)
            result = self.draw(frame, dets)
            if writer:
                writer.write(result)
            cv2.imshow("Helmet Detection", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if save:
            print(f"[INFO] Saved: {out_path}")
        return out_path

    def process_webcam(self, cam_id: int = 0):
        """Real-time helmet detection from webcam."""
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {cam_id}")

        print("[INFO] Webcam started. Press Q to quit.")
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets   = self.detect(frame)
            result = self.draw(frame, dets)

            # FPS counter
            now  = time.time()
            fps  = 1 / (now - prev_time + 1e-9)
            prev_time = now
            cv2.putText(result, f"FPS: {fps:.1f}", (result.shape[1] - 110, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            cv2.imshow("Helmet Detection – Live", result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_summary(self, dets: list[dict]):
        counts = {}
        for d in dets:
            counts[d["label"]] = counts.get(d["label"], 0) + 1
        print("[RESULT]", counts if counts else "No detections")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Construction Helmet Detector (YOLOv8)")
    parser.add_argument("--source",  default="0",
                        help="Input source: 0=webcam, path to image/video")
    parser.add_argument("--model",   default="yolov8n.pt",
                        help="YOLOv8 weights (default: yolov8n.pt)")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="Confidence threshold (default: 0.4)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save output file")
    args = parser.parse_args()

    detector = HelmetDetector(model_path=args.model, conf=args.conf)
    src = args.source

    if src.isdigit():
        detector.process_webcam(int(src))
    elif Path(src).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        detector.process_image(src, save=not args.no_save)
    elif Path(src).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
        detector.process_video(src, save=not args.no_save)
    else:
        print(f"[ERROR] Unsupported source: {src}")
        sys.exit(1)


if __name__ == "__main__":
    main()
