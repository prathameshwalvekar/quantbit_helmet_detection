# 🦺 Construction Helmet Detection (YOLOv8 + Python)

Detect whether construction workers are wearing safety helmets in real-time using **YOLOv8**.

---

##  Project Structure

```
helmet_detection/
├── helmet_detector.py      # Main detector – webcam / video / image
├── train_helmet_model.py   # Fine-tune YOLOv8 on custom dataset
├── demo_mock.py            # Mock demo (no model/GPU needed)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install ultralytics opencv-python
```

### 2. Run the mock demo (no GPU needed)
```bash
python demo_mock.py
```

### 3. Run on webcam (requires trained model)
```bash
python helmet_detector.py --source 0 --model yolov8n.pt
```

### 4. Run on an image
```bash
python helmet_detector.py --source workers.jpg --model best.pt --conf 0.4
```

### 5. Run on a video
```bash
python helmet_detector.py --source site_footage.mp4 --model best.pt
```

---

## 🏋️ Training Your Own Model

### Step 1 – Get a dataset
Use Roboflow to download a pre-labelled helmet dataset:
```python
pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-universe-projects").project("hard-hat-workers")
dataset = project.version(2).download("yolov8")
```

Or use **Kaggle**: search "safety helmet detection dataset"

### Step 2 – Train
```bash
python train_helmet_model.py --epochs 100 --model yolov8s.pt --batch 16
```

Best weights saved to `runs/helmet/exp/weights/best.pt`

### Step 3 – Validate
```bash
python train_helmet_model.py --validate
```

### Step 4 – Export to ONNX (for deployment)
```bash
python train_helmet_model.py --export onnx
```

---

## 🎯 Model Options

| Model       | Speed   | Accuracy | Use Case               |
|-------------|---------|----------|------------------------|
| yolov8n.pt  | Fastest | Good     | Edge devices / webcam  |
| yolov8s.pt  | Fast    | Better   | Laptop GPU             |
| yolov8m.pt  | Medium  | Great    | Server / production    |
| yolov8l.pt  | Slow    | Best     | High-accuracy required |

---

## 📊 Expected Classes

| Class ID | Name       | Description               |
|----------|------------|---------------------------|
| 0        | helmet     | Worker wearing hard hat ✅ |
| 1        | no_helmet  | Worker without hard hat ❌ |

---

## 🚀 CLI Reference

```
python helmet_detector.py
  --source   0 | image.jpg | video.mp4   (default: 0 = webcam)
  --model    yolov8n.pt | best.pt        (default: yolov8n.pt)
  --conf     0.0–1.0                     (default: 0.4)
  --no-save  Don't save output file
```

---

## 💡 Tips

- For **best accuracy**, use a model specifically fine-tuned on helmet data
- Pre-trained `yolov8n.pt` detects `person` only (no helmet class)
- Use `--conf 0.3` in poor lighting conditions
- Export to ONNX for deployment on Raspberry Pi or Jetson Nano
