# Grip Classification Using YOLOv8

This project implements real-time grip classification using the YOLOv8 object detection model. It is designed for applications in prosthetic hands and robotics, where identifying the type of object being grasped can assist in determining the appropriate grip type.

## Overview

The system uses a webcam to detect common household and tool objects using YOLOv8. Each detected object is then mapped to a predefined grip category (e.g., Pinch, Palmar, Two-Handed) and displayed in real-time. This project is particularly relevant for embedded AI systems in assistive technology.

## Features

- Real-time object detection using YOLOv8
- Grip type classification based on detected object
- Custom mapping of COCO-class objects to functional grip types
- Visual display of object label, confidence score, and grip type
- Live FPS monitoring
- Adjustable for webcam or video file input

## Grip Categories

Objects are mapped to the following grip types:
- **Palmar Wrist Neutral**
- **Pinch**
- **Palmar Wrist Pronated**
- **Precision Grip**
- **Two-Handed Grip**

## Technologies Used

- Python
- OpenCV
- YOLOv8 (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- cvzone
- Math & time modules

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/grip-classification-yolov8.git
cd grip-classification-yolov8
```

2. **Install dependencies:**
```bash
pip install ultralytics opencv-python cvzone
```

3. **Download YOLOv8 weights:**
Place your YOLOv8 model weights (e.g., `yolov8l.pt`) inside a `Yolo-Weights/` directory at the root level.

##  How to Run

```bash
python grip_classification.py
```

Ensure your webcam is connected, or adjust the script to use a video file.

## Example Output

Detected: `bottle`  
Grip Type: `Palmar Wrist Neutral`  
Confidence: `0.87`  
FPS: `25.3`

## File Structure

```
grip-classification-yolov8/
│
├── grip_classification.py         # Main script
├── Yolo-Weights/
│   └── yolov8l.pt                 # Model weights
├── README.md
```

## Customization

You can add or modify the `gripTypes` dictionary in the script to include more classes or remap grips as needed.

## Contributions

Feel free to fork this project and contribute by improving accuracy, expanding object mappings, or optimizing performance.

