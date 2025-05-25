from ultralytics import YOLO
import cv2
import cvzone
import math
import time

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush", "calculator", "glasses", "pen"]

# Define grip types
gripTypes = {
    # Palmar Wrist Neutral
    "bottle": "Palmar Wrist Neutral",
    "cup": "Palmar Wrist Neutral",
    "vase": "Palmar Wrist Neutral",
"bowl": "Palmar Wrist Neutral",

    # Pinch
    "wine glass": "Pinch",
    "banana": "Pinch",
    "apple": "Pinch",
    "orange": "Pinch",
    "remote": "Pinch",
    "cell phone": "Pinch",
    "book": "Pinch",
    "knife": "Pinch",
    "fork": "Pinch",
    "spoon": "Pinch",
    "toothbrush": "Pinch",
    "calculator": "Pinch",
    "glasses": "Pinch",
    "pen": "Pinch",

    # Palmar Wrist Pronated
    "handbag": "Palmar Wrist Pronated",
    "suitcase": "Palmar Wrist Pronated",
    "skateboard": "Palmar Wrist Pronated",
    "surfboard": "Palmar Wrist Pronated",
    "tennis racket": "Palmar Wrist Pronated",
    "frisbee": "Palmar Wrist Pronated",
    "baseball bat": "Palmar Wrist Pronated",
    "pottedplant": "Palmar Wrist Pronated",
    "hair drier": "Palmar Wrist Pronated",
    "laptop": "Palmar Wrist Pronated",

    # Precision Grip
    "scissors": "Precision Grip",
    "mouse": "Precision Grip",
    "clock": "Precision Grip",
    "teddy bear": "Precision Grip",

    # Two-Handed Grip
    "backpack": "Two-Handed Grip",
    "tvmonitor": "Two-Handed Grip",
    "chair": "Two-Handed Grip",
    "sofa": "Two-Handed Grip",
    "diningtable": "Two-Handed Grip",
    "bed": "Two-Handed Grip",
    "microwave": "Two-Handed Grip",
    "oven": "Two-Handed Grip",
    "refrigerator": "Two-Handed Grip",
    "bench": "Two-Handed Grip"
}

prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            objName = classNames[cls]
            gripType = gripTypes.get(objName, "Unknown Grip")  # Default to "Unknown Grip" if not found

            # Debug output
            print(f'Detected: {objName} with confidence {conf}')
            print(f'Grip Type: {gripType}')

            cvzone.putTextRect(img, f'{objName} {conf} - {gripType}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f'FPS: {fps}')

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()