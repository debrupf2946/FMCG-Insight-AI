from ultralytics import RTDETR
import cv2
import cvzone
import time
import math

# Load a COCO-pretrained RT-DETR model
model = RTDETR("yolo_weights/rtdetr-l.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set video dimensions
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# COCO class names (same as you provided)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

prev_frame_time = 0

while True:
    # Capture frame-by-frame
    success, img = cap.read()

    # Check if frame is captured successfully
    if not success:
        print("Error: Unable to capture video")
        break

    # Get current time for FPS calculation
    new_frame_time = time.time()

    # Perform detection using RT-DETR
    results = model(img, stream=True)

    # Process the results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            # Draw bounding box with corner rectangle style
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Extract confidence and class
            conf = math.ceil((box.conf[0] * 100)) / 100  # Round confidence to two decimal places
            cls = int(box.cls[0])

            # Display class name and confidence
            label = f'{classNames[cls]} {conf}'
            cvzone.putTextRect(img, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Calculate and display FPS
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cvzone.putTextRect(img, f'FPS: {int(fps)}', (20, 50), scale=2, thickness=2, offset=10)

    # Show the result in a window
    cv2.imshow("RT-DETR Object Detection", img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
