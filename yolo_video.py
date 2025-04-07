from ultralytics import YOLO
from ultralytics import RTDETR
import cv2
import cvzone
import time
import math

model=YOLO('yolo_weights/grocery_yolov8_retrained.pt')
model_detr = RTDETR("yolo_weights/rtdetr-l.pt")

results=model(f"Flipkart_data/WhatsApp Video 2024-10-20 at 9.21.12 PM.mp4",show=True)
cv2.waitKey(0)