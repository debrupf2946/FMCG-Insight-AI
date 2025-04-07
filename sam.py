from ultralytics import SAM
import cv2
import cvzone
import time
import math
# Load a model
model = SAM("yolo_weights/sam2_b.pt")


results=model("Flipkart_data/multi_grocery_data/IMG_7667.HEIC",show=True)
cv2.waitKey(0)