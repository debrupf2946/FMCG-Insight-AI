import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics import RTDETR
import cvzone
import time
import math
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the trained YOLO and RTDETR models
model_yolo = YOLO('yolo_weights/grocery_yolov8_retrained.pt')
model_detr = RTDETR("yolo_weights/rtdetr-l.pt")

# Load the shelf-life prediction model
model_shelf_life = load_model('yolo_weights/shelf_life_generalized_model.h5')  # Load your saved Keras model

# Define the categories (shelf life classes)
class_found = {0: '1-5 days', 1: '5-10 days', 2: '10-15 days', 3: 'Expired'}

# Set target image size for your shelf-life model input
target_size = (224, 224)

# Process the video file
cap = cv2.VideoCapture('Flipkart_data/WhatsApp Video 2024-10-20 at 9.44.10 PM.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or failed to capture frame.")
        break

    # YOLO object detection on the frame
    results = model_yolo(frame)

    for result in results:
        # Extract bounding boxes, confidence scores, and class names (e.g., fruits)
        boxes = result.boxes.xyxy  # Get the box coordinates
        class_ids = result.boxes.cls  # Get the class IDs of detected objects
        class_names = result.names  # Get the names of the detected objects

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer values
            detected_obj = frame[y1:y2, x1:x2]  # Crop the detected object from the frame

            # Preprocess the detected object for shelf-life prediction (resize and normalize)
            obj_resized = cv2.resize(detected_obj, target_size)
            obj_array = img_to_array(obj_resized) / 255.0  # Normalize pixel values
            obj_array = np.expand_dims(obj_array, axis=0)  # Add batch dimension

            # Predict shelf life for the detected object
            predictions = model_shelf_life.predict(obj_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = class_found[predicted_class]

            # Fetch the correct class name based on class_id
            class_id = int(class_ids[i])
            class_name = class_names[class_id]

            # Debugging print to verify class name
            print(f"Detected object: {class_name}")

            # Draw bounding box, display object name and shelf life prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(
                frame, f"{class_name}: {predicted_label}",  # Display fruit and shelf life
                (x1, y1 - 10),  # Position (x, y) above the bounding box
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                0.6,  # Font scale (size)
                (255, 0, 0),  # Color (Blue in BGR)
                2  # Thickness of the text
            )

    # Show the frame with detection and shelf-life predictions
    cv2.imshow('Fruit Detection and Shelf-Life Prediction', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

