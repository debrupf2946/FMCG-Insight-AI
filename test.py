# from ultralytics import YOLO
# import cv2
# #
# # Load the YOLO model
# model = YOLO('yolo_weights/fmcg_v8_10.pt')
#
# # Open video capture
# video_path = "Flipkart_data/IMG_7651.MOV"
# cap = cv2.VideoCapture(video_path)
#
# object_count = {}
#
class_names= ['Bisconni Chocolate Chip Cookies 46.8gm', 'Coca Cola Can 250ml', 'Colgate Maximum Cavity Protection 75gm', 'Fanta 500ml', 'Fresher Guava Nectar 500ml', 'Maggi Noodles', 'Islamabad Tea 238gm', 'Kolson Slanty Jalapeno 18gm', 'Kurkure Chutney Chaska 62gm', 'LU Candi Biscuit 60gm', 'LU Oreo Biscuit 19gm', 'LU Prince Biscuit 55.2gm', 'Lays Masala 34gm', 'Lays Wavy Mexican Chili 34gm', 'Lifebuoy Total Protect Soap 96gm', 'Cinthol Soap', 'Meezan Ultra Rich Tea 190gm', 'Peek Freans Sooper Biscuit 13.2gm', 'Safeguard Bar Soap Pure White 175gm', 'Pipo Popcorn', 'Sunsilk Shampoo Soft - Smooth 160ml', 'Super Crisp BBQ 30gm', 'Supreme Tea 95gm', 'Tapal Danedar 95gm', 'Vaseline Healthy White Lotion 100ml','chicken noodle soup','soap','buiscuit','pouch','box','short pouch','sachet','tetrapack','bottle','can','packet','small bottle','small tub']
#
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break  # Stop when the video ends
#
#     # Run YOLO detection on the current frame
#     results = model(frame)
#
#     # Parse results to count objects grouped by class
#     for result in results:
#         boxes = result.boxes  # Detected bounding boxes
#         for box in boxes:
#             class_id = int(box.cls[0])  # Class ID of the detected object
#
#             # Get class name from class ID
#             class_name = class_names[class_id] if class_id < len(class_names) else 'unknown_class'
#
#             # Create a key to group objects by class
#             key = class_name
#
#             # Count occurrences of each class
#             if key in object_count:
#                 object_count[key] += 1
#             else:
#                 object_count[key] = 1
#
#     # Manually draw bounding boxes without labels
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates of the bounding box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw the bounding box
#
#     # Prepare text for display in the top-left corner
#     start_y = 50  # Adjusted for larger text
#     for class_name, count in object_count.items():
#         # Draw class name
#         text_class = f"Class: {class_name}"
#         cv2.putText(frame, text_class, (10, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 0, 0), 3)
#
#         # Draw count below the class name
#         text_count = f"Count: {count}"
#         cv2.putText(frame, text_count, (10, start_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 0, 0), 3)
#
#         start_y += 60  # Adjust line height for next item (30 for class and 30 for count)
#
#     # Show the frame with object counts in the corner
#     cv2.imshow("YOLO Detection", frame)
#
#     # Break if 'q' key is pressed
#     if cv2.waitKey(0) & 0xFF == ord('q'):
#         break
#
# # # Release resources
# # cap.release()
# # cv2.destroyAllWindows()
# ----------------------------------------------------------
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
from PIL import Image  # Importing Pillow for image handling

# Load the YOLO model
model = YOLO('yolo_weights/fmcg_v8_10.pt')

# Assuming model.names contains class names
# class_names = model.names  # Get class names from the model

# Streamlit app title
st.title("YOLO Object Detection Count by Class")

# Upload file (video or image)
uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "mov", "jpeg", "jpg", "heic"])

if uploaded_file is not None:
    # Check if the uploaded file is an image or a video
    if uploaded_file.type in ["video/mp4", "video/quicktime"]:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.read())
            video_path = temp_file.name

        # Dictionary to count the number of objects by class
        object_count = {}

        # Open video capture
        cap = cv2.VideoCapture(video_path)

        # Process the video frame by frame
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break  # Stop when the video ends

            # Run YOLO detection on the current frame
            results = model(frame)

            # Parse results to count objects grouped by class and draw bounding boxes
            for result in results:
                boxes = result.boxes  # Detected bounding boxes
                for box in boxes:
                    class_id = int(box.cls[0])  # Class ID of the detected object
                    class_name = class_names[class_id] if class_id < len(class_names) else 'Bebnica Desseret'

                    # Draw bounding box on the frame
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw rectangle

                    # Count occurrences of each class
                    if class_name in object_count:
                        object_count[class_name] += 1
                    else:
                        object_count[class_name] = 1

        cap.release()

        # Display video with bounding boxes
        st.subheader("Uploaded Video")
        st.video(video_path)

    else:
        # For images (JPEG, JPG, HEIC)
        # Read the image file
        image = Image.open(uploaded_file)

        # Convert image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run YOLO detection on the image
        results = model(image_cv)

        # Dictionary to count the number of objects by class
        object_count = {}

        # Parse results to count objects grouped by class and draw bounding boxes
        for result in results:
            boxes = result.boxes  # Detected bounding boxes
            for box in boxes:
                class_id = int(box.cls[0])  # Class ID of the detected object
                class_name = class_names[class_id] if class_id < len(class_names) else 'unknown_class'

                # Draw bounding box on the image
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (255, 0, 0), 5)  # Draw rectangle

                # Count occurrences of each class
                if class_name in object_count:
                    object_count[class_name] += 1
                else:
                    object_count[class_name] = 1

        # Display image with bounding boxes
        st.subheader("Uploaded Image")
        st.image(image_cv, caption='Uploaded Image', use_column_width=True)

    # Display results in a dataframe
    st.subheader("Detected Object Counts by Class")
    counts_df = {"Class": list(object_count.keys()), "Count": list(object_count.values()),"Description":"na"}
    st.table(counts_df)

    # Cleanup temporary file (for video)
    if uploaded_file.type in ["video/mp4", "video/quicktime"]:
        os.remove(video_path)

# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import tempfile
# from PIL import Image
# import pandas as pd  # Import pandas for DataFrame handling
#
# # Load the YOLO model
# model = YOLO('yolo_weights/fmcg_v8_10.pt')  # Update with your model path
#
# # Define class names
# class_names = [
#     'Bisconni Chocolate Chip Cookies 46.8gm', 'Coca Cola Can 250ml', 'Colgate Maximum Cavity Protection 75gm',
#     'Fanta 500ml', 'Fresher Guava Nectar 500ml', 'Maggi Noodles', 'Islamabad Tea 238gm',
#     'Kolson Slanty Jalapeno 18gm', 'Kurkure Chutney Chaska 62gm', 'LU Candi Biscuit 60gm',
#     'LU Oreo Biscuit 19gm', 'LU Prince Biscuit 55.2gm', 'Lays Masala 34gm', 'Lays Wavy Mexican Chili 34gm',
#     'Lifebuoy Total Protect Soap 96gm', 'Lipton Yellow Label Tea 95gm', 'Meezan Ultra Rich Tea 190gm',
#     'Peek Freans Sooper Biscuit 13.2gm', 'Safeguard Bar Soap Pure White 175gm', 'Shezan Apple 250ml',
#     'Sunsilk Shampoo Soft - Smooth 160ml', 'Super Crisp BBQ 30gm', 'Supreme Tea 95gm', 'Tapal Danedar 95gm',
#     'Vaseline Healthy White Lotion 100ml', 'chicken noodle soup', 'soap', 'buiscuit', 'pouch', 'box',
#     'short pouch', 'sachet', 'tetrapack', 'bottle', 'can', 'packet', 'small bottle', 'small tub'
# ]
#
# # Streamlit app title
# st.title("YOLO Object Detection Count by Class")
#
# # Upload file (video or image)
# uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "mov", "jpeg", "jpg", "heic"])
#
# if uploaded_file is not None:
#     # Check if the uploaded file is a video or an image
#     if uploaded_file.type in ["video/mp4", "video/quicktime"]:
#         # Save the uploaded video to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
#             temp_file.write(uploaded_file.read())
#             video_path = temp_file.name
#
#         # Open video capture
#         cap = cv2.VideoCapture(video_path)
#
#         # Dictionary to count the number of objects by class
#         object_count = {}
#
#         # Process the video frame by frame
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break  # Stop when the video ends
#
#             # Run YOLO detection on the current frame
#             results = model(frame)
#
#             # Reset object_count for each frame
#             object_count.clear()
#
#             # Parse results to count objects grouped by class
#             for result in results:
#                 boxes = result.boxes  # Detected bounding boxes
#                 for box in boxes:
#                     class_id = int(box.cls[0])  # Class ID of the detected object
#
#                     # Get class name from class ID
#                     class_name = class_names[class_id] if class_id < len(class_names) else 'unknown_class'
#
#                     # Count occurrences of each class
#                     if class_name in object_count:
#                         object_count[class_name] += 1
#                     else:
#                         object_count[class_name] = 1
#
#             # Display video
#             st.subheader("Uploaded Video")
#             st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
#
#             # Display results in a dataframe
#             st.subheader("Detected Object Counts by Class")
#             # Create a list of detected classes and their counts
#             detected_classes = [(class_name, count) for class_name, count in object_count.items() if count > 0]
#
#             # Create a DataFrame from the detected classes
#             counts_df = pd.DataFrame(detected_classes, columns=["Class", "Count"])
#
#             # Display the DataFrame
#             st.table(counts_df)
#
#         cap.release()
#
#     else:
#         # For images (JPEG, JPG, HEIC)
#         # Read the image file
#         image = Image.open(uploaded_file)
#
#         # Convert image to OpenCV format
#         image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#         # Run YOLO detection on the image
#         results = model(image_cv)
#
#         # Dictionary to count the number of objects by class
#         object_count = {}
#
#         # Parse results to count objects grouped by class
#         for result in results:
#             boxes = result.boxes  # Detected bounding boxes
#             for box in boxes:
#                 class_id = int(box.cls[0])  # Class ID of the detected object
#
#                 # Get class name from class ID
#                 class_name = class_names[class_id] if class_id < len(class_names) else 'unknown_class'
#
#                 # Count occurrences of each class
#                 if class_name in object_count:
#                     object_count[class_name] += 1
#                 else:
#                     object_count[class_name] = 1
#
#         # Display image
#         st.subheader("Uploaded Image")
#         st.image(image, caption='Uploaded Image', use_column_width=True)
#
#         # Display results in a dataframe
#         st.subheader("Detected Object Counts by Class")
#         detected_classes = [(class_name, count) for class_name, count in object_count.items() if count > 0]
#         counts_df = pd.DataFrame(detected_classes, columns=["Class", "Count"])
#         st.table(counts_df)
#
#         # Cleanup temporary file (for video)
#         # if uploaded_file.type in ["video/mp4", "video/quicktime"]:
#         #     os.remove(video_path)
#
#
#         # Cleanup temporary file (for video)
#         # if uploaded_file.type in ["video/mp4", "video/quicktime"]:
#         #     os.remove(video_path)






