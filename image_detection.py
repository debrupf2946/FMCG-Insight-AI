import os
import cv2
import time
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model (replace with your custom-trained model if needed)
model = YOLO("yolo_weights/fmcg_v8_10.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=2, nn_budget=20)

# Directory to store images sent to Qwen
SAVE_DIR = "qwen_images_1"
os.makedirs(SAVE_DIR, exist_ok=True)

# Cache to track which product IDs were sent recently
qwen_sent_cache = {}

def parse_args():
    """Parse command-line arguments to choose input source."""
    parser = argparse.ArgumentParser(description="YOLOv8 with DeepSORT and Qwen integration.")
    parser.add_argument("--video", type=str, help="Path to input video file (leave empty for webcam).")
    return parser.parse_args()

def detect_products(frame):
    """Use YOLOv8 to detect products and return bounding boxes."""
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            conf = box.conf.item()
            if conf > 0.3:  # Confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(([x1, y1, x2, y2], conf, 'product'))

    return detections

def track_products(frame, detections):
    """Track products using DeepSORT and return tracked product IDs with bounding boxes."""
    tracks = tracker.update_tracks(detections, frame=frame)
    tracked_products = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()  # Left-Top-Right-Bottom format
        tracked_products.append((track_id, bbox))

    return tracked_products

def save_image(frame, bbox, product_id):
    """Save the cropped product image with a unique filename."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_frame = frame[y1:y2, x1:x2]

    # Generate a unique filename using product ID and timestamp
    timestamp = int(time.time())
    filename = f"{product_id}_{timestamp}.jpg"
    file_path = os.path.join(SAVE_DIR, filename)

    # Save the cropped product image
    cv2.imwrite(file_path, cropped_frame)
    print(f"Saved image: {file_path}")

    return file_path

def send_to_qwen(product_id, frame, bbox, cache_duration=30):
    """Send product frame to Qwen if not recently sent."""
    if product_id in qwen_sent_cache:
        last_sent = qwen_sent_cache[product_id]
        if time.time() - last_sent < cache_duration:
            print(f"Skipping Qwen call for product ID {product_id}")
            return

    # Save the image and (simulate) send it to Qwen
    file_path = save_image(frame, bbox, product_id)

    # Simulate sending to Qwen VLM (replace with actual API call)
    print(f"Sending product ID {product_id} to Qwen...")
    # send_to_qwen_api(file_path)  # Uncomment for real API call

    # Update cache with the current timestamp
    qwen_sent_cache[product_id] = time.time()

def process_video(input_source):
    """Process input from either webcam or video file."""
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print(f"Error: Unable to open video source {input_source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or no frames available.")
            break

        # Step 1: Detect products using YOLOv8
        detections = detect_products(frame)

        # Step 2: Track products and get unique IDs
        tracked_products = track_products(frame, detections)

        # Step 3: Send only necessary frames to Qwen
        for product_id, bbox in tracked_products:
            send_to_qwen(product_id, frame, bbox)

        # Display the video feed
        cv2.imshow("Product Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to handle input source and start processing."""
    args = parse_args()
    input_source = args.video if args.video else 0  # Use video file or webcam

    print(f"Using input source: {input_source}")
    process_video(input_source)

if __name__ == "__main__":
    main()

