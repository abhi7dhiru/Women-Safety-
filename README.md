import cv2
import numpy as np
from deepface import DeepFace
import threading
from datetime import datetime
import os
import geopy
from geopy.geocoders import Nominatim

# Function to improve brightness, contrast, and sharpen the frame
def enhance_image(frame, alpha=1.3, beta=30, sharpen=False):
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    if sharpen:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        frame = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    return frame

# Function to analyze the frame with DeepFace
def analyze_frame(rgb_frame):
    try:
        return DeepFace.analyze(rgb_frame, actions=['age', 'gender'], enforce_detection=False)
    except Exception as e:
        print(f"DeepFace error: {e}")
        return []

# Get real-time location using geopy
def get_location():
    try:
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.geocode("Jagriti Vihar Tyagi Home")  # You can change this to dynamic fetching if you have GPS coordinates.
        return f"{location.address}" if location else "Jagriti Vihar Tyagi Home"
    except:
        return "Jagriti Vihar Tyagi Home"

# Create a directory to save captured images
save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "Image_Collection", "captured_images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam with error checking
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for threading and processing
results = []
processing_frame = False
frame_count = 0
analyze_interval = 5  # Analyze every 5th frame
location = get_location()

# Function to process the frame in a separate thread
def analyze_and_store_results(rgb_frame, original_frame):
    global results, processing_frame

    analyzed_results = analyze_frame(rgb_frame)

    # Process the detected faces in the frame
    for i, result in enumerate(analyzed_results):
        gender = result.get('gender')
        face_positions = result.get('region', {})

        if face_positions:
            # Ensure valid face positions
            x, y, w, h = face_positions.get('x', 0), face_positions.get('y', 0), face_positions.get('w', 0), face_positions.get('h', 0)
            if w > 0 and h > 0:
                # Overlay the location on the image
                cv2.putText(original_frame, f"Location: {location}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                filename = os.path.join(save_dir, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                
                # Save the current frame with text information
                cv2.imwrite(filename, original_frame)
                print(f"Saved: {filename}")

                # Show the captured image in a separate window
                captured_img = cv2.imread(filename)
                captured_img_resized = cv2.resize(captured_img, (640, 480))  # Resize if needed
                cv2.imshow('Captured Image', captured_img_resized)

    results.clear()
    results.extend(analyzed_results)
    processing_frame = False

# Create a named window to set the size explicitly
cv2.namedWindow('Age and Gender Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Age and Gender Detection', 1280, 720)  # Resize to your desired dimensions

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        frame_count += 1

        # Enhance the image for better detection
        enhanced_frame = enhance_image(frame, alpha=1.3, beta=30, sharpen=True)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        # Only analyze every 5th frame to reduce delay
        if frame_count % analyze_interval == 0 and not processing_frame:
            processing_frame = True
            # Use a separate thread to analyze the frame
            analysis_thread = threading.Thread(target=analyze_and_store_results, args=(rgb_frame, frame.copy()))
            analysis_thread.daemon = True
            analysis_thread.start()

        # Get current date, time, and day of the week
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_day = datetime.now().strftime("%A")

        # If results are available, display them
        if results:
            for result in results:
                age = result.get('age', 'N/A')
                gender = result.get('gender', 'N/A')
                gender_confidence = result.get('gender_confidence', 0)
                face_positions = result.get('region', {})

                if face_positions:
                    x, y, w, h = face_positions['x'], face_positions['y'], face_positions['w'], face_positions['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label_text = f"{gender} ({gender_confidence :.2f}%)"
                    label_text_age = f"Age: {age} years old"
                    cv2.putText(frame, label_text, (50, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, label_text_age, (50, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display location, date, time, and day
        cv2.putText(frame, f"Location: {location}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Date & Time: {current_datetime}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Day: {current_day}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Resize frame for display
        resized_frame = cv2.resize(frame, (1280, 720))

        # Save the frame with overlays (location, date, time, day, etc.)
        save_filename = os.path.join(save_dir, f"full_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(save_filename, frame)
        print(f"Full image saved with overlays: {save_filename}")

        # Display the resulting frame
        cv2.imshow('Age and Gender Detection', resized_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
