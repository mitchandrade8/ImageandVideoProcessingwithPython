import cv2
import time
import numpy as np # Import NumPy for stacking images

# Initialize video capture and the first_frame variable
video = cv2.VideoCapture(0)
first_frame = None

# Let the camera warm up
print("Letting camera warm up...")
for i in range(30):
    check, frame = video.read()
    if not check:
        time.sleep(0.1)
print("Camera ready.")

while True:
    check, frame = video.read()
    if not check:
        break

    # --- 1. PREPARE ALL FOUR IMAGE PANELS ---
    
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # In the first valid loop, set the first_frame
    if first_frame is None:
        first_frame = gray_blur
        continue

    # Calculate the difference and threshold
    delta_frame = cv2.absdiff(first_frame, gray_blur)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # --- 2. CONVERT GRAYSCALE IMAGES TO 3-CHANNEL BGR FOR STACKING ---
    # This makes them compatible with the original color frame.
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    delta_bgr = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

    # --- 3. STACK THE FRAMES INTO A 2x2 GRID ---
    
    # Horizontally stack the top two frames (Color and Grayscale)
    top_row = np.hstack((frame, gray_bgr))
    
    # Horizontally stack the bottom two frames (Delta and Threshold)
    bottom_row = np.hstack((delta_bgr, thresh_bgr))
    
    # Vertically stack the top and bottom rows to create the dashboard
    dashboard = np.vstack((top_row, bottom_row))

    # --- 4. DISPLAY THE FINAL DASHBOARD ---
    
    # Display the single, combined frame
    cv2.imshow("Motion Detector Dashboard", dashboard)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()