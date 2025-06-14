import cv2
import time
from datetime import datetime
import numpy as np
import pandas as pd # Import Pandas for data handling

# --- INITIALIZATION ---
first_frame = None
status_list = [None, None] # Initialize with two None items to prevent index errors
times = [] # A list to store the start times of motion
df = pd.DataFrame(columns=["Start", "End"]) # Create an empty DataFrame

video = cv2.VideoCapture(0)

print("Letting camera warm up...")
time.sleep(2) # A simple sleep is often enough for warmup
print("Camera ready.")

# --- MAIN LOOP ---
while True:
    check, frame = video.read()
    status = 0 # Status is 0 for no motion, 1 for motion
    if not check:
        break

    # --- IMAGE PREPARATION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    # --- MOTION CALCULATION ---
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    # Dilate the thresholded image to fill in holes
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # --- OBJECT DETECTION & LOGGING ---
    # Find contours of moving objects
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        # If the contour is too small, ignore it (noise)
        if cv2.contourArea(contour) < 10000:
            continue
        
        # A significant contour was found, so motion is detected
        status = 1

        # Draw a green bounding box around the moving object on the original color frame
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # --- RECORD MOTION START & END TIMES ---
    status_list.append(status)
    
    # We only care about the last two status entries
    status_list = status_list[-2:]

    # Record the timestamp when motion STARTS
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    
    # Record the timestamp when motion ENDS
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())


    # --- DASHBOARD PREPARATION & DISPLAY ---
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    delta_bgr = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack((frame, gray_bgr))
    bottom_row = np.hstack((delta_bgr, thresh_bgr))
    dashboard = np.vstack((top_row, bottom_row))

    cv2.imshow("Motion Detector Dashboard", dashboard)

    key = cv2.waitKey(1)
    if key == ord('q'):
        # If we are in the middle of a motion event when quitting, log the end time
        if status == 1:
            times.append(datetime.now())
        break

# --- POST-LOOP ANALYSIS & REPORT ---
print("Generating motion report...")

# Store the start and end times in the DataFrame
for i in range(0, len(times), 2):
    if i + 1 < len(times):
        df = pd.concat([df, pd.DataFrame({"Start": [times[i]], "End": [times[i+1]]})], ignore_index=True)

# Save the report to a CSV file
if not df.empty:
    df.to_csv("Motion_Log.csv")