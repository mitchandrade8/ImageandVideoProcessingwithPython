import cv2
import time
from datetime import datetime
import numpy as np
import pandas as pd

# --- INITIALIZATION ---
first_frame = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns=["Start", "End"])

video = cv2.VideoCapture(0)

print("Letting camera warm up...")
time.sleep(2)
print("Camera ready.")

# --- MAIN LOOP ---
while True:
    check, frame = video.read()
    status = 0
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
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # --- OBJECT DETECTION & UNIFIED BOUNDING BOX ---
    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the combined bounding box of all detected motion
    min_x, min_y = frame.shape[1], frame.shape[0]
    max_x, max_y = 0, 0
    motion_detected_in_frame = False

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        
        # If we are here, it means a valid contour was found
        motion_detected_in_frame = True

        # Get the bounding box for the current contour
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Update the overall min/max coordinates to encompass this new contour
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
        
        # NOTE: We have REMOVED the cv2.rectangle() call from inside the loop

    # After checking all contours, draw ONE unified rectangle if motion was detected
    if motion_detected_in_frame:
        status = 1
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)


    # --- RECORD MOTION START & END TIMES ---
    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    # --- DASHBOARD PREPARATION & DISPLAY ---
    # (This section remains the same)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    delta_bgr = cv2.cvtColor(delta_frame, cv2.COLOR_GRAY2BGR)
    thresh_bgr = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

    top_row = np.hstack((frame, gray_bgr))
    bottom_row = np.hstack((delta_bgr, thresh_bgr))
    dashboard = np.vstack((top_row, bottom_row))

    cv2.imshow("Motion Detector Dashboard", dashboard)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# --- POST-LOOP ANALYSIS & REPORT ---
# (This section remains the same)
print("Generating motion report...")
for i in range(0, len(times), 2):
    if i + 1 < len(times):
        df = pd.concat([df, pd.DataFrame({"Start": [times[i]], "End": [times[i+1]]})], ignore_index=True)

if not df.empty:
    df.to_csv("Motion_Log.csv")
    print("Motion log saved to Motion_Log.csv")
    print(df)
else:
    print("No motion events were recorded.")

# --- CLEANUP ---
video.release()
cv2.destroyAllWindows()