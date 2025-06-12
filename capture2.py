import cv2
import time

# Initialize video capture and the first_frame variable
video = cv2.VideoCapture(0)
first_frame = None

# --- FIX 1: Give the camera a moment to warm up ---
# We read and discard a few frames to let the sensor adjust.
print("Letting camera warm up...")
for i in range(30):
    check, frame = video.read()
    if not check:
        time.sleep(0.1) # Wait a bit if frames aren't coming in yet

print("Camera ready.")

while True:
    check, frame = video.read()
    if not check:
        break

    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # In the first valid loop, set the first_frame
    if first_frame is None:
        first_frame = gray
        continue

    # Calculate the difference between the background and current frame
    delta_frame = cv2.absdiff(first_frame, gray)

    # --- We can add a threshold to make motion stand out ---
    # Any pixel with a difference > 30 will be turned white (255)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    
    # Display the different frames
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame (Motion)", thresh_frame) # This shows motion best

    # --- FIX 2: Removed print() statements for performance ---

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()