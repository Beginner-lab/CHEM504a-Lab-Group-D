import cv2
import numpy as np

# Initialize webcam
cam = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window
cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for blue (change these values for other colors)
    lower_blue = np.array([75, 150, 0])    # Lower bound for blue
    upper_blue = np.array([140, 255, 255])  # Upper bound for blue

    # Create a mask that detects blue color
    mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Optional: You can combine the mask with the original frame to see what is detected
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours of the blue color areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame (optional)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Only consider large enough areas (to avoid noise)
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Optional: Show the center of the color area
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            print(f"Detected blue color at: ({cx}, {cy})")

    # Show the original frame with color detection
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k % 256 == 32:
        # SPACE pressed
        img_name = f"opencv_frame_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

# Release the camera and close the window
