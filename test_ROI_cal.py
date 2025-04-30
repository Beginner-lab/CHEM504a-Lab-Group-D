import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

# === Start with these values and adjust them ===
x, y, w, h = 290, 63, 119, 120  # (x, y, width, height)

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Draw ROI rectangle on frame
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Extract ROI for preview
    roi = frame[y:y + h, x:x + w]
    cv2.imshow("Webcam with ROI", frame)
    cv2.imshow("ROI Preview", roi)

    print(f"ROI Position: x={x}, y={y}, width={w}, height={h}", end='\r')

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
