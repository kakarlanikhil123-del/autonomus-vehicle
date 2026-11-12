
import cv2
from lane_detection import detect_lane

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit()

print("Lane detection started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    output, direction, error = detect_lane(frame)

    print(f"\rLane: {direction} | Error: {error}", end="", flush=True)

    cv2.imshow("Lane Detection Test", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\nLane detection stopped.")
