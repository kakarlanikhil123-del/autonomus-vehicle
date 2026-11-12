import cv2
from ultralytics import YOLO

print("Loading YOLO...")

model = YOLO("yolo11n.pt")

print("YOLO loaded!")
print("Starting camera...")
print("Press Q to quit.")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera")
        break

    # YOLO detection
    results = model(frame, conf=0.5, verbose=False)

    # Draw detections
    output = results[0].plot()

    cv2.imshow("YOLO Object Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()