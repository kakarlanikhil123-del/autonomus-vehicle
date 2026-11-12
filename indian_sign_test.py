import cv2
from ultralytics import YOLO

print("Loading Indian traffic-sign model...")

model = YOLO("indian_traffic.pt")

print("Model loaded!")
print("Starting camera...")
print("Press Q to quit.")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened")
    exit()

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame")
        break

    results = model(frame, conf=0.20, verbose=False)

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            print(
                f"Detected: {class_name} "
                f"Confidence: {confidence:.2f}"
            )

    output = result.plot()

    cv2.imshow("Indian Traffic Sign Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
