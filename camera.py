# import cv2

# camera = cv2.VideoCapture(0)

# if not camera.isOpened():
#     print("ERROR: Camera could not be opened")
#     exit()

# print("Camera started successfully!")
# print("Press Q to quit.")

# while True:

#     ret, frame = camera.read()

#     if not ret:
#         print("ERROR: Could not read camera frame")
#         break

#     cv2.imshow("Autonomous Vehicle Camera", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()




import cv2
from lane_detection import detect_lane

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Lane detection started!")
print("Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame")
        break

    output, direction, error = detect_lane(frame)

    cv2.putText(
        output,
        "Direction: " + direction,
        (20, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Lane Detection", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()