
import cv2
import re
import pytesseract
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best.pt"

# Change this if Tesseract is installed somewhere else
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CAMERA_INDEX = 0

# YOLO confidence threshold
CONFIDENCE = 0.35


# ============================================================
# OCR SETUP
# ============================================================

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("=============================================")
print("Loading traffic-sign detection model...")
print("==============================================")

model = YOLO(MODEL_PATH)

print("Traffic-sign model loaded successfully!")
print()
print("Classes available in your model:")

for class_id, class_name in model.names.items():
    print(f"{class_id} : {class_name}")

print()


# ============================================================
# CAMERA SETUP
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started successfully!")
print("Press Q to quit.")
print()


# ============================================================
# SPEED LIMIT OCR FUNCTION
# ============================================================

def read_speed_limit(crop):
    """
    Reads a speed-limit number from a detected sign.
    """

    if crop is None or crop.size == 0:
        return None

    try:

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        crop = cv2.resize(
            crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        # ----------------------------------------------------
        # Convert to grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY
        )

        # ----------------------------------------------------
        # Reduce noise
        # ----------------------------------------------------

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        _, threshold = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        config = (
            "--psm 7 "
            "-c tessedit_char_whitelist=0123456789"
        )

        text = pytesseract.image_to_string(
            threshold,
            config=config
        )

        # ----------------------------------------------------
        # Extract numbers
        # ----------------------------------------------------

        numbers = re.findall(
            r"\d+",
            text
        )

        if not numbers:
            return None

        # Take longest number
        number = max(
            numbers,
            key=len
        )

        return number

    except Exception as error:

        print(
            "OCR Error:",
            error
        )

        return None


# ============================================================
# DETERMINE ACTION FOR EACH SIGN
# ============================================================

def get_action(class_name):
    """
    Converts traffic-sign class into driving action.
    """

    name = class_name.lower().strip()

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if name == "stop" or "stop sign" in name:

        return "STOP"

    # --------------------------------------------------------
    # RED LIGHT
    # --------------------------------------------------------

    elif "red light" in name:

        return "STOP"

    # --------------------------------------------------------
    # GREEN LIGHT
    # --------------------------------------------------------

    elif "green light" in name:

        return "FORWARD"

    # --------------------------------------------------------
    # YELLOW / AMBER LIGHT
    # --------------------------------------------------------

    elif (
        "yellow light" in name
        or "amber light" in name
    ):

        return "SLOW DOWN"

    # --------------------------------------------------------
    # SPEED LIMIT
    # --------------------------------------------------------

    elif "speed limit" in name:

        return "SLOW"

    # --------------------------------------------------------
    # NO ENTRY
    # --------------------------------------------------------

    elif "no entry" in name:

        return "STOP"

    # --------------------------------------------------------
    # NO LEFT TURN
    # --------------------------------------------------------

    elif "no left" in name:

        return "DO NOT TURN LEFT"

    # --------------------------------------------------------
    # NO RIGHT TURN
    # --------------------------------------------------------

    elif "no right" in name:

        return "DO NOT TURN RIGHT"

    # --------------------------------------------------------
    # NO U-TURN
    # --------------------------------------------------------

    elif "no u" in name or "u-turn prohibited" in name:

        return "NO U-TURN"

    # --------------------------------------------------------
    # LEFT TURN
    # --------------------------------------------------------

    elif (
        "left turn" in name
        or "turn left" in name
    ):

        return "TURN LEFT"

    # --------------------------------------------------------
    # RIGHT TURN
    # --------------------------------------------------------

    elif (
        "right turn" in name
        or "turn right" in name
    ):

        return "TURN RIGHT"

    # --------------------------------------------------------
    # U-TURN
    # --------------------------------------------------------

    elif "u-turn" in name or "u turn" in name:

        return "U-TURN"

    # --------------------------------------------------------
    # KEEP LEFT
    # --------------------------------------------------------

    elif "keep left" in name:

        return "KEEP LEFT"

    # --------------------------------------------------------
    # KEEP RIGHT
    # --------------------------------------------------------

    elif "keep right" in name:

        return "KEEP RIGHT"

    # --------------------------------------------------------
    # PEDESTRIAN CROSSING
    # --------------------------------------------------------

    elif (
        "pedestrian" in name
        or "pedestrian crossing" in name
    ):

        return "SLOW - PEDESTRIAN"

    # --------------------------------------------------------
    # SCHOOL ZONE
    # --------------------------------------------------------

    elif "school" in name:

        return "SLOW - SCHOOL ZONE"

    # --------------------------------------------------------
    # ROUNDABOUT
    # --------------------------------------------------------

    elif "roundabout" in name:

        return "ROUNDABOUT"

    # --------------------------------------------------------
    # GIVE WAY
    # --------------------------------------------------------

    elif (
        "give way" in name
        or "yield" in name
    ):

        return "GIVE WAY"

    # --------------------------------------------------------
    # HAZARD
    # --------------------------------------------------------

    elif (
        "warning" in name
        or "hazard" in name
        or "danger" in name
    ):

        return "SLOW DOWN"

    # --------------------------------------------------------
    # UNKNOWN SIGN
    # --------------------------------------------------------

    else:

        return "CAUTION"


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ CAMERA FRAME
    # --------------------------------------------------------

    ret, frame = camera.read()

    if not ret:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    output = frame.copy()


    # ========================================================
    # DETECTIONS
    # ========================================================

    detections = []

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = float(
                box.conf[0]
            )

            # ------------------------------------------------
            # Class ID
            # ------------------------------------------------

            class_id = int(
                box.cls[0]
            )

            # ------------------------------------------------
            # Class name
            # ------------------------------------------------

            class_name = model.names[
                class_id
            ]

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Keep coordinates inside frame
            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )

            detections.append({

                "class": class_name,

                "confidence": confidence,

                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            })


    # ========================================================
    # PROCESS ALL DETECTIONS
    # ========================================================

    if len(detections) > 0:

        # Sort detections by confidence
        detections.sort(
            key=lambda x: x["confidence"],
            reverse=True
        )


        # ----------------------------------------------------
        # Process every detected sign
        # ----------------------------------------------------

        for detection in detections:

            class_name = detection["class"]

            confidence = detection["confidence"]

            x1, y1, x2, y2 = detection["box"]


            # ------------------------------------------------
            # Determine action
            # ------------------------------------------------

            action = get_action(
                class_name
            )


            # ------------------------------------------------
            # Speed-limit OCR
            # ------------------------------------------------

            speed_number = None

            if "speed limit" in class_name.lower():

                crop = frame[
                    y1:y2,
                    x1:x2
                ]

                speed_number = read_speed_limit(
                    crop
                )

                if speed_number is not None:

                    label = (
                        f"SPEED LIMIT "
                        f"{speed_number} km/h "
                        f"{confidence:.2f}"
                    )

                    print(
                        f"[SIGN] Speed Limit: "
                        f"{speed_number} km/h | "
                        f"Confidence: "
                        f"{confidence:.2f}"
                    )

                else:

                    label = (
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )

                    print(
                        f"[SIGN] {class_name} | "
                        f"Confidence: "
                        f"{confidence:.2f} | "
                        f"OCR: Not detected"
                    )

            else:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                print(
                    f"[SIGN] {class_name} | "
                    f"Confidence: "
                    f"{confidence:.2f}"
                )


            # =================================================
            # BOUNDING BOX COLOR
            # =================================================

            # Default green
            box_color = (
                0,
                255,
                0
            )

            # Red for STOP
            if action == "STOP":

                box_color = (
                    0,
                    0,
                    255
                )

            # Yellow for slow
            elif (
                "SLOW" in action
                or action == "CAUTION"
            ):

                box_color = (
                    0,
                    255,
                    255
                )


            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # ------------------------------------------------
            # Draw label background
            # ------------------------------------------------

            text_y = max(
                y1 - 10,
                30
            )

            cv2.putText(
                output,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                box_color,
                2
            )


        # ====================================================
        # BEST DETECTION
        # ====================================================

        best_detection = detections[0]

        best_class = best_detection[
            "class"
        ]

        best_confidence = best_detection[
            "confidence"
        ]

        best_action = get_action(
            best_class
        )


        # ----------------------------------------------------
        # Main action display
        # ----------------------------------------------------

        action_text = (
            "ACTION: "
            + best_action
        )

        cv2.putText(
            output,
            action_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Number of signs detected
        # ----------------------------------------------------

        cv2.putText(
            output,
            f"SIGNS DETECTED: {len(detections)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    else:

        # ====================================================
        # NO SIGN DETECTED
        # ====================================================

        cv2.putText(
            output,
            "No traffic sign detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ========================================================
    # DISPLAY CAMERA
    # ========================================================

    cv2.imshow(
        "Traffic Sign Detection",
        output
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Camera stopped.")
print("Program ended.")

