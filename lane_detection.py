import cv2
import numpy as np


def region_of_interest(image):
    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.35), int(height * 0.55)),
        (int(width * 0.65), int(height * 0.55)),
        (int(width * 0.95), height)
    ]])

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)

def average_slope_intercept(lines):

    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:

        x1, y1, x2, y2 = line.reshape(-1)

        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        length = np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

        if length < 30 or abs(slope) < 0.4:
            continue

        if slope < 0:
            left_lines.append((slope, intercept, length))
        else:
            right_lines.append((slope, intercept, length))

    def weighted_average(lines):

        if len(lines) == 0:
            return None

        slopes = np.array([x[0] for x in lines])
        intercepts = np.array([x[1] for x in lines])
        weights = np.array([x[2] for x in lines])

        return (
            np.average(slopes, weights=weights),
            np.average(intercepts, weights=weights)
        )

    return (
        weighted_average(left_lines),
        weighted_average(right_lines)
    )


def make_line_points(y1, y2, line):

    if line is None:
        return None

    slope, intercept = line

    if abs(slope) < 0.001:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return x1, y1, x2, y2


def detect_lane(frame):

    # Resize camera frame
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur, 50, 150)

    # Select road region
    roi = region_of_interest(edges)

    # Detect lines
    lines = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=30,
        maxLineGap=100
    )

    # Separate left and right lines
    left, right = average_slope_intercept(lines)

    output = frame.copy()

    height, width = frame.shape[:2]

    y_bottom = height
    y_top = int(height * 0.58)

    left_points = make_line_points(
        y_bottom,
        y_top,
        left
    )

    right_points = make_line_points(
        y_bottom,
        y_top,
        right
    )

    # Draw left lane
    if left_points:

        x1, y1, x2, y2 = left_points

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    # Draw right lane
    if right_points:

        x1, y1, x2, y2 = right_points

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            5
        )

    # Determine direction
    if left_points is not None and right_points is not None:

        left_x = left_points[2]
        right_x = right_points[2]

        lane_center = int(
            (left_x + right_x) / 2
        )

        vehicle_center = width // 2

        error = lane_center - vehicle_center

        if error < -60:
            direction = "LEFT"

        elif error > 60:
            direction = "RIGHT"

        else:
            direction = "STRAIGHT"

        # Vehicle center
        cv2.line(
            output,
            (vehicle_center, height),
            (vehicle_center, int(height * 0.65)),
            (255, 0, 0),
            2
        )

        # Lane center
        cv2.circle(
            output,
            (lane_center, int(height * 0.75)),
            8,
            (0, 0, 255),
            -1
        )

    else:

        direction = "NO_LANE"
        error = 0

    # Display direction
    cv2.putText(
        output,
        "Lane: " + direction,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return output, direction, error