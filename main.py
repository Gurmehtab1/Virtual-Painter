import cv2
import mediapipe as mp
import numpy as np
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"

WIDTH = 640
HEIGHT = 480

BRUSH_SIZE = 8
ERASER_SIZE = 40

PINCH_THRESHOLD = 0.10
COLOR_HOLD_TIME = 0.35


# ============================================================
# CAMERA
# ============================================================

print("Starting camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    input("Press Enter to exit...")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

print("Camera opened successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

print("Loading MediaPipe...")

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(
    options
)

print("MediaPipe loaded.")


# ============================================================
# CANVAS
# ============================================================

canvas = np.zeros(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
)


# ============================================================
# COLORS
# ============================================================

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 255)
]

color_names = [
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "PINK",
    "WHITE"
]

current_color = colors[0]


# ============================================================
# TOOLBAR
# ============================================================

COLOR_START_X = 30
COLOR_SPACING = 55
COLOR_Y = 30
TOOLBAR_HEIGHT = 60


def draw_toolbar(frame):

    overlay = frame.copy()

    # Transparent toolbar
    cv2.rectangle(
        overlay,
        (0, 0),
        (WIDTH, TOOLBAR_HEIGHT),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.18,
        frame,
        0.82,
        0
    )

    # Colors
    for i, color in enumerate(colors):

        x = (
            COLOR_START_X
            +
            i * COLOR_SPACING
        )

        cv2.circle(
            frame,
            (x, COLOR_Y),
            17,
            color,
            -1
        )

        cv2.circle(
            frame,
            (x, COLOR_Y),
            18,
            (255, 255, 255),
            1
        )

        # Selected color
        if color == current_color:

            cv2.circle(
                frame,
                (x, COLOR_Y),
                22,
                (255, 255, 255),
                2
            )

    # Brush
    cv2.circle(
        frame,
        (360, 30),
        8,
        (255, 255, 255),
        -1
    )

    # Eraser
    cv2.rectangle(
        frame,
        (399, 19),
        (421, 41),
        (255, 255, 255),
        1
    )

    # Save
    cv2.rectangle(
        frame,
        (445, 13),
        (505, 47),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "SAVE",
        (454, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# FINGER DETECTION
# ============================================================

def distance(a, b):

    dx = a.x - b.x
    dy = a.y - b.y

    return math.sqrt(
        dx * dx + dy * dy
    )


def finger_up(hand, tip, pip):

    return hand[tip].y < hand[pip].y


def get_fingers(hand):

    index = finger_up(hand, 8, 6)
    middle = finger_up(hand, 12, 10)
    ring = finger_up(hand, 16, 14)
    pinky = finger_up(hand, 20, 18)

    return index, middle, ring, pinky


def index_only(hand):

    index, middle, ring, pinky = get_fingers(hand)

    return (
        index
        and not middle
        and not ring
        and not pinky
    )


def index_middle(hand):

    index, middle, ring, pinky = get_fingers(hand)

    return (
        index
        and middle
        and not ring
        and not pinky
    )


def open_hand(hand):

    wrist = hand[0]

    index_dist = distance(
        hand[8],
        wrist
    )

    middle_dist = distance(
        hand[12],
        wrist
    )

    ring_dist = distance(
        hand[16],
        wrist
    )

    pinky_dist = distance(
        hand[20],
        wrist
    )

    return (
        index_dist > 0.28
        and
        middle_dist > 0.28
        and
        ring_dist > 0.25
        and
        pinky_dist > 0.20
    )


def pinch(hand):

    return distance(
        hand[4],
        hand[8]
    ) < PINCH_THRESHOLD


# ============================================================
# HAND TRACKER
# ============================================================

def draw_hand_tracker(frame, hand):

    h, w = frame.shape[:2]

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20)
    ]

    points = []

    for landmark in hand:

        x = int(
            landmark.x * w
        )

        y = int(
            landmark.y * h
        )

        points.append(
            (x, y)
        )

    # White skeleton
    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 255, 255),
            1
        )

    # Black dots
    for point in points:

        cv2.circle(
            frame,
            point,
            3,
            (0, 0, 0),
            -1
        )


# ============================================================
# TOOLBAR COLOR
# ============================================================

def get_toolbar_color(x, y):

    if y > TOOLBAR_HEIGHT:
        return None

    for i, color in enumerate(colors):

        circle_x = (
            COLOR_START_X
            +
            i * COLOR_SPACING
        )

        d = math.sqrt(
            (x - circle_x) ** 2
            +
            (y - COLOR_Y) ** 2
        )

        if d <= 25:

            return i, color

    return None


# ============================================================
# SAVE PAINTING
# ============================================================

def save_painting(camera_clean):

    filename = (
        f"painting_{int(time.time())}.png"
    )

    final_image = cv2.add(
        camera_clean,
        canvas
    )

    cv2.imwrite(
        filename,
        final_image
    )

    print(
        "Painting saved:",
        filename
    )


# ============================================================
# SAVE SCREENSHOT
# ============================================================

def save_screenshot(
    camera_clean,
    start,
    end
):

    x1 = min(
        start[0],
        end[0]
    )

    y1 = min(
        start[1],
        end[1]
    )

    x2 = max(
        start[0],
        end[0]
    )

    y2 = max(
        start[1],
        end[1]
    )

    x1 = max(
        0,
        min(WIDTH - 1, x1)
    )

    y1 = max(
        0,
        min(HEIGHT - 1, y1)
    )

    x2 = max(
        0,
        min(WIDTH, x2)
    )

    y2 = max(
        0,
        min(HEIGHT, y2)
    )

    if (
        x2 - x1 < 20
        or
        y2 - y1 < 20
    ):

        print(
            "Screenshot area too small."
        )

        return

    screenshot = camera_clean[
        y1:y2,
        x1:x2
    ]

    filename = (
        f"screenshot_{int(time.time())}.png"
    )

    cv2.imwrite(
        filename,
        screenshot
    )

    print(
        "Screenshot saved:",
        filename
    )


# ============================================================
# SCREENSHOT STATE
# ============================================================

screenshot_active = False

screenshot_start = None
screenshot_end = None

previous_pinch = False


# ============================================================
# COLOR STATE
# ============================================================

color_candidate = None
color_candidate_start = None


# ============================================================
# DRAW STATE
# ============================================================

previous_points = {}


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    "Virtual Painter"
)


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("Virtual Painter started.")
print()
print("Index only       = Draw")
print("Index + middle   = Select color")
print("Open hand        = Erase")
print("Thumb + index    = Screenshot")
print()
print("Q = Quit")
print("C = Clear")
print()


while True:

    # ========================================================
    # CAMERA
    # ========================================================

    ret, frame = cap.read()

    if not ret:

        print(
            "ERROR: Could not read camera."
        )

        break

    frame = cv2.flip(
        frame,
        1
    )

    camera_clean = frame.copy()


    # ========================================================
    # MEDIAPIPE
    # ========================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(
        mp_image
    )


    # ========================================================
    # SCREENSHOT PINCH
    # ========================================================

    pinch_found = False

    pinch_position = None


    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            if pinch(hand):

                pinch_found = True

                # Use index fingertip
                pinch_position = (
                    int(hand[8].x * WIDTH),
                    int(hand[8].y * HEIGHT)
                )

                break


    # --------------------------------------------------------
    # Pinch START
    # --------------------------------------------------------

    if pinch_found and not previous_pinch:

        screenshot_active = True

        screenshot_start = pinch_position

        screenshot_end = pinch_position

        print(
            "Screenshot selection started."
        )


    # --------------------------------------------------------
    # Pinch MOVE
    # --------------------------------------------------------

    if pinch_found and screenshot_active:

        screenshot_end = pinch_position


    # --------------------------------------------------------
    # Pinch RELEASE
    # --------------------------------------------------------

    if (
        not pinch_found
        and
        previous_pinch
        and
        screenshot_active
    ):

        if (
            screenshot_start is not None
            and
            screenshot_end is not None
        ):

            save_screenshot(
                camera_clean,
                screenshot_start,
                screenshot_end
            )

        screenshot_active = False

        screenshot_start = None
        screenshot_end = None

        print(
            "Screenshot selection finished."
        )


    previous_pinch = pinch_found


    # ========================================================
    # NORMAL HAND PROCESSING
    # ========================================================

    if result.hand_landmarks:

        for hand_index, hand in enumerate(
            result.hand_landmarks
        ):

            # Hand tracker
            draw_hand_tracker(
                frame,
                hand
            )

            ix = int(
                hand[8].x * WIDTH
            )

            iy = int(
                hand[8].y * HEIGHT
            )


            # =================================================
            # SCREENSHOT HAS PRIORITY
            # =================================================

            if screenshot_active:

                previous_points.pop(
                    hand_index,
                    None
                )

                continue


            # =================================================
            # COLOR SELECTION
            # =================================================

            if index_middle(hand):

                previous_points.pop(
                    hand_index,
                    None
                )

                selected = get_toolbar_color(
                    ix,
                    iy
                )

                if selected is not None:

                    color_id, selected_color = selected

                    if color_candidate != color_id:

                        color_candidate = color_id

                        color_candidate_start = (
                            time.time()
                        )

                    else:

                        if color_candidate_start is not None:

                            elapsed = (
                                time.time()
                                -
                                color_candidate_start
                            )

                            if elapsed >= COLOR_HOLD_TIME:

                                current_color = selected_color

                                print(
                                    "COLOR SELECTED:",
                                    color_names[color_id]
                                )

                                color_candidate = None

                                color_candidate_start = None

                else:

                    color_candidate = None
                    color_candidate_start = None

                # Cursor
                cv2.circle(
                    frame,
                    (ix, iy),
                    10,
                    (255, 255, 255),
                    1
                )

                continue


            # =================================================
            # RESET COLOR SELECTION
            # =================================================

            color_candidate = None
            color_candidate_start = None


            # =================================================
            # OPEN HAND = ERASE
            # =================================================

            if open_hand(hand):

                cv2.circle(
                    canvas,
                    (ix, iy),
                    ERASER_SIZE,
                    (0, 0, 0),
                    -1
                )

                previous_points.pop(
                    hand_index,
                    None
                )

                continue


            # =================================================
            # INDEX ONLY = DRAW
            # =================================================

            if index_only(hand):

                current = (
                    ix,
                    iy
                )

                previous = previous_points.get(
                    hand_index
                )

                if previous is not None:

                    cv2.line(
                        canvas,
                        previous,
                        current,
                        current_color,
                        BRUSH_SIZE,
                        cv2.LINE_AA
                    )

                previous_points[hand_index] = current

                continue


            # =================================================
            # NOTHING
            # =================================================

            previous_points.pop(
                hand_index,
                None
            )


    # ========================================================
    # DISPLAY
    # ========================================================

    display = cv2.add(
        camera_clean,
        canvas
    )


    # ========================================================
    # HAND TRACKER
    # ========================================================

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            draw_hand_tracker(
                display,
                hand
            )


    # ========================================================
    # SCREENSHOT RECTANGLE
    # ========================================================

    if (
        screenshot_active
        and
        screenshot_start is not None
        and
        screenshot_end is not None
    ):

        cv2.rectangle(
            display,
            screenshot_start,
            screenshot_end,
            (255, 255, 255),
            1
        )


    # ========================================================
    # TOOLBAR
    # ========================================================

    display = draw_toolbar(
        display
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Virtual Painter",
        display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break

    if key == ord("c"):

        canvas[:] = 0

        previous_points.clear()

        print(
            "Canvas cleared."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "Virtual Painter closed."
)