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

WIDTH = 1280
HEIGHT = 720

BRUSH_SIZES = [4, 8, 14, 22]
BRUSH_SIZE = BRUSH_SIZES[1]
ERASER_SIZE = 40

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
TOOLBAR_HEIGHT = 65


def draw_toolbar(frame):

    # Fully transparent toolbar.
    # Only the controls themselves are drawn over the camera feed.

    # Colors
    for i, color in enumerate(colors):

        x = COLOR_START_X + i * COLOR_SPACING

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

    # Brush sizes
    brush_x_positions = [390, 430, 470, 510]

    for i, size in enumerate(BRUSH_SIZES):

        x = brush_x_positions[i]

        cv2.circle(
            frame,
            (x, COLOR_Y),
            max(4, min(size // 2, 11)),
            (255, 255, 255),
            -1
        )

        if size == BRUSH_SIZE:
            cv2.circle(
                frame,
                (x, COLOR_Y),
                18,
                (255, 255, 255),
                2
            )

    # Eraser
    eraser_x = 565

    cv2.rectangle(
        frame,
        (eraser_x - 12, 18),
        (eraser_x + 12, 42),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (eraser_x - 15, 45),
        (eraser_x + 15, 45),
        (255, 255, 255),
        2
    )

    # Save
    save_x1 = 610
    save_x2 = 675

    cv2.rectangle(
        frame,
        (save_x1, 13),
        (save_x2, 47),
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "SAVE",
        (save_x1 + 9, 36),
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
# BRUSH SIZE
# ============================================================

def get_brush_size(x, y):
    if y > TOOLBAR_HEIGHT:
        return None

    brush_x_positions = [390, 430, 470, 510]

    for i, brush_x in enumerate(brush_x_positions):
        if abs(x - brush_x) <= 18:
            return BRUSH_SIZES[i]

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
# COLOR STATE
# ============================================================

color_candidate = None
color_candidate_start = None


# ============================================================
# DRAW STATE
# ============================================================

previous_points = {}
smooth_points = {}


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    "Virtual Painter",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Virtual Painter",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
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
print("Index + middle   = Select brush size")
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

            current_x = int(hand[8].x * WIDTH)
            current_y = int(hand[8].y * HEIGHT)

            if hand_index in smooth_points:
                old_x, old_y = smooth_points[hand_index]

                ix = int(old_x * 0.7 + current_x * 0.3)
                iy = int(old_y * 0.7 + current_y * 0.3)
            else:
                ix = current_x
                iy = current_y

            smooth_points[hand_index] = (ix, iy)


            # =================================================
            # COLOR SELECTION
            # =================================================

            if index_middle(hand):

                previous_points.pop(
                    hand_index,
                    None
                )

                selected = get_toolbar_color(ix, iy)
                selected_brush = get_brush_size(ix, iy)

                if selected is not None:
                    color_id, selected_color = selected
                    candidate_id = ("color", color_id)

                    if color_candidate != candidate_id:
                        color_candidate = candidate_id
                        color_candidate_start = time.time()
                    elif color_candidate_start is not None:
                        elapsed = time.time() - color_candidate_start

                        if elapsed >= COLOR_HOLD_TIME:
                            current_color = selected_color
                            print("COLOR SELECTED:", color_names[color_id])
                            color_candidate = None
                            color_candidate_start = None

                elif selected_brush is not None:
                    candidate_id = ("brush", selected_brush)

                    if color_candidate != candidate_id:
                        color_candidate = candidate_id
                        color_candidate_start = time.time()
                    elif color_candidate_start is not None:
                        elapsed = time.time() - color_candidate_start

                        if elapsed >= COLOR_HOLD_TIME:
                            BRUSH_SIZE = selected_brush
                            print("BRUSH SIZE:", BRUSH_SIZE)
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
                smooth_points.pop(
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
            smooth_points.pop(
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
    # TOOLBAR
    # ========================================================

    display = draw_toolbar(
        display
    )


    # ========================================================
    # SHOW
    # ========================================================

    # Keep the 16:9 aspect ratio in fullscreen.
    # This adds black bars instead of stretching the image.
    screen_w = 1920
    screen_h = 1080

    try:
        _, _, detected_w, detected_h = cv2.getWindowImageRect("Virtual Painter")
        if detected_w > 0 and detected_h > 0:
            screen_w = detected_w
            screen_h = detected_h
    except:
        pass

    scale = min(screen_w / WIDTH, screen_h / HEIGHT)

    new_w = int(WIDTH * scale)
    new_h = int(HEIGHT * scale)

    resized = cv2.resize(
        display,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    fullscreen_frame = np.zeros(
        (screen_h, screen_w, 3),
        dtype=np.uint8
    )

    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2

    fullscreen_frame[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    cv2.imshow(
        "Virtual Painter",
        fullscreen_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        cv2.setWindowProperty(
            "Virtual Painter",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )

    if key == 0x7A:  # F11 on some OpenCV/Windows setups
        cv2.setWindowProperty(
            "Virtual Painter",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_NORMAL
        )

    if key == ord("q"):

        break

    if key == ord("c"):

        canvas[:] = 0

        previous_points.clear()
        smooth_points.clear()

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
