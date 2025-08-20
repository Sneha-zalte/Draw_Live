import cv2
import numpy as np
import mediapipe as mp

# -------------------- Setup --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

canvas = None

# Brush settings
brush_color = (255, 0, 255)
brush_thickness = 7
eraser_thickness = 50
prev_x, prev_y = 0, 0

# Colors (add more)
colors = {
    "Purple": (255, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Cyan": (255, 255, 0),
    "White": (255, 255, 255),
    "Eraser": (0, 0, 0)
}

# Button positions and radii
color_buttons = {
    "Purple": (50, 50),
    "Green": (130, 50),
    "Blue": (210, 50),
    "Red": (290, 50),
    "Yellow": (370, 50),
    "Cyan": (450, 50),
    "White": (530, 50),
    "Eraser": (610, 50)
}
button_radius = 30

# Undo stack
undo_stack = []

# -------------------- Functions --------------------
def draw_color_buttons(img, selected_color):
    for name, pos in color_buttons.items():
        color = colors[name]
        thickness = 4 if color == selected_color else -1
        cv2.circle(img, pos, button_radius, color, thickness)
        if thickness != -1:  # Draw inner filled circle for selection
            cv2.circle(img, pos, button_radius-5, color, -1)
        cv2.putText(img, name, (pos[0]-25, pos[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# -------------------- Main Loop --------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_color_buttons(frame, brush_color)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x1, y1 = lm_list[8]  # Index tip
            x2, y2 = lm_list[4]  # Thumb tip

            pinch = abs(x1 - x2) < 40 and abs(y1 - y2) < 40

            # Check color selection
            for name, pos in color_buttons.items():
                dist = np.hypot(x1 - pos[0], y1 - pos[1])
                if dist < button_radius:
                    brush_color = colors[name]

            # Drawing
            if not pinch:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                thickness = eraser_thickness if brush_color == (0,0,0) else brush_thickness
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), brush_color, thickness)
                prev_x, prev_y = x1, y1
            else:
                prev_x, prev_y = 0, 0

            # Draw brush indicator
            indicator_radius = eraser_thickness if brush_color==(0,0,0) else brush_thickness
            cv2.circle(frame, (x1, y1), indicator_radius, brush_color, 2)

    # Merge canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv_mask = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    frame = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Virtual Drawing App", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)  # Clear
        undo_stack = []
    elif key == ord('u') and undo_stack:
        canvas = undo_stack.pop()  # Undo

cap.release()
cv2.destroyAllWindows()
