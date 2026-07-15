import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from collections import deque

# ---------- CONFIG ----------
STREAM_URL = "http://192.168.0.100:81/stream"   # CHANGE TO YOUR ESP IP
MODEL_PATH = Path("training_output/tinydrowsy_fp32.onnx")
CLASS_NAMES = ["closed", "open"]
IMAGE_SIZE = 64

# ---------- PERCLOS & ALERT PARAMETERS ----------
WINDOW_SECONDS = 45
CLOSED_THRESHOLD = 0.5
PERCLOS_LEVEL1 = 0.10
PERCLOS_LEVEL2 = 0.20
MICROSLEEP_SECONDS = 3.0

# ---------- LOAD CASCADES ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Failed to load cascade classifiers")

# ---------- LOAD ONNX MODEL ----------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")

session = ort.InferenceSession(str(MODEL_PATH), providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------- HELPER FUNCTIONS ----------
def softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def preprocess_eye(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    inp = gray.astype(np.float32) / 255.0
    inp = (inp - 0.5) / 0.5
    inp = inp[np.newaxis, np.newaxis, :, :]
    return gray, inp

def predict_eye(roi):
    if roi is None or roi.size == 0:
        return 0.5, "invalid"
    _, inp = preprocess_eye(roi)
    logits = session.run([output_name], {input_name: inp})[0]
    probs = softmax(logits)[0]
    closed_prob = probs[0]
    open_prob = probs[1]
    pred_class = "closed" if closed_prob > open_prob else "open"
    return closed_prob, pred_class

def get_fixed_eye_region(face, side):
    x, y, w, h = face
    if side == "left":
        ex = x + int(w * 0.25)
        ey = y + int(h * 0.25)
        ew = int(w * 0.20)
        eh = int(h * 0.18)
    else:
        ex = x + int(w * 0.55)
        ey = y + int(h * 0.25)
        ew = int(w * 0.20)
        eh = int(h * 0.18)
    return ex, ey, ew, eh

# ======================================================
# NEW: LED & BUZZER SIMULATION HELPERS
# ======================================================

def get_gpio_state(alert_state, current_time, last_toggle):
    """
    Returns: (led_color, led_on, buzzer_on, blink_interval, state_label)
    """
    led_color = (0, 255, 0)  # Green default
    interval = 1.5
    buzzer_on = False
    label = "Awake"

    if alert_state == 0:
        led_color = (0, 255, 0)  # Green
        interval = 1.5
        buzzer_on = False
        label = "Awake"

    elif alert_state == 1:
        led_color = (0, 0, 255)  # Red
        interval = 1.0
        buzzer_on = ( (current_time % interval) < 0.1 )  # short beep
        label = "Soft"

    elif alert_state == 2:
        led_color = (0, 0, 255)  # Red
        interval = 0.5
        buzzer_on = ( (current_time % interval) < 0.2 )  # medium beep
        label = "Strong"

    elif alert_state == 3:
        led_color = (0, 0, 255)  # Red
        interval = 0.15
        buzzer_on = True  # CONTINUOUS
        label = "CRITICAL"

    # Toggle LED
    led_on = ( (current_time % interval) < (interval * 0.5) )

    return led_color, led_on, buzzer_on, interval, label

# For console logging (avoid spam)
last_gpio_print_state = -1
last_buzzer_print_state = None

def log_gpio_state(alert_state, buzzer_on):
    global last_gpio_print_state, last_buzzer_print_state
    if alert_state != last_gpio_print_state or buzzer_on != last_buzzer_print_state:
        state_names = ["Awake", "Soft Alert", "Strong Alert", "CRITICAL"]
        buzzer_str = "ON" if buzzer_on else "OFF"
        print(f"[GPIO] State: {state_names[alert_state]} | Buzzer: {buzzer_str}")
        last_gpio_print_state = alert_state
        last_buzzer_print_state = buzzer_on

# ======================================================

# ---------- STATE VARIABLES ----------
eye_history = deque()
closed_start_time = None
alert_state = 0
current_led_color = (0, 255, 0)
led_on = False

# ---------- CAMERA ----------
print(f"[INFO] Connecting to camera: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    raise RuntimeError(f"Could not open stream: {STREAM_URL}")

print("[INFO] Camera opened. Starting loop...")
frame_count = 0
fps = 0.0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        frame_count += 1

        # Estimate FPS
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = 10 / elapsed if elapsed > 0 else 0
            start_time = time.time()

        if not ret or frame is None:
            time.sleep(0.05)
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()
        frame_height, frame_width = frame.shape[:2]

        # ---------- 1. FACE DETECTION ----------
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # ---------- 2. EYE DETECTION ----------
            face_roi_gray = gray_frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            detected_eyes = []
            for (ex, ey, ew, eh) in eyes:
                detected_eyes.append((x + ex, y + ey, ew, eh))
            detected_eyes.sort(key=lambda e: e[0])

            if len(detected_eyes) >= 2:
                left_eye = detected_eyes[0]
                right_eye = detected_eyes[1]
            elif len(detected_eyes) == 1:
                if detected_eyes[0][0] < x + w/2:
                    left_eye = detected_eyes[0]
                    right_eye = get_fixed_eye_region(face, "right")
                else:
                    right_eye = detected_eyes[0]
                    left_eye = get_fixed_eye_region(face, "left")
            else:
                left_eye = get_fixed_eye_region(face, "left")
                right_eye = get_fixed_eye_region(face, "right")

            # ---------- 3. PREDICT BOTH EYES ----------
            lx, ly, lw, lh = left_eye
            left_roi = frame[ly:ly+lh, lx:lx+lw]
            left_closed, left_class = predict_eye(left_roi)

            rx, ry, rw, rh = right_eye
            right_roi = frame[ry:ry+rh, rx:rx+rw]
            right_closed, right_class = predict_eye(right_roi)

            avg_closed = (left_closed + right_closed) / 2.0

            # ---------- 4. PERCLOS & MICROSLEEP ----------
            eye_history.append((current_time, avg_closed))
            while eye_history and eye_history[0][0] < current_time - WINDOW_SECONDS:
                eye_history.popleft()

            if len(eye_history) > 0:
                closed_frames = sum(1 for _, p in eye_history if p > CLOSED_THRESHOLD)
                perclos = closed_frames / len(eye_history)
            else:
                perclos = 0.0

            microsleep_active = False
            if avg_closed > CLOSED_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = current_time
                elif current_time - closed_start_time >= MICROSLEEP_SECONDS:
                    microsleep_active = True
            else:
                closed_start_time = None

            # ---------- 5. ALERT STATE MACHINE ----------
            if microsleep_active:
                alert_state = 3
            elif perclos >= PERCLOS_LEVEL2:
                alert_state = 2
            elif perclos >= PERCLOS_LEVEL1:
                alert_state = 1
            else:
                alert_state = 0

            # ---------- 6. DRAW EYE RECTANGLES ----------
            left_color = (0, 255, 0) if left_class == "open" else (0, 0, 255)
            right_color = (0, 255, 0) if right_class == "open" else (0, 0, 255)

            cv2.rectangle(frame, (lx, ly), (lx+lw, ly+lh), left_color, 2)
            cv2.putText(frame, f"L: {left_class} {left_closed:.2f}", (lx, ly-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)

            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), right_color, 2)
            cv2.putText(frame, f"R: {right_class} {right_closed:.2f}", (rx, ry-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)

            # ---------- 7. LED & BUZZER SIMULATION ----------
            led_color, led_on, buzzer_on, interval, state_label = get_gpio_state(
                alert_state, current_time, 0.0  # last_toggle not needed here anymore
            )
            log_gpio_state(alert_state, buzzer_on)

            # Draw virtual LED (top-right corner)
            led_radius = 20
            led_center = (frame_width - 50, 50)
            led_display_color = led_color if led_on else (50, 50, 50)
            cv2.circle(frame, led_center, led_radius, led_display_color, -1)
            cv2.circle(frame, led_center, led_radius, (255, 255, 255), 1)  # border

            # Display buzzer status text
            buzzer_text = "BUZZER: ON" if buzzer_on else "BUZZER: OFF"
            cv2.putText(frame, buzzer_text, (frame_width - 120, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if buzzer_on else (100, 100, 100), 1)

            # ---------- 8. DISPLAY PERCLOS & ALERT STATUS ----------
            info_color = (0, 255, 255) if alert_state == 0 else (0, 0, 255)
            cv2.putText(frame, f"PERCLOS: {perclos:.2%}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"State: {state_label}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Critical alert: red flashing border
            if alert_state == 3:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(frame, "!!! CRITICAL - WAKE UP !!!", (50, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("Drowsiness Detector (LED + Buzzer Sim)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Exited.")