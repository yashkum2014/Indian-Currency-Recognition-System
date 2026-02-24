import cv2
import numpy as np
import tensorflow as tf
import json
import pyttsx3
import time
import speech_recognition as sr
import winsound
from collections import Counter
from keras.layers import TFSMLayer

# ===============================
# CONFIG
# ===============================
IMG_HEIGHT = 160
IMG_WIDTH = 160

CONFIDENCE_THRESHOLD = 0.45
STABLE_COUNT_REQUIRED = 4
PRED_BUFFER_SIZE = 7
SPEAK_DELAY = 2.5

# ===============================               
# LOAD MODEL (KERAS 3 SAFE)
# ===============================
model = tf.keras.Sequential([
    TFSMLayer(
        "../models/mobilenetv2",
        call_endpoint="serving_default"
    )
])

print("✅ Model loaded")

# ===============================
# LOAD CLASS NAMES
# ===============================
with open("../class_names.json", "r") as f:
    class_names = json.load(f)

print("Classes:", class_names)

# ===============================
# TEXT TO SPEECH
# ===============================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# BEEP + PROMPT (ACCESSIBILITY)
# ===============================
def beep_and_prompt():
    winsound.Beep(1000, 400)   # beep
    time.sleep(0.3)
    speak("Say open camera")

# ===============================
# VOICE COMMAND TO START CAMERA
# ===============================
recognizer = sr.Recognizer()
mic = sr.Microphone()

print("🔔 Waiting for voice command...")

with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        beep_and_prompt()
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)

            if "open camera" in command:
                speak("Camera opened")
                break

        except sr.UnknownValueError:
            speak("I did not understand. Please say open camera")
        except sr.RequestError:
            speak("Speech service error")

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

print("🎥 Camera started | Press 'q' to quit")

# ===============================
# STATE VARIABLES
# ===============================
prediction_buffer = []
last_spoken_time = 0
last_spoken_label = None

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ---- ROI BOX ----
    box_size = int(min(w, h) * 0.55)
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(
        frame,
        "Place note inside the box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # ---- PREPROCESS ----
    img = cv2.resize(roi, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ---- PREDICTION ----
    pred_dict = model(img, training=False)
    preds = list(pred_dict.values())[0].numpy()

    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    prediction_buffer.append(class_id)
    if len(prediction_buffer) > PRED_BUFFER_SIZE:
        prediction_buffer.pop(0)

    label_text = "Detecting..."
    color = (0, 255, 255)

    if len(prediction_buffer) >= STABLE_COUNT_REQUIRED:
        counter = Counter(prediction_buffer)
        final_class_id, count = counter.most_common(1)[0]
        final_confidence = preds[0][final_class_id]

        if (
            final_confidence >= CONFIDENCE_THRESHOLD
            and count >= STABLE_COUNT_REQUIRED
        ):
            denomination = class_names[final_class_id]
            label_text = f"₹{denomination} ({final_confidence*100:.1f}%)"
            color = (0, 255, 0)

            current_time = time.time()
            if (
                current_time - last_spoken_time > SPEAK_DELAY
                or last_spoken_label != denomination
            ):
                speak(f"{denomination} rupees")
                last_spoken_time = current_time
                last_spoken_label = denomination
        else:
            label_text = "Hold note steady"

    # ---- DISPLAY ----
    cv2.putText(
        frame,
        label_text,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Indian Currency Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
