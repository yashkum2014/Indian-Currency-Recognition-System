import cv2
import numpy as np
import tensorflow as tf
import json
import pyttsx3
import os
from keras.layers import TFSMLayer

# ===============================
# CONFIG
# ===============================
IMG_HEIGHT = 160
IMG_WIDTH = 160
CONFIDENCE_THRESHOLD = 0.25   # LOWERED for testing/debugging
IMAGE_FOLDER = "../data/test_images"

# ===============================
# LOAD MODEL (KERAS 3 SAFE)
# ===============================
model = tf.keras.Sequential([
    TFSMLayer(
        "../models/mobilenetv2",
        call_endpoint="serving_default"
    )
])

print("✅ Model loaded successfully")

# ===============================
# LOAD CLASS NAMES
# ===============================
with open("../class_names.json", "r") as f:
    class_names = json.load(f)

print("📌 Classes:", class_names)

# ===============================
# TEXT TO SPEECH
# ===============================
engine = pyttsx3.init()
engine.setProperty("rate", 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ===============================
# PROCESS IMAGES
# ===============================
for img_name in os.listdir(IMAGE_FOLDER):

    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_FOLDER, img_name)
    print(f"\n🖼 Processing: {img_name}")

    image = cv2.imread(img_path)
    if image is None:
        print("❌ Could not read image")
        continue

    # ===============================
    # PREPROCESS (MATCH TRAINING)
    # ===============================
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ===============================
    # PREDICTION
    # ===============================
    pred_dict = model(img, training=False)
    preds = list(pred_dict.values())[0].numpy()

    print("🔎 Probabilities:", preds[0])

    # Top-3 predictions
    top3 = np.argsort(preds[0])[-3:][::-1]
    print("🏆 Top-3 Predictions:")
    for i in top3:
        print(f"   ₹{class_names[i]} → {preds[0][i]*100:.2f}%")

    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    # ===============================
    # RESULT DECISION
    # ===============================
    if confidence >= CONFIDENCE_THRESHOLD:
        denomination = class_names[class_id]
        result_text = f"₹{denomination} ({confidence*100:.1f}%)"
        speak(f"{denomination} rupees")
    else:
        result_text = "Low confidence"
        speak("Unable to recognize note")

    # ===============================
    # DISPLAY RESULT
    # ===============================
    display = image.copy()
    cv2.putText(
        display,
        result_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255),
        2
    )

    cv2.imshow("Currency Recognition - Image Mode", display)
    cv2.waitKey(0)

cv2.destroyAllWindows()
