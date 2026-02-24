# Indian-Currency-Recognition-System
CNN-based Indian currency denomination recognition system using TensorFlow and transfer learning (MobileNet). Achieved 86.75% accuracy with real-time detection support.

A real-time Indian currency denomination recognition system built using deep learning and transfer learning.  
The system detects currency notes via webcam and provides **voice feedback**, making it accessible for visually impaired users.

---

## 🚀 Project Overview

This project uses Convolutional Neural Networks (CNN) with transfer learning to classify Indian currency notes.  
Two models were trained and evaluated:

- MobileNetV2 (used for deployment)
- EfficientNetB0 (used for comparison)

MobileNetV2 was selected for real-time inference due to its lightweight architecture and faster performance on local systems.

---

## ✨ Features

- Real-time currency detection using webcam  
- Voice output for detected denomination  
- Stable prediction buffer to avoid flickering results  
- TensorFlow SavedModel deployment  
- Transfer learning with MobileNetV2  
- Test image inference support  

---

## 📊 Model Performance

- **MobileNetV2 Accuracy:** 86.75%

---

## 🧠 Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- SpeechRecognition  
- pyttsx3  

---

## 📁 Project Structure
Indian-Currency-Recognition-System/
│
├── data/
│ └── test_images/ # Sample images for testing
│
├── models/
│ └── mobilenetv2/ # Trained MobileNetV2 (SavedModel format)
│ ├── saved_model.pb
│ ├── fingerprint.pb
│ ├── assets/
│ └── variables/
│
├── notebook/
│ ├── train_mobilenetv2.ipynb # Training notebook (MobileNetV2)
│ └── train_efficientnetb0.ipynb # Training notebook (EfficientNetB0)
│
├── src/
│ ├── realtime_currency_detection.py # Real-time webcam detection + voice
│ └── test_image_trial.py # Image-based testing script
│
├── class_names.json # Class label mapping
├── requirements.txt # Project dependencies
├── .gitignore # Ignored files (env, cache, etc.)
└── README.md # Project documentation

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Indian-Currency-Recognition-System.git
cd Indian-Currency-Recognition-System
```
### 2. Create virtual environment (optional but recommended)
python -m venv tf_env

tf_env\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run Real-Time Detection
cd src

python realtime_currency_detection.py

Say "open camera" when prompted.

Press Q to quit.

### 5. Test on Images
python test_image_trial.py

### 6. Test on Images
python test_image_trial.py
