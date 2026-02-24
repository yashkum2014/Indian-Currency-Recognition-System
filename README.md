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
Currency_Detection/
│
├── data/
│ └── test_images/
│
├── models/
│ └── mobilenetv2/
│ ├── saved_model.pb
│ ├── fingerprint.pb
│ ├── assets/
│ └── variables/
│
├── notebook/
│ ├── train_mobilenetv2.ipynb
│ └── train_efficientnetb0.ipynb
│
├── src/
│ ├── realtime_currency_detection.py
│ └── test_image_trial.py
│
├── class_names.json
├── requirements.txt
└── README.md


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

### 5. Test on Images
python test_image_trial.py

Say "open camera" when prompted.

Press Q to quit.

###🧪 Test on Images
python test_image_trial.py
