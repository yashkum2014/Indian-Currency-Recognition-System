# Indian-Currency-Recognition-System
CNN-based Indian currency denomination recognition system using TensorFlow and transfer learning (MobileNet). Achieved 86.75% accuracy with real-time detection support.

A real-time Indian currency denomination recognition system built using deep learning and transfer learning.  
The system detects currency notes via webcam and provides **voice feedback**, making it accessible for visually impaired users.

---

## Project Overview

This project uses Convolutional Neural Networks (CNN) with transfer learning to classify Indian currency notes.  
Two models were trained and evaluated:

- MobileNetV2 (used for deployment)
- EfficientNetB0 (used for comparison)

MobileNetV2 was selected for real-time inference due to its lightweight architecture and faster performance on local systems.

---

## Features

- Real-time currency detection using webcam  
- Voice output for detected denomination  
- Stable prediction buffer to avoid flickering results  
- TensorFlow SavedModel deployment  
- Transfer learning with MobileNetV2  
- Test image inference support  

---

## Model Performance

- **MobileNetV2 Accuracy:** 86.75%

---

## Tools & Technologies Used

| Tool / Technology | Type | Usage in Project |
|-------------------|------|------------------|
| **Python** | Programming Language | Main language used for AI model development, preprocessing, deployment, and automation |
| **TensorFlow** | Deep Learning Framework | Used for training and implementing deep learning models |
| **Keras** | Deep Learning API | Used for building CNN architectures and transfer learning pipelines |
| **MobileNetV2** | CNN Architecture | Final lightweight model selected for real-time currency classification |
| **EfficientNetB0** | CNN Architecture | Used for comparative evaluation and performance analysis |
| **VGG16** | CNN Architecture | Evaluated during experimentation for image classification |
| **Custom CNN** | Deep Learning Model | Used as baseline CNN architecture for comparison |
| **Transfer Learning** | Machine Learning Technique | Used pretrained models to improve accuracy and reduce training time |
| **OpenCV (cv2)** | Computer Vision Library | Used for webcam integration, image capture, preprocessing, ROI extraction, and real-time detection |
| **NumPy** | Numerical Computing Library | Used for image arrays, preprocessing, normalization, and numerical operations |
| **Google Colab** | Cloud Development Platform | Used for GPU-based model training and experimentation |
| **NVIDIA T4 GPU** | GPU Hardware | Used in Google Colab to speed up deep learning model training |
| **VS Code** | IDE / Code Editor | Used for local development and real-time deployment coding |
| **TensorFlow SavedModel** | Model Export Format | Used for exporting and loading trained models locally |
| **TFSMLayer** | TensorFlow/Keras Layer | Used for safely loading exported TensorFlow SavedModel in Keras 3 |
| **pyttsx3** | Text-to-Speech Library | Used for speech-based denomination announcement |
| **SpeechRecognition** | Speech Processing Library | Used for voice-command recognition |
| **Google Speech Recognition API** | Speech Recognition Service | Used for voice-command-based camera activation |
| **winsound** | Python Audio Module | Used for beep prompts before voice command input |
| **JSON** | Data Format | Used for storing and loading class labels |
| **collections.Counter** | Python Utility | Used for temporal prediction stabilization and majority voting |
| **Data Augmentation** | Deep Learning Technique | Used to improve robustness against lighting, rotation, blur, and occlusions |
| **Image Preprocessing** | Computer Vision Technique | Used for resizing, normalization, and improving model input quality |
| **Temporal Prediction Stabilization** | AI Logic | Used to reduce prediction fluctuation during real-time inference |
| **Confidence Thresholding** | Prediction Logic | Used to improve reliability of real-time predictions |
| **Sliding Window Buffering** | Prediction Stabilization Technique | Used for stable multi-frame prediction consistency |
| **Webcam (Live Camera Feed)** | Input Hardware | Used for real-time currency image acquisition |
| **CPU-based Inference** | Deployment Strategy | Allowed deployment on standard laptops without dedicated GPU |
| **Git** | Version Control Tool | Used for source code version management |
| **GitHub** | Repository Platform | Used for project hosting and code management |
| **TensorFlow/Keras Callbacks** | Training Optimization | Used for `EarlyStopping` and `ReduceLROnPlateau` during training |
| **Adam Optimizer** | Optimization Algorithm | Used for efficient model training convergence |
| **Sparse Categorical Crossentropy** | Loss Function | Used for multi-class currency classification |
| **ROI (Region of Interest)** | Computer Vision Technique | Used to focus currency detection area in webcam feed |
| **Audio Feedback System** | Accessibility Feature | Enabled visually impaired users to hear predictions |
| **Voice Command Activation** | Accessibility Feature | Enabled hands-free interaction with the system |

---

## Project Structure

```bash
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
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Indian-Currency-Recognition-System.git
cd Indian-Currency-Recognition-System
```
### 2. Create virtual environment (optional but recommended)

```bash
python -m venv tf_env
tf_env\Scripts\activate   # Windows
```
### 3. Install dependencies

```bash 
pip install -r requirements.txt
```
### 4. Run Real-Time Detection

```bash
cd src
python realtime_currency_detection.py
```
Say "open camera" when prompted.

Press Q to quit.

### 5. Test on Images

```bash
python test_image_trial.py
```
### 6. Test on Images

```bash
python test_image_trial.py
```
