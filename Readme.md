# 🧴 Mood & Skin Care AI Detection System

## 📌 Overview
This project is a **deep learning-based web application** that detects a user's **mood** and **skin condition** using CNN models.  

The system allows users to:
- 📸 Upload an image
- 🎥 Use webcam for real-time detection  

After prediction, it provides **personalized skincare and mood improvement suggestions**.

---

## 🚀 Features

### 📸 Image Upload Detection
- Users can upload an image from device
- System processes image and predicts:
  - Mood
  - Skin condition

---

### 🎥 Webcam Real-Time Detection
- Live camera input supported
- Real-time prediction using trained CNN models
- Instant results without uploading image

---

### 😊 Mood Detection
Detects emotions like:
- Happy
- Sad
- Angry
- Neutral

---

### 🧖 Skin Condition Detection
Detects skin types like:
- Acne-prone
- Oily
- Dry
- Clear skin

---

### 💡 Smart Recommendation System
Based on predictions:
- Skincare routine suggestions
- Mood improvement tips
- Lifestyle recommendations

---

## 🧠 Tech Stack

- Python 🐍
- TensorFlow / Keras 🤖
- CNN (Deep Learning)
- Flask 🌐
- OpenCV 📷 (for webcam)
- HTML, CSS 🎨
- JavaScript (basic frontend interaction)

---

## 📂 Project Structure
mood_skincare/
│
├── dataset/
│ ├── mood/
│ │ ├── angry/
│ │ ├── happy/
│ │ ├── sad/
│ │ └── neutral/
│ │
│ └── skin/
│   ├── acne/
│   ├── dry/
│   ├── oily/
│   └── healthy/
│
├── models/
│ ├── mood_model.keras
│ ├── skin_model.keras
│ ├── mood_classes
│ └── skin_classes
│
├── static/
│ └── style.css
│
├── templates/
│ └── index.html
│
├── app.py
├── train_mood.py
├── train_skin.py
├── requirements.txt
└── README.md
|__screenshots