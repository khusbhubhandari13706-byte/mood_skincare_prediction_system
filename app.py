import os
import json
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from io import BytesIO
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)
# =========================
# APP CONFIG
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
print("Loading models...")

mood_model = tf.keras.models.load_model("models/mood_model.keras")
skin_model = tf.keras.models.load_model("models/skin_model.keras")

with open("models/mood_classes.json", "r") as f:
    mood_classes = json.load(f)

with open("models/skin_classes.json", "r") as f:
    skin_classes = json.load(f)

mood_labels = {v: k for k, v in mood_classes.items()}
skin_labels = {v: k for k, v in skin_classes.items()}

print("Models loaded successfully!")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# =========================
# SUGGESTIONS
# =========================

MOOD_TIPS = {
    "happy": [
        "Keep doing activities that make you happy.",
        "Share your positivity with others.",
        "Maintain your healthy routine."
    ],
    "sad": [
        "Take a short walk outdoors.",
        "Talk with friends or family.",
        "Listen to relaxing music."
    ],
    "angry": [
        "Take deep breaths.",
        "Pause before reacting.",
        "Try light exercise."
    ],
    "neutral": [
        "Stay productive.",
        "Set small goals for the day.",
        "Keep a balanced routine."
    ]
}

SKIN_TIPS = {
    "acne": {
        "tips": [
            "Use gentle cleanser",
            "Avoid oily food"
        ],
        "products": [
            "Clean & Clear Face Wash",
            "The Derma Co Salicylic Acid Serum",
            "Minimalist Niacinamide Serum"
        ]
    },
    "dry": {
        "tips": [
            "Use moisturizer daily",
            "Drink more water"
        ],
        "products": [
            "Cetaphil Moisturizer",
            "Nivea Soft Cream",
            "Neutrogena Hydro Boost"
        ]
    },
    "oily": {
        "tips": [
            "Wash face twice daily",
            "Use oil-free products"
        ],
        "products": [
            "Garnier Oil Clear Face Wash",
            "The Face Shop Oil Control Cream"
        ]
    },
    "healthy": {
        "tips": [
            "Maintain routine",
            "Use sunscreen"
        ],
        "products": [
            "Sunscreen SPF 50",
            "Vitamin C Serum"
        ]
    }
}
# =========================
# IMAGE PREPROCESSING
# =========================

def detect_face(image):
    """
    Detect and crop face.
    Returns cropped face image.
    If no face found, returns original image.
    """

    img_np = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) > 0:

        # Largest face
        x, y, w, h = max(
            faces,
            key=lambda f: f[2] * f[3]
        )

        face = img_np[y:y+h, x:x+w]

        return Image.fromarray(face)

    return image


def preprocess_mood(image):
    """
    Mood model:
    grayscale
    48x48
    """

    img = image.convert("L")
    img = img.resize((48, 48))

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


def preprocess_skin(image):
    """
    Skin model:
    RGB
    48x48
    """

    img = image.convert("RGB")

# Contrast improve
    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.bilateralFilter(img, 7, 50, 50)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img)
    img = img.resize((96,96))

    img = np.array(img).astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    return img


# =========================
# PREDICTION FUNCTION
# =========================

def predict_both(image):

    face_image = detect_face(image)

    mood_input = preprocess_mood(face_image)
    skin_input = preprocess_skin(face_image)

    mood_pred = mood_model.predict(mood_input, verbose=0)[0]
    skin_pred = skin_model.predict(skin_input, verbose=0)[0]

    print("\n========================")
    print("MOOD PROBS:")
    print(mood_pred)

    print("SKIN PROBS:")
    print(skin_pred)
    print("========================\n")

    mood_idx = np.argmax(mood_pred)
    skin_idx = np.argmax(skin_pred)

    mood_name = mood_labels[mood_idx]
    skin_name = skin_labels[skin_idx]

    mood_confidence = float(np.max(mood_pred) * 100)
    skin_confidence = float(np.max(skin_pred) * 100)

    return {
    "mood": mood_name.capitalize(),
    "mood_confidence": round(mood_confidence, 2),

    "skin": skin_name.capitalize(),
    "skin_confidence": round(skin_confidence, 2),

    "mood_tips": MOOD_TIPS.get(mood_name, []),

    "skin_tips": SKIN_TIPS.get(skin_name, {}).get("tips", []),

    "skin_products": SKIN_TIPS.get(skin_name, {}).get("products", [])
}

# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


# =========================
# UPLOAD IMAGE
# =========================

@app.route("/predict_upload", methods=["POST"])
def predict_upload():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        })

    file = request.files["image"]

    try:
        image = Image.open(file.stream)

        result = predict_both(image)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })


# =========================
# WEBCAM IMAGE
# =========================

@app.route("/predict_webcam", methods=["POST"])
def predict_webcam():

    try:

        data = request.json.get("image")

        if not data:
            return jsonify({
                "success": False,
                "message": "No image data received"
            })

        encoded = data.split(",")[1]

        image_bytes = base64.b64decode(encoded)

        image = Image.open(BytesIO(image_bytes))

        result = predict_both(image)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        })


# =========================
# RUN
# =========================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )