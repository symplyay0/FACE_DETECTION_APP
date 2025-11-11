from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model("trained_emotion_model.h5")

def predict_emotion(face_image):
    # Preprocess image
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    img = np.expand_dims(resized, axis=[0, -1]) / 255.0

    # Predict
    prediction = model.predict(img)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion
