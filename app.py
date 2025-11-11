from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sqlite3
from werkzeug.utils import secure_filename

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, static_url_path='/uploads', static_folder='uploads')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------------------
# Load model
# ----------------------------
import os
from tensorflow.keras.models import load_model

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "trained_emotion_model_compressed_fp16.h5")
model = load_model(model_path)


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ----------------------------
# Database setup
# ----------------------------
DB_PATH = "emotion_users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            image_path TEXT,
            detected_emotion TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_user_result(name, image_path, emotion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO users (name, image_path, detected_emotion) VALUES (?, ?, ?)',
                   (name, image_path, emotion))
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# Emotion prediction
# ----------------------------
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1,48,48,1))
    result = model.predict(reshaped)
    return emotion_labels[np.argmax(result)]

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    name = request.form.get('name', 'Anonymous')

    if not file or file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    emotion = predict_emotion(save_path)
    save_user_result(name, save_path, emotion)

    # Note the corrected image URL
    image_url = url_for('static', filename=filename) if os.path.exists(os.path.join('static', filename)) else f"/uploads/{filename}"
    return render_template('result.html', name=name, emotion=emotion, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
