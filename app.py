from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sqlite3
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ----------------------------
# CONFIG
# ----------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_emotion_model_compressed.h5')
model = load_model(MODEL_PATH)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ----------------------------
# DATABASE SETUP
# ----------------------------
DB_PATH = 'emotion_users.db'

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
    cursor.execute('''
        INSERT INTO users (name, image_path, detected_emotion)
        VALUES (?, ?, ?)
    ''', (name, image_path, emotion))
    conn.commit()
    conn.close()

init_db()

# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1,48,48,1))
    result = model.predict(reshaped)
    emotion = emotion_labels[np.argmax(result)]
    return emotion

# ----------------------------
# ROUTES
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    name = request.form.get('name', 'Anonymous')
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        emotion = predict_emotion(save_path)
        save_user_result(name, filename, emotion)  # save only filename for easier serving
        # Use url_for to serve uploaded image
        image_url = url_for('uploaded_file', filename=filename)
        return render_template('result.html', name=name, emotion=emotion, image_url=image_url)
    return redirect('/')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/users')
def users():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    html = """
    <html>
    <head>
        <title>Emotion App Users</title>
        <style>
            body { font-family: Arial; background-color: #0b2e13; color: #fff; text-align:center; }
            table { margin: auto; border-collapse: collapse; width: 80%; background: #07210a; }
            th, td { border: 1px solid #00ff99; padding: 10px; }
            th { background-color: #00ff99; color: #0b2e13; }
            tr:nth-child(even) { background-color: #0b2e13; }
            a { color: #00ff99; text-decoration: none; }
        </style>
    </head>
    <body>
        <h2>📋 Users Who Used the Emotion App</h2>
        <table>
            <tr><th>ID</th><th>Name</th><th>Image</th><th>Emotion</th><th>Time</th></tr>
    """
    for row in rows:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td><a href='/uploads/{row[2]}' target='_blank'>View</a></td><td>{row[3]}</td><td>{row[4]}</td></tr>"
    html += "</table></body></html>"
    return html

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)
