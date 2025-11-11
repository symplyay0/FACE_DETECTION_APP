from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

# --- App Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Load Model ---
MODEL_PATH = "trained_emotion_model_compressed.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# --- Emotion labels ---
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    name = request.form['name']
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(48, 48), color_mode='grayscale')
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        emotion = emotions[np.argmax(prediction)]

        return render_template(
            'result.html',
            name=name,
            emotion=emotion,
            image_path=filepath
        )

    return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=True)
