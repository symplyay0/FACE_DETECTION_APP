import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import numpy as np

MODEL_INPUT = "trained_emotion_model.h5"
MODEL_OUTPUT = "trained_emotion_model_compressed_fp16.h5"

print("Loading model...")
model = load_model(MODEL_INPUT)

# Convert model weights to float16
for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
        new_weights = [np.array(w, dtype=np.float16) for w in weights]
        layer.set_weights(new_weights)

print("Saving compressed model...")
model.save(MODEL_OUTPUT, include_optimizer=False)

size = os.path.getsize(MODEL_OUTPUT) / (1024 * 1024)
print(f"✅ Model compressed successfully! New size: {size:.2f} MB")
