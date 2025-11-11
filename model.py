# model.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Dataset paths
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
image_size = (48, 48)
batch_size = 32
epochs = 20
model = load_model("trained_emotion_model.h5")

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=image_size,
                                                    color_mode='grayscale', batch_size=batch_size,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=image_size,
                                                color_mode='grayscale', batch_size=batch_size,
                                                class_mode='categorical')

# Model architecture
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# Save model
model.save('trained_emotion_model.h5')
print("Model saved as trained_emotion_model.h5")
