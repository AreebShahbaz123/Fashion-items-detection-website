import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Unzip dataset.zip
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('fashion_mnist_data')

# Load CSV files
train_df = pd.read_csv('fashion_mnist_data/train.csv')
test_df = pd.read_csv('fashion_mnist_data/test.csv')

# Separate features and labels
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize pixel values (0 to 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape to (28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, 10)
y_test_cat = tf.keras.utils.to_categorical(y_test, 10)

# Split train into train/validation
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train_cat, test_size=0.1, random_state=42
)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train_final)

# Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks to prevent overfitting
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint('fashion_mnist_model.h5', save_best_only=True)

# Train model
history = model.fit(
    datagen.flow(X_train_final, y_train_final, batch_size=64),
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

# Evaluate model on test data
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Save the final model (optional if using checkpoint)
model.save('fashion_mnist_model.h5')
