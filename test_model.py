import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

# Label mapping
labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Load trained model
model = load_model('fashion_mnist_model.h5')

# Load and preprocess image
def preprocess_image(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize while maintaining aspect ratio
    h, w = img.shape
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)
    elif w > h:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=255)

    img = cv2.resize(img, (28, 28))

    # Invert colors if background is white
    if np.mean(img) > 127:
        img = 255 - img

    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

# Path to your image
image_path = 'shoe.jpg'  # Replace with your own image path
img = preprocess_image(image_path)

# Predict
predictions = model.predict(img)
predicted_label = np.argmax(predictions)

# Print result
print(f"Predicted class: {predicted_label} ({labels_map[predicted_label]})")
