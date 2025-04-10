#visual_predict_cv2.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config

# Loading the trained model
model = load_model('models/best_model.keras')

# Path to the image for testing
img_path = 'data/test/biodegradable/farmwastetest.png'

# Loading image using OpenCV (BGR format)
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found at path: {img_path}")

# Converting BGR to RGB (as used during training)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resizing image
img_resized = cv2.resize(img_rgb, (config.IMG_WIDTH, config.IMG_HEIGHT))

# Normalizing and adding batch dimension
img_array = img_resized / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)[0][0]
threshold = 0.6

# Classifying with reusable/recyclable tag
if prediction > threshold:
    label = "Non-Biodegradable"
    tag = "Recyclable"
    confidence = prediction
    color = (0, 0, 255)  # Red
elif prediction < 1 - threshold:
    label = "Biodegradable"
    tag = "Reusable"
    confidence = 1 - prediction
    color = (0, 255, 0)  # Green
else:
    label = "Uncertain"
    tag = ""
    confidence = max(prediction, 1 - prediction)
    color = (0, 255, 255)  # Yellow

# Overlay text
display_text = f"{label} ({tag}: {confidence * 100:.2f}%)"
cv2.putText(img_bgr, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Show the image
cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
