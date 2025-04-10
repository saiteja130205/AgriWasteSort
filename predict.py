#predict.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from config import MODEL_PATH, IMAGE_SIZE

# Loading model
model = load_model(MODEL_PATH)

# Loading image
img_path = 'data/test/biodegradable/farmwastetest.png'  # Replace with your test image path
img = image.load_img(img_path, target_size=IMAGE_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predictions
prediction = model.predict(img_array)[0][0]
predicted_class = int(prediction > 0.5)

# Labels
main_labels = ['Biodegradable', 'Non-Biodegradable']
sub_labels = ['Reusable', 'Recyclable']

main_label = main_labels[predicted_class]
sub_label = sub_labels[predicted_class]
confidence = prediction if predicted_class else 1 - prediction
confidence_percent = round(confidence * 100, 2)

# Output
print(f"\n🧾 Prediction: {main_label} (Confidence: {confidence_percent}%)")
print(f"🏷️ {sub_label} Confidence: {confidence_percent}%")
