#realtime_predict_cv2.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config

# Loading the trained model
model = load_model('models/best_model.keras')

# Starting webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera

# Confidence threshold
threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Copy the frame for processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Determine label and confidence
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

    # Display result
    display_text = f"{label} ({tag}: {confidence * 100:.2f}%)"
    cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow("Real-Time Prediction", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
