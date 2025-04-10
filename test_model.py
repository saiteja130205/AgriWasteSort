# test_model.py

from tensorflow.keras.models import load_model
from utils.preprocess import load_data
import config

# Choosing which model to load:
# Final saved model after training (default)
model_path = config.MODEL_PATH  # 'models/agri_model.h5'

# Best model saved during training (uncomment to use this)
# model_path = 'models/best_model.keras'

# Loading the model
print(f"Loading model from: {model_path}")
model = load_model(model_path)

# Loading the test dataset
print("Loading test dataset...")
_, test_data = load_data()  # Only need the test data

# Evaluating the model
print("\nEvaluating model on test set...")
loss, accuracy = model.evaluate(test_data)

# Printing the results
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
