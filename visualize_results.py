#visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import MODEL_PATH, TEST_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

# Load the trained model
model = load_model(MODEL_PATH)
print(f"✅ Model loaded from: {MODEL_PATH}")

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Predict on test data
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
true_classes = test_data.classes

# Enhanced class labels
class_labels = ['Biodegradable\n(Reusable)', 'Non-Biodegradable\n(Recyclable)']

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix with Reusable/Recyclable Labels")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.savefig("confusion_matrix_visual.png")
plt.show(block=True)

# Sample accuracy and loss values from training
epochs = list(range(1, 11))
train_acc = [0.8832, 0.9299, 0.9324, 0.9393, 0.9412, 0.9468, 0.9449, 0.9505, 0.9497, 0.9519]
val_acc =   [0.8957, 0.8548, 0.8774, 0.8802, 0.8782, 0.8559, 0.9065, 0.8862, 0.8747, 0.9216]
train_loss = [0.2802, 0.1827, 0.1723, 0.1534, 0.1527, 0.1382, 0.1398, 0.1331, 0.1311, 0.1195]
val_loss =   [0.2517, 0.3138, 0.2846, 0.2813, 0.2886, 0.3065, 0.2430, 0.2676, 0.3156, 0.2168]

# Accuracy Graph
plt.figure()
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_graph.png")
plt.show(block=True)

# Loss Graph
plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title("Model Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_graph.png")
plt.show(block=True)
