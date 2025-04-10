# train_model.py

from utils.preprocess import load_data
from utils.model_utils import build_model
import config

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Loading the dataset
train_data, test_data = load_data()

# Building the model
model = build_model(input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))

# Setting up callbacks

# Early stopping if val_loss doesn't improve for 7 consecutive epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

# Saving the best model based on validation loss
checkpoint = ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Training the model
print("Training started...\n")
model.fit(
    train_data,
    validation_data=test_data,
    epochs=config.EPOCHS,
    callbacks=[early_stop, checkpoint]
)
print("\nTraining complete!")

# Saving the final model
model.save(config.MODEL_PATH)
print(f"Model saved to {config.MODEL_PATH}")
