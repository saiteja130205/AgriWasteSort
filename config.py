# config.py

# Image size
IMG_HEIGHT = 160
IMG_WIDTH = 160

# Training settings
BATCH_SIZE = 32
EPOCHS = 10

# Paths to dataset folders
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Model save path
MODEL_PATH = 'models/agri_model.h5'

# Required input size for MobileNetV2
IMAGE_SIZE = (160, 160)

# Class labels (mapped from folder names)
CLASS_NAMES = ['biodegradable', 'non_biodegradable']
