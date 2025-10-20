import os

class Config:
    # Paths
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    MODELS_PATH = "models"
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    
    # Training
    EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42