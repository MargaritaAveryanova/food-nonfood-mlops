import os
import tensorflow as tf

class Config:
    RAW_DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    MODELS_PATH = "models"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 5
    VALIDATION_SPLIT = 0.2
    RANDOM_SEED = 42

def create_data_generators():
    print("📊 Creating data generators...")
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(Config.RAW_DATA_PATH, 'training'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        subset='training',
        seed=Config.RANDOM_SEED
    )
    
    val_generator = val_datagen.flow_from_directory(
        os.path.join(Config.RAW_DATA_PATH, 'training'),
        target_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        seed=Config.RANDOM_SEED
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    
    os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
    
    with open(os.path.join(Config.PROCESSED_DATA_PATH, 'class_indices.txt'), 'w') as f:
        for class_name, class_id in train_generator.class_indices.items():
            f.write(f"{class_name}: {class_id}\n")
    
    return train_generator, val_generator

if __name__ == "__main__":
    train_gen, val_gen = create_data_generators()
    print("✅ Data preprocessing completed!")
