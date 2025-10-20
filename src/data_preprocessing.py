import tensorflow as tf
import os
from src.config import Config

def create_data_generators():
    """Create train and validation data generators"""
    print("📊 Creating data generators...")
    
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Only rescaling for validation
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=Config.VALIDATION_SPLIT
    )
    
    # Create generators
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
    print(f"✅ Classes: {train_generator.class_indices}")
    
    # Создаем папку processed для вывода
    os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
    
    # Сохраняем информацию о классах
    with open(os.path.join(Config.PROCESSED_DATA_PATH, 'class_indices.txt'), 'w') as f:
        for class_name, class_id in train_generator.class_indices.items():
            f.write(f"{class_name}: {class_id}\n")
    
    return train_generator, val_generator

if __name__ == "__main__":
    train_gen, val_gen = create_data_generators()
    print("✅ Data preprocessing completed successfully!")