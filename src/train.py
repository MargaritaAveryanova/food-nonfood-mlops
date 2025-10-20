import mlflow
import mlflow.tensorflow
import yaml
import tensorflow as tf
import json
import os

# Определяем конфиг здесь
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

def create_simple_cnn(input_shape=(224, 224, 3)):
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_transfer_learning_model(input_shape=(224, 224, 3)):
    """Create transfer learning model with MobileNetV2"""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import layers, models
    
    # Загружаем предобученную модель
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False  # Замораживаем веса
    
    # Создаем новую модель поверх базовой
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_data_generators():
    import tensorflow as tf
    import os
    
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
    return train_generator, val_generator

def train():
    print("🚀 Starting model training...")
    
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow.set_experiment("food_classification")
    
    with mlflow.start_run():
        mlflow.log_params(params['data'])
        mlflow.log_params(params['training'])
        mlflow.log_param("model_type", params['model']['type'])
        
        train_gen, val_gen = create_data_generators()
        
        # Выбираем модель
        if params['model']['type'] == 'simple_cnn':
            model = create_simple_cnn()
            print("✅ Simple CNN model created")
        elif params['model']['type'] == 'simple_cnn_v2':
            model = create_simple_cnn_v2()
            print("✅ Simple CNN V2 model created")
        elif params['model']['type'] == 'mobilenet_v2':
            model = create_transfer_learning_model()
            print("✅ MobileNetV2 transfer learning model created")
        else:
            raise ValueError(f"Unknown model type: {params['model']['type']}")
        
        mlflow.tensorflow.autolog()
        
        print("📚 Training model...")
        history = model.fit(
            train_gen,
            epochs=params['training']['epochs'],
            validation_data=val_gen,
            verbose=1
        )
        
        os.makedirs(Config.MODELS_PATH, exist_ok=True)
        model_path = os.path.join(Config.MODELS_PATH, 'food_classifier.h5')
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        final_accuracy = history.history['val_accuracy'][-1]
        
        train_metrics = {
            "train_accuracy": float(history.history['accuracy'][-1]),
            "train_loss": float(history.history['loss'][-1]),
            "val_accuracy": float(final_accuracy),
            "val_loss": float(history.history['val_loss'][-1])
        }
        
        with open('train_metrics.json', 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        mlflow.log_metric("val_accuracy", final_accuracy)
        
        print(f"✅ Training completed! Final accuracy: {final_accuracy:.4f}")

def create_simple_cnn_v2(input_shape=(224, 224, 3)):
    """Alternative simple CNN architecture - smaller and faster"""
    from tensorflow.keras import layers, models
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train()