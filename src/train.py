import mlflow
import mlflow.tensorflow
import yaml
import tensorflow as tf
import json
import os
from src.data_preprocessing import create_data_generators
from src.models.cnn_simple import create_simple_cnn
from src.models.transfer_learning import create_transfer_learning_model
from src.config import Config

def train():
    print("🚀 Starting model training...")
    
    # Загружаем параметры
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Настраиваем MLflow
    mlflow.set_experiment("food_classification")
    
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params(params['data'])
        mlflow.log_params(params['training'])
        mlflow.log_param("model_type", params['model']['type'])
        
        # Загружаем данные
        train_gen, val_gen = create_data_generators()
        
        # Создаем модель
        if params['model']['type'] == 'simple_cnn':
            model = create_simple_cnn()
            print("✅ Simple CNN model created")
        elif params['model']['type'] == 'mobilenet_v2':
            model = create_transfer_learning_model()
            print("✅ MobileNetV2 transfer learning model created")
        else:
            raise ValueError(f"Unknown model type: {params['model']['type']}")
        
        # Логируем модель
        mlflow.tensorflow.autolog()
        
        # Обучаем модель
        print("📚 Training model...")
        history = model.fit(
            train_gen,
            epochs=params['training']['epochs'],
            validation_data=val_gen,
            verbose=1
        )
        
        # Сохраняем модель
        os.makedirs(Config.MODELS_PATH, exist_ok=True)
        model_path = os.path.join(Config.MODELS_PATH, 'food_classifier.h5')
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Сохраняем тренировочные метрики
        final_accuracy = history.history['val_accuracy'][-1]
        final_loss = history.history['val_loss'][-1]
        
        train_metrics = {
            "train_accuracy": float(history.history['accuracy'][-1]),
            "train_loss": float(history.history['loss'][-1]),
            "val_accuracy": float(final_accuracy),
            "val_loss": float(final_loss)
        }
        
        with open('train_metrics.json', 'w') as f:
            json.dump(train_metrics, f, indent=2)
        
        mlflow.log_metric("val_accuracy", final_accuracy)
        mlflow.log_metric("val_loss", final_loss)
        
        print(f"✅ Training completed! Final accuracy: {final_accuracy:.4f}")
        print(f"✅ Model saved to: {model_path}")

if __name__ == "__main__":
    train()