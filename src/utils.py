import os
import yaml
import mlflow
import mlflow.keras
import mlflow.sklearn
import tensorflow as tf
from typing import Dict, Any
import numpy as np

def get_mlflow_tracking_uri():
    """Получение правильного tracking URI для Windows"""
    current_dir = os.path.abspath(".")
    return f"file:///{current_dir.replace(':', '')}/mlruns"

def load_params() -> Dict[str, Any]:
    """Загрузка параметров из YAML файла"""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow_experiment(experiment_name: str = "Food_Classification"):
    """Настройка MLflow эксперимента"""
    # Абсолютный путь для Windows
    current_dir = os.path.abspath(".")
    tracking_uri = f"file:///{current_dir.replace(':', '')}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Создаем папку если не существует
    os.makedirs("mlruns", exist_ok=True)
    
    try:
        # Пробуем получить эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Создаем новый эксперимент
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(current_dir, "mlruns", experiment_name)
            )
            print(f"✅ Создан эксперимент: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Используем существующий эксперимент: {experiment_name}")
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"⚠️ Ошибка при настройке MLflow: {e}")
        print("🔧 Продолжаем без MLflow...")
        return None

def log_params_to_mlflow(params: Dict[str, Any]):
    """Логирование параметров в MLflow"""
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                mlflow.log_param(f"{key}.{sub_key}", sub_value)
        else:
            mlflow.log_param(key, value)

def create_model_checkpoint_callback():
    """Создание callback для сохранения лучшей модели"""
    checkpoint_path = "models/best_model/best_model.keras"
    os.makedirs("models/best_model", exist_ok=True)
    return tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

def setup_mlflow_experiment_safe(experiment_name: str = "Food_Classification"):
    """Безопасная настройка MLflow эксперимента с созданием если не существует"""
    # Абсолютный путь для Windows
    current_dir = os.path.abspath(".")
    tracking_uri = f"file:///{current_dir.replace(':', '')}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    
    # Создаем папку если не существует
    os.makedirs("mlruns", exist_ok=True)
    
    try:
        # Пробуем получить эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Создаем новый эксперимент
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(current_dir, "mlruns", experiment_name)
            )
            print(f"✅ Создан эксперимент: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✅ Используем существующий эксперимент: {experiment_name}")
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)
        return experiment_id
        
    except Exception as e:
        print(f"❌ Критическая ошибка MLflow: {e}")
        return None
    
def convert_numpy_types(obj):
    """Конвертация numpy типов в стандартные Python типы для JSON"""
    if isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj