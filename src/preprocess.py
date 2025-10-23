import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from utils import load_params, setup_mlflow_experiment_safe
import mlflow

def load_and_preprocess_data(data_path: str, img_size: tuple = (224, 224)):
    """Загрузка и предобработка данных"""
    categories = ['food', 'non_food']
    images = []
    labels = []
    
    print("Загрузка данных...")
    
    for label, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        if not os.path.exists(category_path):
            print(f"Предупреждение: директория {category_path} не существует!")
            continue
            
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"Предупреждение: в {category_path} нет изображений!")
            continue
            
        for img_file in image_files:
            img_path = os.path.join(category_path, img_file)
            
            # Загрузка и предобработка изображения
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(label)
    
    if len(images) == 0:
        raise ValueError("Не загружено ни одного изображения! Проверьте путь к данным.")
    
    return np.array(images), np.array(labels)

def create_data_generators(X_train, y_train, X_val, y_val, batch_size: int = 32):
    """Создание генераторов данных с аугментацией"""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator

def main():
    """Основная функция предобработки"""
    params = load_params()
    data_params = params['data']
    
    # Настройка MLflow
    try:
        experiment_id = setup_mlflow_experiment_safe()
        if experiment_id:
            with mlflow.start_run(run_name="data_preprocessing"):
                mlflow.log_params({
                    "image_size": data_params['image_size'],
                    "batch_size": data_params['batch_size'],
                    "validation_split": data_params['validation_split']
                })
                
                # Загрузка данных
                data_path = "data/raw"
                X, y = load_and_preprocess_data(data_path, tuple(data_params['image_size']))
                
                # Разделение на train/validation/test
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, 
                    test_size=data_params['test_split'], 
                    random_state=42, 
                    stratify=y
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp,
                    test_size=data_params['validation_split'],
                    random_state=42,
                    stratify=y_temp
                )
                
                # Создание директорий
                os.makedirs("data/processed/train", exist_ok=True)
                os.makedirs("data/processed/validation", exist_ok=True)
                os.makedirs("data/processed/test", exist_ok=True)
                
                # Сохранение обработанных данных
                np.save("data/processed/train/X_train.npy", X_train)
                np.save("data/processed/train/y_train.npy", y_train)
                np.save("data/processed/validation/X_val.npy", X_val)
                np.save("data/processed/validation/y_val.npy", y_val)
                np.save("data/processed/test/X_test.npy", X_test)
                np.save("data/processed/test/y_test.npy", y_test)
                
                # Логирование информации о данных
                mlflow.log_metrics({
                    "train_samples": len(X_train),
                    "validation_samples": len(X_val),
                    "test_samples": len(X_test),
                    "food_ratio_train": float(np.mean(y_train)),
                    "food_ratio_test": float(np.mean(y_test))
                })
                
                print(f"Данные предобработаны и сохранены:")
                print(f"  Train: {len(X_train)} samples")
                print(f"  Validation: {len(X_val)} samples")
                print(f"  Test: {len(X_test)} samples")
        else:
            raise Exception("Не удалось настроить MLflow")
            
    except Exception as e:
        print(f"⚠️ Ошибка MLflow: {e}")
        print("🔧 Запуск предобработки без MLflow...")
        
        # Загрузка данных без MLflow
        data_path = "data/raw"
        X, y = load_and_preprocess_data(data_path, tuple(data_params['image_size']))
        
        # Разделение на train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=data_params['test_split'], 
            random_state=42, 
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=data_params['validation_split'],
            random_state=42,
            stratify=y_temp
        )
        
        # Создание директорий
        os.makedirs("data/processed/train", exist_ok=True)
        os.makedirs("data/processed/validation", exist_ok=True)
        os.makedirs("data/processed/test", exist_ok=True)
        
        # Сохранение обработанных данных
        np.save("data/processed/train/X_train.npy", X_train)
        np.save("data/processed/train/y_train.npy", y_train)
        np.save("data/processed/validation/X_val.npy", X_val)
        np.save("data/processed/validation/y_val.npy", y_val)
        np.save("data/processed/test/X_test.npy", X_test)
        np.save("data/processed/test/y_test.npy", y_test)
        
        print(f"Данные предобработаны и сохранены:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")

def run_preprocessing_without_mlflow(params):
    """Запуск предобработки без MLflow"""
    data_params = params['data']
    
    # Загрузка и обработка данных
    data_path = "data/raw"
    X, y = load_and_preprocess_data(data_path, tuple(data_params['image_size']))
    
    # Разделение на train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=data_params['test_split'], 
        random_state=42, 
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=data_params['validation_split'],
        random_state=42,
        stratify=y_temp
    )
    
    # Создание директорий
    os.makedirs("data/processed/train", exist_ok=True)
    os.makedirs("data/processed/validation", exist_ok=True)
    os.makedirs("data/processed/test", exist_ok=True)
    
    # Сохранение обработанных данных
    np.save("data/processed/train/X_train.npy", X_train)
    np.save("data/processed/train/y_train.npy", y_train)
    np.save("data/processed/validation/X_val.npy", X_val)
    np.save("data/processed/validation/y_val.npy", y_val)
    np.save("data/processed/test/X_test.npy", X_test)
    np.save("data/processed/test/y_test.npy", y_test)
    
    print(f"Данные предобработаны и сохранены:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

if __name__ == "__main__":
    main()