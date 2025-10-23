import os
import json
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import load_params, setup_mlflow_experiment_safe, log_params_to_mlflow, create_model_checkpoint_callback
import mlflow
import mlflow.sklearn
import joblib

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

def create_cnn_model(input_shape, params):
    """Создание CNN модели"""
    model = tf.keras.Sequential()
    
    # Сверточные слои
    filters = params.get('filters', [32, 64, 128])
    for i, filter_size in enumerate(filters):
        if i == 0:
            model.add(tf.keras.layers.Conv2D(
                filter_size, 
                params.get('kernel_size', 3), 
                activation='relu', 
                input_shape=input_shape
            ))
        else:
            model.add(tf.keras.layers.Conv2D(
                filter_size, 
                params.get('kernel_size', 3), 
                activation='relu'
            ))
        model.add(tf.keras.layers.MaxPooling2D(params.get('pool_size', 2)))
        model.add(tf.keras.layers.BatchNormalization())
    
    # Полносвязные слои
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(params.get('dense_units', 128), activation='relu'))
    model.add(tf.keras.layers.Dropout(params.get('dropout_rate', 0.5)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Бинарная классификация
    
    # Компиляция
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_transfer_learning_model(input_shape, params):
    """Создание модели Transfer Learning"""
    # Базовая модель
    if params.get('base_model', "MobileNetV2") == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            weights=params.get('weights', "imagenet"),
            include_top=params.get('include_top', False),
            input_shape=input_shape
        )
    
    # Заморозка базовой модели
    base_model.trainable = False
    
    # Добавление собственных слоев
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(params.get('dense_units', 128), activation='relu'),
        tf.keras.layers.Dropout(params.get('dropout_rate', 0.3)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Компиляция
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.0001)),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_random_forest(X_train, y_train, params):
    """Обучение Random Forest модели"""
    # Изменение формы данных для RF
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=params['random_state']
    )
    
    model.fit(X_train_flat, y_train)
    return model

def main():
    """Основная функция обучения"""
    params = load_params()
    model_type = params['training']['current_model']
    model_params = params['models'][model_type]
    
    # Настройка MLflow
    try:
        experiment_id = setup_mlflow_experiment_safe()
        if experiment_id:
            with mlflow.start_run(run_name=f"train_{model_type}") as run:
                print(f"🎯 MLflow Run ID: {run.info.run_id}")
                
                # Логирование параметров
                log_params_to_mlflow(params)
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("dataset", "food_vs_nonfood")
                
                # Загрузка данных
                X_train = np.load("data/processed/train/X_train.npy")
                y_train = np.load("data/processed/train/y_train.npy")
                X_val = np.load("data/processed/validation/X_val.npy")
                y_val = np.load("data/processed/validation/y_val.npy")
                
                print(f"Обучение модели: {model_type}")
                print(f"Размер тренировочных данных: {X_train.shape}")
                
                # Обучение в зависимости от типа модели
                if model_type in ['cnn', 'transfer_learning']:
                    # Нейросетевые модели
                    input_shape = X_train.shape[1:]
                    
                    if model_type == 'cnn':
                        model = create_cnn_model(input_shape, model_params)
                        mlflow.set_tag("architecture", "custom_cnn")
                    else:
                        model = create_transfer_learning_model(input_shape, model_params)
                        mlflow.set_tag("architecture", "transfer_learning")
                    
                    # Callbacks
                    callbacks = [
                        create_model_checkpoint_callback(),
                        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                    ]
                    
                    # Обучение
                    history = model.fit(
                        X_train, y_train,
                        batch_size=params['data']['batch_size'],
                        epochs=model_params['epochs'],
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Логирование истории обучения КАЖДОЙ эпохи
                    for epoch in range(len(history.history['accuracy'])):
                        mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    
                    # Сохранение модели
                    model_path = f"models/{model_type}_model.keras"
                    model.save(model_path)
                    mlflow.keras.log_model(model, "model")
                    
                    # Оценка на валидации
                    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                    
                    # Финальные метрики
                    training_metrics = {
                        "final_val_accuracy": float(val_accuracy),
                        "final_val_loss": float(val_loss),
                        "final_epoch": len(history.history['loss'])
                    }
                    
                else:  # Random Forest
                    mlflow.set_tag("architecture", "random_forest")
                    model = train_random_forest(X_train, y_train, model_params)
                    
                    # Сохранение модели - СОЗДАЕМ ПАПКУ ПРЕЖДЕ
                    os.makedirs("models", exist_ok=True)  # ← ДОБАВЬТЕ ЭТУ СТРОКУ
                    model_path = f"models/{model_type}_model.joblib"
                    import joblib
                    joblib.dump(model, model_path)
                    mlflow.sklearn.log_model(model, "model")
                    
                    # Предсказание на валидации
                    X_val_flat = X_val.reshape(X_val.shape[0], -1)
                    y_val_pred = model.predict(X_val_flat)
                    
                    from sklearn.metrics import accuracy_score
                    
                    val_accuracy = accuracy_score(y_val, y_val_pred)
                    
                    training_metrics = {
                        "final_val_accuracy": float(val_accuracy)
                    }
                
                # Конвертация numpy типов для JSON
                training_metrics = convert_numpy_types(training_metrics)
                
                # Сохранение метрик
                os.makedirs("metrics", exist_ok=True)
                with open("metrics/training_metrics.json", "w") as f:
                    json.dump(training_metrics, f, indent=2)
                
                # Логирование ФИНАЛЬНЫХ метрик
                for metric, value in training_metrics.items():
                    mlflow.log_metric(metric, value)
                
                mlflow.log_artifact("metrics/training_metrics.json")
                
                print(f"✅ Обучение завершено. Validation Accuracy: {training_metrics.get('final_val_accuracy', 'N/A'):.4f}")
        else:
            raise Exception("Не удалось настроить MLflow")
            
    except Exception as e:
        print(f"⚠️ Ошибка MLflow: {e}")
        print("🔧 Запуск обучения без MLflow...")
        
        # Загрузка данных
        X_train = np.load("data/processed/train/X_train.npy")
        y_train = np.load("data/processed/train/y_train.npy")
        X_val = np.load("data/processed/validation/X_val.npy")
        y_val = np.load("data/processed/validation/y_val.npy")
        
        print(f"Обучение модели: {model_type}")
        print(f"Размер тренировочных данных: {X_train.shape}")
        
        # Обучение в зависимости от типа модели
        if model_type in ['cnn', 'transfer_learning']:
            # Нейросетевые модели
            input_shape = X_train.shape[1:]
            
            if model_type == 'cnn':
                model = create_cnn_model(input_shape, model_params)
            else:
                model = create_transfer_learning_model(input_shape, model_params)
            
            # Callbacks
            callbacks = [
                create_model_checkpoint_callback(),
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
            
            # Обучение
            history = model.fit(
                X_train, y_train,
                batch_size=params['data']['batch_size'],
                epochs=model_params['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Сохранение модели
            model_path = f"models/{model_type}_model.keras"
            model.save(model_path)
            
            # Оценка на валидации
            val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            # Финальные метрики
            training_metrics = {
                "final_val_accuracy": float(val_accuracy),
                "final_val_loss": float(val_loss),
                "final_epoch": len(history.history['loss'])
            }
            
        else:  # Random Forest
            mlflow.set_tag("architecture", "random_forest")
            model = train_random_forest(X_train, y_train, model_params)
            
            # СОЗДАЕМ ПАПКУ ПРЕЖДЕ СОХРАНЕНИЯ
            os.makedirs("models", exist_ok=True)
            
            # Сохранение модели
            model_path = f"models/{model_type}_model.joblib"
            import joblib
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(model, "model")
            
            # Предсказание на валидации
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            y_val_pred = model.predict(X_val_flat)
            
            from sklearn.metrics import accuracy_score
            
            val_accuracy = accuracy_score(y_val, y_val_pred)
            
            training_metrics = {
                "final_val_accuracy": float(val_accuracy)
            }
        
        # Конвертация numpy типов для JSON
        training_metrics = convert_numpy_types(training_metrics)
        
        # Сохранение метрик
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/training_metrics.json", "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        print(f"✅ Обучение завершено. Validation Accuracy: {training_metrics.get('final_val_accuracy', 'N/A'):.4f}")

if __name__ == "__main__":
    main()