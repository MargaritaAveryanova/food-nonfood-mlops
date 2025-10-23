import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_params, setup_mlflow_experiment_safe, convert_numpy_types
import mlflow
import mlflow.sklearn
import joblib

def load_model(model_type: str):
    """Загрузка модели в зависимости от типа"""
    if model_type in ['cnn', 'transfer_learning']:
        model_path = f"models/{model_type}_model.keras"
        return tf.keras.models.load_model(model_path)
    else:  # random_forest
        model_path = f"models/{model_type}_model.joblib"
        return joblib.load(model_path)

def convert_numpy_types(obj):
    """Конвертация numpy типов в стандартные Python типы для JSON"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def evaluate_model(model, X_test, y_test, model_type: str):
    """Оценка модели"""
    if model_type in ['cnn', 'transfer_learning']:
        # Для нейросетевых моделей - получаем только 2 значения
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Вычисляем остальные метрики вручную
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1_score": float(test_f1)
        }
    else:
        # Для Random Forest
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_flat)
        y_pred_proba = model.predict_proba(X_test_flat)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "test_f1_score": float(f1_score(y_test, y_pred, zero_division=0))
        }
    
    # ROC-AUC
    from sklearn.metrics import roc_auc_score
    try:
        if model_type != 'random_forest':
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        metrics["test_roc_auc"] = float(roc_auc)
    except:
        metrics["test_roc_auc"] = 0.0
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Построение матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Food', 'Food'],
                yticklabels=['Non-Food', 'Food'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Сохранение
    os.makedirs("plots", exist_ok=True)
    plot_path = f"plots/confusion_matrix_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path

def plot_roc_curve(y_true, y_pred_proba, model_name: str):
    """Построение ROC кривой"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    # Сохранение
    plot_path = f"plots/roc_curve_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path, roc_auc

def main():
    """Основная функция оценки"""
    params = load_params()
    model_type = params['training']['current_model']
    
    # Настройка MLflow
    try:
        experiment_id = setup_mlflow_experiment_safe()
        if experiment_id:
            with mlflow.start_run(run_name=f"evaluate_{model_type}"):
                # Загрузка тестовых данных
                X_test = np.load("data/processed/test/X_test.npy")
                y_test = np.load("data/processed/test/y_test.npy")
                
                print(f"Оценка модели: {model_type}")
                print(f"Размер тестовых данных: {X_test.shape}")
                
                # Загрузка модели
                model = load_model(model_type)
                
                # Оценка модели
                metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, model_type)
                
                # Визуализации
                cm_plot_path = plot_confusion_matrix(y_test, y_pred, model_type)
                roc_plot_path, roc_auc = plot_roc_curve(y_test, y_pred_proba, model_type)
                
                # Обновление метрик
                metrics["test_roc_auc"] = float(roc_auc)
                
                # Отчет классификации
                class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                
                # Конвертация всех numpy типов
                metrics = convert_numpy_types(metrics)
                class_report = convert_numpy_types(class_report)
                
                # Сохранение метрик и отчетов
                os.makedirs("metrics", exist_ok=True)
                
                with open("metrics/test_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)
                
                with open("metrics/classification_report.json", "w") as f:
                    json.dump(class_report, f, indent=2)
                
                # Логирование в MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_artifact("metrics/test_metrics.json")
                mlflow.log_artifact("metrics/classification_report.json")
                mlflow.log_artifact(cm_plot_path)
                mlflow.log_artifact(roc_plot_path)
                
                # Вывод результатов
                print("\n" + "="*50)
                print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
                print("="*50)
                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")
                print("="*50)
                
                print("\nОТЧЕТ КЛАССИФИКАЦИИ:")
                print(classification_report(y_test, y_pred, target_names=['Non-Food', 'Food'], zero_division=0))
        else:
            raise Exception("Не удалось настроить MLflow")
            
    except Exception as e:
        print(f"⚠️ Ошибка MLflow: {e}")
        print("🔧 Запуск оценки без MLflow...")
        
        # Загрузка тестовых данных
        X_test = np.load("data/processed/test/X_test.npy")
        y_test = np.load("data/processed/test/y_test.npy")
        
        print(f"Оценка модели: {model_type}")
        print(f"Размер тестовых данных: {X_test.shape}")
        
        # Загрузка модели
        model = load_model(model_type)
        
        # Оценка модели
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, model_type)
        
        # Визуализации
        cm_plot_path = plot_confusion_matrix(y_test, y_pred, model_type)
        roc_plot_path, roc_auc = plot_roc_curve(y_test, y_pred_proba, model_type)
        
        # Обновление метрик
        metrics["test_roc_auc"] = float(roc_auc)
        
        # Отчет классификации
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Конвертация всех numpy типов
        metrics = convert_numpy_types(metrics)
        class_report = convert_numpy_types(class_report)
        
        # Сохранение метрик и отчетов
        os.makedirs("metrics", exist_ok=True)
        
        with open("metrics/test_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        with open("metrics/classification_report.json", "w") as f:
            json.dump(class_report, f, indent=2)
        
        # Вывод результатов
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
        print("="*50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
        print("\nОТЧЕТ КЛАССИФИКАЦИИ:")
        print(classification_report(y_test, y_pred, target_names=['Non-Food', 'Food'], zero_division=0))

if __name__ == "__main__":
    main()