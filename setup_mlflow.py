import os
import mlflow

def setup_mlflow_correctly():
    """Правильная настройка MLflow"""
    print("🔧 Настройка MLflow...")
    
    # Абсолютный путь для Windows
    current_dir = os.path.abspath(".")
    tracking_uri = f"file:///{current_dir.replace(':', '')}/mlruns"
    print(f"📁 Tracking URI: {tracking_uri}")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Создаем папку mlruns если не существует
    os.makedirs("mlruns", exist_ok=True)
    
    # Создаем эксперимент
    experiment_name = "Food_Classification"
    
    try:
        # Пробуем получить эксперимент
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            # Создаем новый эксперимент
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(current_dir, "mlruns", experiment_name)
            )
            print(f"✅ Создан эксперимент: {experiment_name} (ID: {experiment_id})")
        else:
            print(f"✅ Эксперимент уже существует: {experiment_name} (ID: {experiment.experiment_id})")
            
        # Устанавливаем эксперимент
        mlflow.set_experiment(experiment_name)
        
        # Тестовый запуск
        with mlflow.start_run(run_name="test_setup"):
            mlflow.log_param("test", "success")
            mlflow.log_metric("accuracy", 0.95)
            print("✅ Тестовый запуск создан успешно!")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    setup_mlflow_correctly()