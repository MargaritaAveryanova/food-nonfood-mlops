import os
import mlflow
import json

def fix_mlflow_setup():
    """Исправление настроек MLflow"""
    print("🔧 Исправление настроек MLflow...")
    
    # Явно указываем абсолютный путь для Windows
    current_dir = os.path.abspath(".")
    tracking_uri = f"file:///{current_dir.replace(':', '')}/mlruns"
    
    print(f"📁 Текущая директория: {current_dir}")
    print(f"🔗 Tracking URI: {tracking_uri}")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Проверяем подключение
    try:
        experiments = mlflow.search_experiments()
        print(f"✅ Найдено экспериментов: {len(experiments)}")
        
        for exp in experiments:
            print(f"  📊 Эксперимент: {exp.name}")
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"    🚀 Запусков: {len(runs)}")
            
            for _, run in runs.iterrows():
                print(f"      - {run['run_id']}: {run.get('tags.mlflow.runName', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Ошибка при проверке: {e}")

def create_test_experiment():
    """Создание тестового эксперимента"""
    print("\n🎯 Создание тестового эксперимента...")
    
    mlflow.set_experiment("Food_Classification")
    
    with mlflow.start_run(run_name="test_fix") as run:
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("accuracy", 0.85)
        mlflow.log_metric("loss", 0.15)
        mlflow.set_tag("model_type", "cnn")
        mlflow.set_tag("purpose", "test_fix")
        
        # Создаем тестовый артефакт
        test_data = {"test": "data", "value": 123}
        with open("test_artifact.json", "w") as f:
            json.dump(test_data, f)
        
        mlflow.log_artifact("test_artifact.json")
        
        print(f"✅ Тестовый запуск создан: {run.info.run_id}")

if __name__ == "__main__":
    fix_mlflow_setup()
    create_test_experiment()