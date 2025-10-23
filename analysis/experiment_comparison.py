import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import os

# Настройка MLflow
mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace(':', '')}")

def compare_experiments():
    """Сравнение всех экспериментов"""
    print("📊 Сравнение экспериментов")
    
    # Получаем все запуски
    runs = mlflow.search_runs()
    
    if runs.empty:
        print("❌ Нет данных для сравнения")
        return
    
    # Выбираем важные колонки
    important_metrics = ['final_val_accuracy', 'final_val_precision', 'final_val_recall', 'final_val_f1']
    available_metrics = [m for m in important_metrics if m in runs.columns]
    
    # Создаем таблицу сравнения
    comparison = runs[['tags.mlflow.runName'] + available_metrics].copy()
    comparison.columns = ['Model'] + [col.replace('final_val_', '').title() for col in available_metrics]
    
    # Убираем NaN
    comparison = comparison.fillna(0)
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Сохраняем в файл
    os.makedirs("analysis", exist_ok=True)
    comparison.to_csv("analysis/model_comparison.csv", index=False)
    print(f"\n✅ Результаты сохранены в analysis/model_comparison.csv")
    
    # Создаем визуализацию
    if len(available_metrics) > 0:
        plt.figure(figsize=(12, 8))
        
        metrics_df = runs[available_metrics].fillna(0)
        metrics_df.index = runs['tags.mlflow.runName']
        
        metrics_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Сравнение метрик моделей')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('analysis/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ График сохранен в analysis/metrics_comparison.png")

if __name__ == "__main__":
    compare_experiments()