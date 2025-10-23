import os
import numpy as np
import albumentations as A
from PIL import Image
import cv2
from utils import load_params, setup_mlflow_experiment_safe
import mlflow

def augment_images(images, labels, augmentations_per_image: int = 3):
    """Аугментация изображений"""
    params = load_params()
    aug_params = params['augmentation']
    
    # Определение трансформаций
    transform = A.Compose([
        A.Rotate(limit=aug_params['rotation_range'], p=0.7),
        A.HorizontalFlip(p=aug_params['horizontal_flip']),
        A.ShiftScaleRotate(
            shift_limit=aug_params['width_shift_range'],
            scale_limit=aug_params['zoom_range'],
            rotate_limit=0,
            p=0.7
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug_params['brightness_range'][1] - 1,
            contrast_limit=0.2,
            p=0.5
        ),
    ])
    
    augmented_images = []
    augmented_labels = []
    
    print("Аугментация данных...")
    for i, (image, label) in enumerate(zip(images, labels)):
        # Конвертация обратно в uint8 для albumentations
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Добавление оригинального изображения
        augmented_images.append(image)
        augmented_labels.append(label)
        
        # Создание аугментированных версий
        for j in range(augmentations_per_image):
            augmented = transform(image=image_uint8)
            augmented_image = augmented['image'].astype(np.float32) / 255.0
            augmented_images.append(augmented_image)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels)

def main():
    """Основная функция аугментации"""
    # Настройка MLflow
    try:
        experiment_id = setup_mlflow_experiment_safe()
        if experiment_id:
            with mlflow.start_run(run_name="data_augmentation"):
                # Загрузка тренировочных данных
                X_train = np.load("data/processed/train/X_train.npy")
                y_train = np.load("data/processed/train/y_train.npy")
                
                # Логирование параметров аугментации
                params = load_params()
                mlflow.log_params(params['augmentation'])
                
                # Аугментация данных
                X_augmented, y_augmented = augment_images(X_train, y_train)
                
                # Сохранение аугментированных данных
                os.makedirs("data/augmented", exist_ok=True)
                np.save("data/augmented/X_augmented.npy", X_augmented)
                np.save("data/augmented/y_augmented.npy", y_augmented)
                
                # Логирование результатов аугментации
                mlflow.log_metrics({
                    "original_samples": len(X_train),
                    "augmented_samples": len(X_augmented),
                    "augmentation_ratio": float(len(X_augmented) / len(X_train))
                })
                
                print(f"Аугментация завершена:")
                print(f"  Оригинальные данные: {len(X_train)} samples")
                print(f"  После аугментации: {len(X_augmented)} samples")
                print(f"  Увеличение в {len(X_augmented) / len(X_train):.2f} раз")
        else:
            raise Exception("Не удалось настроить MLflow")
            
    except Exception as e:
        print(f"⚠️ Ошибка MLflow: {e}")
        print("🔧 Запуск аугментации без MLflow...")
        
        # Загрузка тренировочных данных
        X_train = np.load("data/processed/train/X_train.npy")
        y_train = np.load("data/processed/train/y_train.npy")
        
        # Аугментация данных
        X_augmented, y_augmented = augment_images(X_train, y_train)
        
        # Сохранение аугментированных данных
        os.makedirs("data/augmented", exist_ok=True)
        np.save("data/augmented/X_augmented.npy", X_augmented)
        np.save("data/augmented/y_augmented.npy", y_augmented)
        
        print(f"Аугментация завершена:")
        print(f"  Оригинальные данные: {len(X_train)} samples")
        print(f"  После аугментации: {len(X_augmented)} samples")
        print(f"  Увеличение в {len(X_augmented) / len(X_train):.2f} раз")

if __name__ == "__main__":
    main()