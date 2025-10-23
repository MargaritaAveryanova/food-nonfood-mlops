import os
import numpy as np
from PIL import Image
import cv2

def create_sample_dataset():
    """Создание тестового датасета"""
    categories = ['food', 'non_food']
    
    for category in categories:
        category_path = f"data/raw/{category}"
        os.makedirs(category_path, exist_ok=True)
        
        # Создаем 20 тестовых изображений для каждой категории
        for i in range(20):
            # Создаем случайное изображение
            if category == 'food':
                # "Еда" - более теплые цвета
                img = np.random.randint(100, 255, (224, 224, 3), dtype=np.uint8)
                img = cv2.addWeighted(img, 0.7, np.zeros((224,224,3), dtype=np.uint8), 0.3, 0)
            else:
                # "Не еда" - более холодные цвета
                img = np.random.randint(0, 150, (224, 224, 3), dtype=np.uint8)
            
            # Сохраняем изображение
            img_path = os.path.join(category_path, f"{category}_{i:02d}.jpg")
            cv2.imwrite(img_path, img)
            print(f"Создано: {img_path}")

if __name__ == "__main__":
    create_sample_dataset()
    print("Тестовый датасет создан!")