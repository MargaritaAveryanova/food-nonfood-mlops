import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil

class Config:
    RAW_DATA_PATH = "data/raw"
    AUGMENTED_DATA_PATH = "data/augmented"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

def augment_images():
    """Create augmented versions of the dataset"""
    print("🔄 Creating augmented dataset...")
    
    # Удаляем старую папку если существует
    if os.path.exists(Config.AUGMENTED_DATA_PATH):
        shutil.rmtree(Config.AUGMENTED_DATA_PATH)
    
    # Создаем структуру папок для аугментированных данных
    aug_folders = ['training/0_food', 'training/1_non_food', 'validation/0_food', 'validation/1_non_food']
    for folder in aug_folders:
        os.makedirs(f'{Config.AUGMENTED_DATA_PATH}/{folder}', exist_ok=True)
    
    def apply_augmentations(image_path, output_path, count=2):
        """Apply various augmentations to an image"""
        try:
            original = Image.open(image_path)
            
            for i in range(count):
                img = original.copy()
                
                # Случайные аугментации
                augment_type = random.choice(['rotate', 'flip', 'brightness', 'contrast'])
                
                if augment_type == 'rotate':
                    angle = random.randint(-30, 30)
                    img = img.rotate(angle, fillcolor=(255, 255, 255))
                elif augment_type == 'flip':
                    if random.random() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                elif augment_type == 'brightness':
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                elif augment_type == 'contrast':
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(random.uniform(0.7, 1.3))
                
                # Сохраняем аугментированное изображение
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                aug_path = os.path.join(output_path, f"{base_name}_aug_{i}.jpg")
                img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))
                img.save(aug_path)
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    # Аугментируем все изображения
    total_augmented = 0
    for folder in aug_folders:
        input_folder = os.path.join(Config.RAW_DATA_PATH, folder)
        output_folder = os.path.join(Config.AUGMENTED_DATA_PATH, folder)
        
        if os.path.exists(input_folder):
            for filename in os.listdir(input_folder):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(input_folder, filename)
                    
                    # Копируем оригинал
                    try:
                        original = Image.open(input_path)
                        original.save(os.path.join(output_folder, filename))
                        
                        # Создаем аугментированные версии
                        apply_augmentations(input_path, output_folder, count=2)
                        total_augmented += 2
                    except Exception as e:
                        print(f"❌ Error with {filename}: {e}")
    
    print(f"✅ Augmentation completed! Created {total_augmented} additional images")
    
    # Посчитаем итоговую статистику
    total_images = 0
    for folder in aug_folders:
        aug_folder = os.path.join(Config.AUGMENTED_DATA_PATH, folder)
        if os.path.exists(aug_folder):
            count = len([f for f in os.listdir(aug_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
            total_images += count
            print(f"   {folder}: {count} images")
    
    print(f"📊 Total images in augmented dataset: {total_images}")

if __name__ == "__main__":
    augment_images()