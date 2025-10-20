import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_quality_dataset():
    """Create a quality test dataset that looks like real food vs objects"""
    
    # Создаем структуру папок
    folders = [
        "data/raw/training/0_food",
        "data/raw/training/1_non_food", 
        "data/raw/validation/0_food",
        "data/raw/validation/1_non_food"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    # Цветовые палитры
    food_palettes = [
        [(210, 180, 140), (139, 69, 19), (205, 133, 63)],  # коричневые тона (хлеб, мясо)
        [(255, 165, 0), (255, 140, 0), (255, 215, 0)],     # оранжевые (фрукты)
        [(255, 192, 203), (219, 112, 147), (199, 21, 133)], # розовые (десерты)
        [(50, 205, 50), (34, 139, 34), (107, 142, 35)]     # зеленые (овощи)
    ]
    
    non_food_palettes = [
        [(192, 192, 192), (169, 169, 169), (105, 105, 105)], # серые (металл)
        [(70, 130, 180), (65, 105, 225), (30, 144, 255)],   # синие
        [(106, 90, 205), (123, 104, 238), (147, 112, 219)], # фиолетовые
        [(0, 206, 209), (64, 224, 208), (72, 209, 204)]     # бирюзовые
    ]
    
    def create_food_image():
        """Create food-like image with organic shapes"""
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        palette = random.choice(food_palettes)
        
        # Создаем органические формы (еда)
        for _ in range(random.randint(3, 6)):
            color = random.choice(palette)
            # Овалы и круги - как еда
            x, y = random.randint(30, 194), random.randint(30, 194)
            size = random.randint(40, 80)
            draw.ellipse([x, y, x+size, y+size], fill=color, outline=(0, 0, 0))
            
            # Добавляем текстуру точками
            for _ in range(random.randint(5, 15)):
                dot_x = random.randint(x, x+size)
                dot_y = random.randint(y, y+size)
                dot_color = tuple(min(255, c + random.randint(-30, 30)) for c in color)
                dot_size = random.randint(2, 5)
                draw.ellipse([dot_x, dot_y, dot_x+dot_size, dot_y+dot_size], fill=dot_color)
        
        return img
    
    def create_non_food_image():
        """Create non-food image with geometric shapes"""
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        palette = random.choice(non_food_palettes)
        
        # Создаем геометрические формы (объекты)
        for _ in range(random.randint(2, 4)):
            color = random.choice(palette)
            shape_type = random.choice(['rect', 'triangle', 'polygon'])
            
            if shape_type == 'rect':
                x, y = random.randint(20, 164), random.randint(20, 164)
                width, height = random.randint(40, 100), random.randint(40, 100)
                draw.rectangle([x, y, x+width, y+height], fill=color, outline=(0, 0, 0))
                
            elif shape_type == 'triangle':
                x, y = random.randint(50, 150), random.randint(50, 150)
                size = random.randint(30, 70)
                points = [(x, y), (x+size, y), (x+size//2, y-size)]
                draw.polygon(points, fill=color, outline=(0, 0, 0))
                
            else:  # polygon
                x, y = random.randint(50, 150), random.randint(50, 150)
                sides = random.randint(4, 6)
                radius = random.randint(30, 60)
                points = []
                for i in range(sides):
                    angle = 2 * 3.14159 * i / sides
                    px = x + radius * np.cos(angle)
                    py = y + radius * np.sin(angle)
                    points.append((px, py))
                draw.polygon(points, fill=color, outline=(0, 0, 0))
        
        return img
    
    # Создаем тренировочные данные (80 изображений)
    print("Creating training data...")
    for i in range(80):
        if i < 40:  # 40 food images
            img = create_food_image()
            img.save(f"data/raw/training/0_food/food_{i:03d}.jpg")
        else:       # 40 non-food images
            img = create_non_food_image()
            img.save(f"data/raw/training/1_non_food/non_food_{i:03d}.jpg")
    
    # Создаем валидационные данные (40 изображений)
    print("Creating validation data...")
    for i in range(40):
        if i < 20:  # 20 food images
            img = create_food_image()
            img.save(f"data/raw/validation/0_food/food_val_{i:03d}.jpg")
        else:       # 20 non-food images
            img = create_non_food_image()
            img.save(f"data/raw/validation/1_non_food/non_food_val_{i:03d}.jpg")
    
    print("✅ Quality test dataset created!")
    print("📊 Statistics:")
    print(f"   Training: 40 food + 40 non-food = 80 images")
    print(f"   Validation: 20 food + 20 non-food = 40 images")
    print(f"   Total: 120 images")

if __name__ == "__main__":
    create_quality_dataset()