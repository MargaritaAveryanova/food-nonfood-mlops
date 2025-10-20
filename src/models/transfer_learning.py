import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def create_transfer_learning_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(weights='imagenet', 
                           include_top=False, 
                           input_shape=input_shape)
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model
