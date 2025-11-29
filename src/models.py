import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models
from src.config import IMG_SIZE, NUM_CLASSES

def build_model(learning_rate=1e-3, freeze_backbone=True):
    """
    Builds the EfficientNetB3 model with a custom classification head.
    
    Args:
        learning_rate (float): Learning rate for the optimizer.
        freeze_backbone (bool): Whether to freeze the backbone layers.
        
    Returns:
        tf.keras.Model: The compiled model.
    """
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Load EfficientNetB3 backbone
    backbone = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs)
    
    if freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
        
    # Custom classification head
    x = layers.GlobalAveragePooling2D()(backbone.output)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model