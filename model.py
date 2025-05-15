from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_model(num_classes=3, input_shape=(224, 224, 3)):
    """
    Build the MobileNetV2 model with pre-trained weights.
    """
    # Load MobileNetV2 base model with pre-trained weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Output layer for 3 classes
    
    # Combine base model and custom layers
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def compile_model(model):
    """
    Compile the model.
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
        metrics=['accuracy']  # Track accuracy during training
    )
    return model