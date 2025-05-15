import os
import tensorflow as tf
from utils import load_data
from model import build_model, compile_model

# Define paths
DATA_DIR = 'data'  
MODEL_SAVE_PATH = 'waste_classification_mobilenetv2_large_dataset_8.keras'

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 64  

def train_model():
    """
    Train the MobileNetV2 model.
    """
    # Set random seed for reproducibility
    seed_value = 42
    tf.random.set_seed(seed_value)

    # Load the dataset
    train_generator, validation_generator = load_data(DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # Build and compile the model
    num_classes = 3  
    model = build_model(num_classes)
    model = compile_model(model)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=10,  
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        verbose=1
    )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved as {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()