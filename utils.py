import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Load the dataset using ImageDataGenerator and flow_from_directory.
    
    Args:
        data_dir (str): Path to the dataset directory.
        img_size (tuple): Target image size (height, width).
        batch_size (int): Batch size for training.
    
    Returns:
        train_generator: Training data generator.
        validation_generator: Validation data generator.
    """
    # Define paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Data augmentation and preprocessing for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        rotation_range=40,  # Data augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Preprocessing for validation (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, validation_generator