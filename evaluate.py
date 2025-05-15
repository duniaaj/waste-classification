import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from utils import load_data

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot the confusion matrix.
    
    Args:
        y_true: True labels (class indices).
        y_pred: Predicted labels (class indices).
        class_names: List of class names.
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

def calculate_metrics(y_true, y_pred):
    """
    Calculate Sensitivity, Specificity, F1-score, Precision, and AUC for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded).
        y_pred: Predicted labels (one-hot encoded).
    
    Returns:
        Dictionary containing accuracy, sensitivity (recall), specificity, f1_score, precision, and AUC.
    """
    # Convert one-hot encoded labels to class indices
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Calculate specificity for multi-class (average specificity)
    specificity = np.mean([cm[i, i] / np.sum(cm[:, i]) for i in range(cm.shape[0])])
    
    # Calculate AUC
    try:
        # Binarize the true labels and predicted labels for AUC calculation
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
        auc = roc_auc_score(y_true_bin, y_pred_bin, multi_class='ovr')  # Use 'ovr' for multi-class
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc = np.nan  # Set AUC to NaN if it cannot be calculated
    
    return {
        'accuracy': accuracy,
        'sensitivity': recall,  # Sensitivity is the same as recall
        'specificity': specificity,
        'f1_score': f1,
        'precision': precision,
        'auc': auc
    }

def evaluate_model():
    """
    Evaluate the trained MobileNetV2 model on the test dataset.
    """
    # Load the trained model
    model = tf.keras.models.load_model('waste_classification_mobilenetv2.tflite')

    # Load the test dataset
    test_dir = os.path.join('data', 'test')  # Path to the test dataset
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)  # Only rescale for test data

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),  # Match the image size used during training
        batch_size=32,  # Adjust batch size as needed
        class_mode='categorical',  # Use categorical for multi-class classification
        shuffle=False  # Do not shuffle the test data
    )

    # Evaluate the model on the test set
    y_true = test_generator.classes  # Get the true labels
    y_pred = model.predict(test_generator)  # Get the predicted probabilities

    # Ensure y_pred is in the correct shape (n_samples, n_classes)
    if y_pred.ndim == 1:
        y_pred = tf.one_hot(y_pred, depth=3)  # Convert to one-hot encoding if necessary

    # Calculate metrics
    metrics = calculate_metrics(tf.one_hot(y_true, depth=3), y_pred)  # Convert y_true to one-hot encoding
    print("\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot confusion matrix
    class_names = list(test_generator.class_indices.keys())  # Get class names
    plot_confusion_matrix(y_true, tf.argmax(y_pred, axis=1), class_names)

if __name__ == "__main__":
    evaluate_model()