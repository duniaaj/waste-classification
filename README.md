# 🗑️ Waste Classification with MobileNetV2

This project uses **MobileNetV2**, a pre-trained convolutional neural network, to classify waste images into three categories: **plastic**, **metal**, and **paper**. The model is trained using TensorFlow and Keras and is evaluated with various metrics including F1-score, precision, recall, AUC, and more.

---

## 📁 Project Structure

- `model.py`: Builds and compiles the MobileNetV2 model with a custom classification head.
- `utils.py`: Handles dataset loading and preprocessing using `ImageDataGenerator`.
- `evaluate.py`: Loads the trained model, runs evaluation on the test set, computes metrics, and displays a confusion matrix.

---




