# 🚀 Image Classification Using MobileNetV2 and CNN

This project implements **image classification** using **MobileNetV2** (a pre-trained model) and a **custom CNN**. The models are trained and evaluated on the **CIFAR-10 dataset** to compare their performance.  

---

## 📌 Features
- ✅ **MobileNetV2 Transfer Learning**: Uses a pre-trained model for image classification.
- ✅ **Custom CNN Model**: A simple convolutional neural network built from scratch.
- ✅ **Data Augmentation**: Uses `ImageDataGenerator` to improve generalization.
- ✅ **Training & Evaluation**: Fine-tunes MobileNetV2 and CNN for comparison.
- ✅ **Performance Metrics**: Accuracy, loss curves, confusion matrices, and F1-score.

---

## 📂 Dataset Structure

The dataset follows the **CIFAR-10 class structure**, with **10 categories**:


Each class folder contains **multiple images** belonging to that category.  
- **`train/`** → Used for model training.  
- **`validation/`** → Used for model tuning and hyperparameter optimization.  
- **`test/`** → Used for final model evaluation.  

## 📥 Loading the CIFAR-10 Dataset

To load the **CIFAR-10 dataset** directly from **Keras**, use the following code:

```python
# Import necessary libraries
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Display dataset shape
print(f"Training Data Shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Testing Data Shape: {x_test.shape}, Labels: {y_test.shape}")

---

## 🔄 Normalizing the CIFAR-10 Dataset

---

### **🔄 Why Normalize?**
CIFAR-10 images have pixel values ranging from **[0, 255]**.  
To improve model efficiency, we **normalize** these values by **scaling them to the [0,1] range**.

✅ **Benefits of Normalization:**
- **Speeds up training** by ensuring stable weight updates.
- **Improves model accuracy** by keeping pixel values within a small range.
- **Prevents large gradient updates**, which can cause instability in deep learning models.

### **📊 Min-Max Check**
After normalization:
- **Minimum pixel value:** `0.0`
- **Maximum pixel value:** `1.0`

This ensures that all input images are properly scaled before training.

---

## 🔄 Normalizing the CIFAR-10 Dataset

