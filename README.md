# Pneumonia Detection using ResNet-50 (Transfer Learning)

## 🎯 Objective

The objective of this project is to fine-tune a **ResNet-50** convolutional neural network to classify chest X-ray images as either **Normal** or **Pneumonia**, and to evaluate the model's performance using appropriate metrics. The task emphasizes handling **class imbalance**, improving **generalization**, and presenting clear **evaluation strategies**.

---

## 📦 Dataset: PneumoniaMNIST

- **Source**: MedMNIST v2 (https://medmnist.com/)
- **Type**: Binary classification (Normal vs. Pneumonia)
- **Format**: `.npz` file with grayscale 28×28 images
- **Splits**:
  - Training set
  - Validation set
  - Test set

---

## 🔧 Task Details

### 1. 🔁 Transfer Learning

- **Base Model**: ResNet-50 pre-trained on ImageNet
- **Modifications**:
  - Replaced final layer with custom classifier: `Linear → ReLU → Dropout → Linear`
  - Only trained the final layers while freezing earlier convolutional blocks

---

### 2. ✅ Evaluation Strategy

#### a) Metrics Used:
- **Accuracy**: Measures overall correctness.
- **Precision & Recall**: Crucial due to class imbalance.
- **F1-Score**: Harmonic mean of precision and recall for balanced evaluation.

#### b) Handling Class Imbalance:
- Applied `class_weight='balanced'` using Scikit-learn to calculate weights for `CrossEntropyLoss`.
- This penalizes misclassification of minority classes appropriately during training.

#### c) Overfitting Prevention Techniques:
- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation
- **Regularization**:
  - Dropout layer added in the classifier
- **Early Stopping**:
  - Model saved only if validation accuracy improved

---

## 🧠 Model Architecture

```python
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
 Hyperparameters
Parameter	Value
Learning Rate	0.001
Batch Size	32
Epochs	10
Optimizer	Adam
