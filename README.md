# Pneumonia Detection using CNN and Transfer Learning (ResNet-50)
This project is about detecting Pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) based on ResNet-50. The model uses transfer learning from ImageNet to improve performance on the medical imaging dataset. The dataset used is PneumoniaMNIST, which includes grayscale X-ray images categorized as either Normal or Pneumonia.
##  Features

- Uses ResNet-50 pre-trained on ImageNet for transfer learning.
- Fine-tuned on PneumoniaMNIST dataset for binary classification.
- Applies class balancing techniques to improve fairness across categories.
- Includes data augmentation to improve generalization and reduce overfitting.
- Generates a classification report and confusion matrix for performance evaluation.

## Requirements
-  Python 3.x for project development.
- Essential Python packages: tensorflow, keras, opencv-python, numpy for image processing.
  
  ## Installation
1.  Clone the repository:
2. Install the required packages:
3. Download the pre-trained pneumonia detection model and label mappings.

   ## Usage
1. Open a new Google Colab notebook.

2. Upload the project files in Google Drive.

3. Load the pre-trained pneumonia detection model and label mappings. Ensure the model files are correctly placed in the Colab working directory.

4. Execute the Pneumonia Detection script in the Colab notebook, which may involve adapting the script to run within a notebook environment.

5. Follow the on-screen instructions or customize input cells in the notebook for pneumonia detection with uploaded medical images.

6. View and analyze the results directly within the Colab notebook.

7. Repeat the process for additional images or iterations as needed.

   ## Program:
   # === Install Dependencies ===
!pip install -q torch torchvision torchaudio scikit-learn matplotlib

# === Import Libraries ===
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50

# === Set Seed for Reproducibility ===
torch.manual_seed(42)
np.random.seed(42)

# === Load PneumoniaMNIST Dataset ===
def load_pneumonia_mnist():
    path = "/content/pneumoniamnist.npz"
    data = np.load(path)
    
    x_train, y_train = data['train_images'], data['train_labels'].flatten()
    x_val, y_val = data['val_images'], data['val_labels'].flatten()
    x_test, y_test = data['test_images'], data['test_labels'].flatten()
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_pneumonia_mnist()

# === Custom Dataset Class ===
class PneumoniaDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)
        image = np.repeat(image[..., np.newaxis], 3, axis=-1)  # Convert to 3 channels
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# === Data Transformations ===
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# === Create DataLoaders ===
batch_size = 32

train_loader = DataLoader(PneumoniaDataset(x_train, y_train, train_transform), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(PneumoniaDataset(x_val, y_val, test_transform), batch_size=batch_size)
test_loader = DataLoader(PneumoniaDataset(x_test, y_test, test_transform), batch_size=batch_size)

# === Setup Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze base layers

# Modify the classifier head
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
model.to(device)

# === Define Loss, Optimizer, and Handle Class Imbalance ===
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# === Training Loop ===
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved improved model.")

# === Train the Model ===
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# === Load and Evaluate Best Model ===
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# === Print Classification Metrics ===
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
100%|██████████| 97.8M/97.8M [00:00<00:00, 178MB/s]
Epoch 1/10: 100%|██████████| 148/148 [00:21<00:00,  6.82it/s]
Validation Accuracy: 0.8073
Best model saved.
Epoch 2/10: 100%|██████████| 148/148 [00:20<00:00,  7.07it/s]
Validation Accuracy: 0.8874
Best model saved.
Epoch 3/10: 100%|██████████| 148/148 [00:20<00:00,  7.12it/s]
Validation Accuracy: 0.8912
Best model saved.
Epoch 4/10: 100%|██████████| 148/148 [00:21<00:00,  7.00it/s]
Validation Accuracy: 0.8855
Epoch 5/10: 100%|██████████| 148/148 [00:21<00:00,  6.84it/s]
Validation Accuracy: 0.8015
Epoch 6/10: 100%|██████████| 148/148 [00:21<00:00,  6.78it/s]
Validation Accuracy: 0.9141
Best model saved.
Epoch 7/10: 100%|██████████| 148/148 [00:21<00:00,  6.92it/s]
Validation Accuracy: 0.9046
Epoch 8/10: 100%|██████████| 148/148 [00:21<00:00,  6.88it/s]
Validation Accuracy: 0.8969
Epoch 9/10: 100%|██████████| 148/148 [00:21<00:00,  6.80it/s]
Validation Accuracy: 0.8931
Epoch 10/10: 100%|██████████| 148/148 [00:21<00:00,  6.87it/s]
Validation Accuracy: 0.9046

Classification Report:

              precision    recall  f1-score   support

      Normal       0.82      0.68      0.74       234
   Pneumonia       0.82      0.91      0.87       390

    accuracy                           0.82       624
   macro avg       0.82      0.79      0.80       624
weighted avg       0.82      0.82      0.82       624


Confusion Matrix:
 [[158  76]
 [ 34 356]]

 Result:
1. Overall Accuracy: The model achieved an accuracy of 82% on the test dataset, demonstrating reliable performance in classifying chest X-rays.

2. Pneumonia Detection: It showed strong sensitivity for pneumonia cases with a recall of 91% and an F1-score of 87%, effectively minimizing false negatives.

3. Balanced Performance: Despite a lower recall of 68% for normal cases, the model maintained balanced metrics overall with a weighted F1-score of 82%, aided by class weighting and regularization techniques.
 


   
