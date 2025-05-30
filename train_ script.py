# === INSTALLATION & IMPORTS ===
# Install necessary packages
!pip install -q torch torchvision torchaudio scikit-learn matplotlib

# Import standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Import PyTorch and torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50

# === REPRODUCIBILITY ===
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# === DATA LOADING FUNCTION ===
def load_pneumonia_mnist(data_path: str):
    """
    Load the PneumoniaMNIST dataset from the specified path.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        tuple: Training, validation, and test datasets.
    """
    data = np.load(data_path)
    x_train, y_train = data['train_images'], data['train_labels']
    x_val, y_val = data['val_images'], data['val_labels']
    x_test, y_test = data['test_images'], data['test_labels']

    # Flatten labels
    y_train = y_train.flatten()
    y_val = y_val.flatten()
    y_test = y_test.flatten()

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# Load dataset
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_pneumonia_mnist("/kaggle/input/pneumoniamnist/pneumoniamnist.npz")

# === DATASET CLASS ===
class PneumoniaDataset(Dataset):
    """
    Custom Dataset class for loading PneumoniaMNIST data.

    Args:
        images (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: callable = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx].astype(np.uint8)
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)  # Convert grayscale to RGB
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# === TRANSFORMS ===
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

# === DATALOADER SETUP ===
batch_size = 32

train_dataset = PneumoniaDataset(x_train, y_train, transform=train_transform)
val_dataset = PneumoniaDataset(x_val, y_val, transform=test_transform)
test_dataset = PneumoniaDataset(x_test, y_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# === MODEL CONFIGURATION ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
model = model.to(device)

# === LOSS FUNCTION & OPTIMIZER ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# === TRAINING FUNCTION ===
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Train the model using the provided data loaders, criterion, and optimizer.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int, optional): Number of epochs to train. Defaults to 10.
    """
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        val_correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = val_correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("Best model saved.")

# === MODEL TRAINING ===
train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)

# === EVALUATION ===
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# === PERFORMANCE REPORT ===
print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Pneumonia"]))

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)
