# Objective
# The main goal of this project is to leverage transfer learning by adapting a ResNet-50 model to classify chest X-ray images and accurately identify cases of pneumonia. The challenge lies in dealing with potential class imbalance and ensuring that the model generalizes well to unseen data.

⚙️ Training Strategy
# Data Augmentation
# To help the model generalize better and reduce the risk of overfitting, data augmentation techniques were applied to the training images:

# Random horizontal flipping

# Small random rotations

# Resizing to 224×224 (required by ResNet-50 input)

# Class Imbalance Handling
# The dataset shows some imbalance between the Normal and Pneumonia classes. To address this:

# We computed class weights using sklearn.utils.class_weight.

# These weights were passed to the CrossEntropyLoss function to penalize the model more for misclassifying underrepresented classes.

# Optimization
# Optimizer: Adam

# Learning Rate: 0.001

# Loss Function: CrossEntropyLoss (with class weights)

# Batch Size: 32

# Epochs: 10

📊 Evaluation Metrics
# Three key metrics were used to evaluate model performance:

# Metric	Reason
# Accuracy	General performance across all classes.
# Precision	Important for minimizing false positives, especially in medical tasks.
# F1-Score	Balanced view of precision and recall in case of class imbalance.

✅ Results
# After training the model on Google Colab using GPU acceleration:

# Test Accuracy: ~82%

# Classification Report:

# makefile
# Copy
# Edit
# Precision: 0.82
# Recall:    0.91
# F1-Score:  0.87 (Pneumonia)
# Confusion Matrix showed strong recall for pneumonia cases, which is desirable in clinical scenarios.
📂 Repository Contents
├── pneumonia_model_training.ipynb   # Training and evaluation notebook
├── best_model.pt                    # Trained model weights
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
