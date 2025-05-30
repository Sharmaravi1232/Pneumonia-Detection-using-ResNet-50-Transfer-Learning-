Pneumonia Detection using ResNet-50 and Transfer Learning
This project applies transfer learning using a pre-trained ResNet-50 model to detect pneumonia from chest X-ray images. The model is fine-tuned on the PneumoniaMNIST dataset, a subset of the MedMNIST collection, designed for binary classification: Normal vs Pneumonia.

🔍 Project Highlights
Utilizes ResNet-50, a deep convolutional neural network pre-trained on ImageNet.

.Tailored for binary medical image classification.

.Incorporates techniques to handle class imbalance, including class weighting.

. Includes data augmentation and dropout to mitigate overfitting.

. Evaluated with multiple metrics: Accuracy, Precision, Recall, F1-score.

📁 Dataset: PneumoniaMNIST
Source: MedMNIST v2

.Format: .npz file containing 28×28 grayscale chest X-ray images.

.Classes: Normal (0) and Pneumonia (1)

.Splits:

.Training Set

.Validation Set

.Test Set

🧠 Model Architecture
We fine-tuned the ResNet-50 model by freezing the convolutional layers and modifying the classifier head as follows:

python
Copy
Edit
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

Loss Function: CrossEntropyLoss with class weights.

Optimizer: Adam

Batch Size: 32

Epochs: 10


