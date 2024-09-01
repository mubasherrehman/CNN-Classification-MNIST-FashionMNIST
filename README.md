
# PyTorch MNIST & FashionMNIST CNN Classification
This repository contains a PyTorch implementation of convolutional neural networks (CNNs) for classifying images from the MNIST and FashionMNIST datasets. The project showcases data loading, model training, evaluation, and visualization of results for both datasets.

## Contents

 1. **Data Preparation**:
 -  Imports and loads the MNIST and FashionMNIST datasets.
 - Preprocesses data with `ToTensor()` transformations and creates data loaders.
 2. **Model Definition**:
 -  Defines a CNN model using PyTorch’s `nn.Module`.
 - Constructs the network with convolutional blocks followed by a classifier.
 3. **Training and Evaluation**:
 -   Implements training and evaluation functions with accuracy tracking and timing.
 - Trains the model on both GPU and CPU for performance comparison.
 4. **Prediction and Visualization**:
 -   Makes predictions on test data and compares with true labels.
 - Visualizes a selection of test images with their predicted and true labels.
 - Plots the confusion matrix to evaluate model performance.

## Features
-   **Device Agnostic**: Runs on GPU if available, otherwise falls back to CPU.
-   **Data Visualization**: Displays sample images with predictions and true labels.
-   **Performance Metrics**: Tracks training time, loss, and accuracy for both MNIST and FashionMNIST datasets.

## Getting Started
1.  **Clone the repository**:
- `https://github.com/mubasherrehman/CNN-Classification-MNIST-FashionMNIST.git`
2. **Install dependencies**:
- `pip install torch torchvision matplotlib tqdm torchmetrics mlxtend
`
3. **Run the File**
- `pytorch_computer_vision.ipynb`

## Results
-   **Training Accuracy**: Achieved ***99.09%*** accuracy on MNIST and ***89.94%*** accuracy on FashionMNIST training data.
-   **Testing Accuracy**: Achieved ***99.29%*** accuracy on MNIST and ***88.97%*** accuracy on FashionMNIST test data.

## Screenshots (MNIST dataset)
- Comparing the Test data with Predicted data
![image](https://github.com/user-attachments/assets/5f8df09a-615e-407a-b254-84e3169c9989)

- Confusion matrix comparing model's predictions to the truth labels
![image](https://github.com/user-attachments/assets/da4a6401-c994-4a94-8cca-a7897c1fb830)

## Screenshots (FashionMNIST dataset)
- Comparing the Test data with Predicted data
![image](https://github.com/user-attachments/assets/667475b7-8504-4ab7-97b8-a3a5d5b86795)

- Confusion matrix comparing model's predictions to the truth labels
![image](https://github.com/user-attachments/assets/f9a4345e-b026-4f1f-bba7-cd1e41d38831)


