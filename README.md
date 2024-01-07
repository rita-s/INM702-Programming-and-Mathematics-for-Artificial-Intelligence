# Developing a multi-layer neural network for classification using numpy on the MNIST dataset.

## Overview 📖
This README provides detailed instructions on how to set up and reproduce the code for the MNIST Neural Network Project.
Our code includes various functionalities such as loading and preprocessing the MNIST dataset, defining neural network architectures, and implementing training procedures.

## Prerequisites 🛠️
Before running the code, ensure you have the following installed:
- Python (3.6 or later)
- NumPy
- Matplotlib
- TensorFlow (2.0 or later)
- Scikit-learn
- Seaborn

## Code Structure 🏗️
mnist_neural_network.py: The main script containing the entire neural network implementation and training procedures.

## Key Functions 🗝️
load_mnist_and_print(): Loads and prints a sample of the MNIST dataset.
plot_image(): Plots a single image from the dataset.
plot_images(): Plots multiple images from the dataset.
NeuralNetwork: Class defining the neural network structure.
train_neural_network_model(): Function to train the neural network model.
plot_training_history(): Plots the training and validation accuracy and loss.
plot_confusion_matrix(): Plots the confusion matrix for model evaluation.

## Example Usage 📚
To train the neural network model and evaluate its performance:

Load and preprocess the dataset.
Define the neural network configuration.
Initialize the model and optimizer.
Train the model using train_neural_network_model().
Visualise the results using plot_training_history() and plot_confusion_matrix().

## Contributing 🤝
Contributions to the project are welcome! Please follow the standard Git workflow for submitting improvements or bug fixes.


---------

# Implementing deep learning networks using PyTorch

## Prerequisites 🛠️
Ensure you have the following installed before running the code:
- Python 3.6 or later
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- Matplotlib

## Running the Code 🚀
Navigate to the project directory and execute the Python script:

## Code Structure 🏗️
image_classification.py: Contains the full implementation of the image classification model, including data loading, preprocessing, model definition, training, and evaluation.

## Key Components 🗝️
Data loading and preprocessing for CIFAR10 and custom image datasets.
Neural network model class ImageClassification defined using PyTorch.
Functions for training (fit) and evaluating (evaluate) the model.
Utility function show_batch to visualize a batch of images.

## Example Usage 📚
Load and preprocess the CIFAR10 and custom image datasets.
Define the neural network model.
Initialize the model and move it to the appropriate device (CPU/GPU).
Define the optimizer and learning rate.
Train the model using the fit function.
Evaluate the model's performance on the validation dataset.

## Contributing 🤝
Contributions to the project are welcome! Please follow the standard Git workflow for submitting improvements or bug fixes.
