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


