# MNIST Digit Classification Neural Network

This project implements a neural network from scratch to classify handwritten digits from the MNIST dataset using Python. The neural network is trained using forward and backward propagation techniques, with parameters updated through gradient descent.

## Overview

The goal of this project is to develop a neural network model capable of accurately classifying handwritten digits. The MNIST dataset, a collection of 28x28 pixel grayscale images of handwritten digits ranging from 0 to 9, is used for training and testing the model.

## Methodology

### Data Preprocessing

- The MNIST dataset is loaded using Keras and preprocessed.
- The images are reshaped into 784-dimensional vectors and normalized to the range [0, 1].

### Neural Network Architecture

- The neural network consists of an input layer, a hidden layer, and an output layer.
- The input layer has 784 neurons corresponding to the flattened image pixels.
- The hidden layer has 10 neurons with ReLU activation function.
- The output layer has 10 neurons with softmax activation function for multi-class classification.

### Training

- The model parameters (weights and biases) are initialized randomly.
- Forward propagation is performed to compute the predicted probabilities for each class.
- Backward propagation is used to calculate the gradients of the loss function with respect to the parameters.
- Gradient descent is employed to update the parameters iteratively, minimizing the cross-entropy loss.

### Evaluation

- The trained model is evaluated on the test set to assess its performance.
- Accuracy score and confusion matrix are computed to measure the classification performance and identify misclassifications.

## Results

- The model achieves an accuracy of approximately 85% on the test set.
- The confusion matrix reveals the distribution of true positive, true negative, false positive, and false negative predictions for each class.

