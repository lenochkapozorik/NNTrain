# NNTrain
script that trains a neural network using the TensorFlow.js library in JavaScript

This script uses the TensorFlow.js library to define, train, and evaluate a neural network. The script defines a simple feedforward neural network with one input layer, one hidden layer, and one output layer. It uses the Adam optimizer, categorical cross-entropy loss function and accuracy as metrics. The script loads the data from a data.json file, which should contain two arrays, one for the inputs and one for the outputs. The script then shuffles the data, splits it into a training and testing set and trains the model for 100 epochs. After training, it evaluates the model on the test data and prints the loss and accuracy
