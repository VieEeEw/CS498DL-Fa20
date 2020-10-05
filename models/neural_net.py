"""Neural network model."""

import os
from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
            self,
            input_size: int,
            hidden_sizes: Sequence[int],
            output_size: int,
            num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_sizes: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

        self.outputs = {}
        self.gradients = {}

    @staticmethod
    def linear(W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """
        return (X @ W) + b

    @staticmethod
    def relu(X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return X * (X > 0)

    @staticmethod
    def softmax(X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        X -= np.max(X, axis=1, keepdims=True)
        X_e = np.exp(X)
        return X_e / np.sum(X_e, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """

        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.
        self.outputs["b0"] = X
        for i in range(1, self.num_layers + 1):
            I = X if i == 1 else self.outputs["b" + str(i - 1)]
            W, b = self.params["W" + str(i)], self.params["b" + str(i)]
            if i < self.num_layers:
                self.outputs["b" + str(i)] = self.relu(self.linear(W, I, b))
            else:
                self.outputs["b" + str(i)] = self.softmax(self.linear(W, I, b))  # what if remove softmax
        return self.outputs["b" + str(self.num_layers)]

    def backward(
            self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        y = y.flatten().astype(int)

        def softmax_grad(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
            grad = np.zeros((1, self.output_size))
            output = self.forward(X_train)
            for i, scores in enumerate(output):
                for j in range(self.output_size):
                    if j == y_train[i]:
                        grad[0, j] += (scores[j] - 1)
                    else:
                        grad[0, j] += scores[j]
            # grad /= X_train.shape[0]
            return grad

        loss = 0
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.
        for k, sample in enumerate(X):
            sample = sample.reshape((1, -1))
            out = self.forward(sample)
            loss -= np.log(out[0, y[k]])
            label = y[k].reshape((1, 1))
            # initialize the upstream gradient to be the gradient wrt the final linear output
            upstream = softmax_grad(sample, label)
            for i in range(self.num_layers, 0, -1):
                if i == self.num_layers:
                    # calculate the gradient wrt b in this layer
                    this_b_grad = upstream
                else:
                    # pass gradient through the ReLU function
                    this_b_grad = upstream * (self.outputs['b' + str(i)] > 0)
                # accumulate the gradient of this sample in the batch
                self.gradients["b" + str(i)] = self.gradients.get("b" + str(i), 0) + this_b_grad
                self.gradients["W" + str(i)] = self.gradients.get("W" + str(i), 0) + (
                        self.outputs["b" + str(i - 1)].T @ this_b_grad)  # @ self.gradients["b"+str(i)])
                if i > 1:  # if this is not the first layer, backpropogate the upstream gradient
                    upstream = this_b_grad @ self.params['W' + str(i)].T

        for i in range(1, self.num_layers + 1):  # cantongjiaoxun
            self.gradients["W" + str(i)] /= X.shape[0]
            self.gradients["W" + str(i)] += reg * self.params["W" + str(i)]
            self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
            self.gradients["b" + str(i)] /= X.shape[0]
            # self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
            self.params["b" + str(i)] -= (lr * self.gradients["b" + str(i)]).flatten()
        return loss / X.shape[0]

    def freeze(self, path_to_dir='stored/', epoch=0, accuracy=0):
        directory = os.path.join(path_to_dir, f'epoch{epoch}_{np.round(accuracy, 4)}')
        if not os.path.exists(directory):
            os.mkdir(directory)
        for k, v in self.params.items():
            np.save(os.path.join(directory, k), v)

    def load(self, path_to_weight):
        for file in os.listdir(path_to_weight):
            self.params[file.split('.')[0]] = np.load(os.path.join(path_to_weight, file))

