"""Logistic regression model."""

import numpy as np
from tqdm import tqdm


class Logistic:
    def __init__(self, lr: float, epochs: int, decay: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.threshold = 0

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + np.exp(-z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis=1)
        if self.w is None:
            self.w = np.random.rand(X_train.shape[1])
            # self.w = np.zeros(X_train.shape[1])
        for it in range(self.epochs):
            lr = self.lr
            with tqdm(zip(X_train, y_train)) as t:
                t.set_postfix(epoch=it)
                cost = 0
                for row, gt in t:
                    gt = -1 if gt == 0 else 1
                    pred = 1 if row @ self.w > self.threshold else -1
                    # cost += np.log(1 + np.exp(-gt * self.w.T @ row))
                    if pred != gt:
                        grad = self.sigmoid(-gt * self.w.T @ row) * gt * row
                        self.w += lr * grad
                    t.set_postfix(loss=cost, lr=lr)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        return (np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1) @ self.w) > self.threshold
