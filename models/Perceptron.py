"""Perceptron model."""

import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.decay = decay
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        decay = False
        X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis=1)
        if self.w is None:
            self.w = np.random.rand(X_train.shape[1], self.n_class)
        pre_loss = None
        for it in range(self.epochs):
            stored_w = self.w.copy()
            loss = 0
            if decay:
                self.lr = self.lr / (1 + it**0.2)
                decay = False
            with tqdm(zip(X_train, y_train)) as t:
                t.set_description(f'epoch: {it}')
                for feature, gt in t:
                    pred = np.argmax(feature @ self.w)
                    temp = feature @ self.w - feature @ self.w[:, gt]
                    loss += np.sum(temp[temp > 0])
                    if pred != gt:
                        grad = -np.outer(feature, (temp > 0))
                        grad[:, gt] = -grad[:, gt]
                        self.w += self.lr * grad
                    t.set_postfix(loss=loss, lr=self.lr)
            if pre_loss is not None and pre_loss <= loss:
                print("Restore to last saved weight")
                self.w = stored_w
                decay = True
                continue
            pre_loss = loss
        np.savetxt("perceptron.csv", X=self.w, delimiter=',')

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
        if self.w is None:
            self.w = np.random.rand(X_test.shape[1], self.n_class)
        return np.argmax(np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1) @ self.w, axis=1)
