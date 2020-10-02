"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        grad_w = np.zeros(self.w.shape)
        for i, sample in enumerate(X_train):
            response = np.matmul(sample, self.w)
            label = y_train[i]
            for j in range(self.n_class):
                if j == label:
                    continue
                margin = response[label] - response[j]
                if margin < 1:
                    grad_w[:-1, j] += X_train[i, :-1]
                    grad_w[-1, j] += 1
                    grad_w[:-1, label] -= X_train[i, :-1]
                    grad_w[-1, label] -= 1

        grad_w /= X_train.shape[0]
        grad_w[:-1] += self.reg_const * self.w[:-1]
        return grad_w

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        if self.w is None:
            self.w = np.random.randn(X_train.shape[1] + 1, self.n_class) * 0.01
            self.w[-1] = 0
        train_size = X_train.shape[0]
        X_train = np.append(X_train, np.ones((train_size, 1)), axis=1)
        batch_size = 64
        indx = np.arange(train_size)
        for i in range(self.epochs):
            if i <= 20:
                rate = self.alpha
            else:
                rate = self.alpha / (1 + (i - 20) * 0.1)  # learning rate decay
            np.random.shuffle(indx)
            X_train = X_train[indx]
            y_train = y_train[indx]
            for j in range(0, train_size, batch_size):
                batch = X_train[j:min(j + batch_size, train_size), ]
                batch_label = y_train[j:min(j + batch_size, train_size), ]
                self.w -= rate * self.calc_gradient(batch, batch_label)
        return

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
        # TODO: implement me
        X_test = np.append(X_test, np.ones((X_test.shape[0], 1)), axis=1)
        response = np.matmul(X_test, self.w)
        return np.argmax(response, axis=1)
