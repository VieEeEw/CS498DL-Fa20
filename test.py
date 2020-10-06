import numpy as np
import matplotlib.pyplot as plt
from kaggle_submission import output_submission_csv
from shutil import rmtree
from glob import glob
from models.neural_net import NeuralNetwork
from utils.data_process import get_CIFAR10_data
from tqdm import tqdm
from thread import Thread

plt.rcParams['figure.figsize'] = (10.0, 8.0)
np.random.seed(27)

TRAIN_IMAGES = 49000
VAL_IMAGES = 1000
TEST_IMAGES = 10000

data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']


def SGD(input_size=32 * 32 * 3,
        num_layers=2,
        hidden_size=20,
        num_classes=10,
        epochs=100,
        batch_size=200,
        learning_rate=1e-3,
        learning_rate_decay=lambda x, y: x,
        regularization=0.1,
        load=None):
    hidden_sizes = [hidden_size] * (num_layers - 1)
    # Initialize a new neural network model
    net = NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)
    if load is not None:
        net.load(load)
    for directory in glob('stored/*'):
        rmtree(directory)
    # Variables to store performance for each epoch
    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    Ay = np.concatenate((X_train, y_train[..., np.newaxis]), axis=1)

    def train():
        np.random.shuffle(Ay)
        features = Ay[:, :-1]
        labels = Ay[:, -1][..., np.newaxis]

        # Training
        # For each mini-batch...
        for batch in inner:
            # Create a mini-batch of training data and labels
            X_batch = features[batch_size * batch: batch_size * (batch + 1), :]
            y_batch = labels[batch_size * batch: batch_size * (batch + 1), :]

            # Run the backward pass of the model to update the weights and compute the loss
            lr = learning_rate_decay(learning_rate, epoch)
            loss = net.backward(X_batch, y_batch, lr, regularization)

            # Run the forward pass of the model to get a prediction and compute the accuracy
            accuracy = np.sum(np.argmax(net.forward(X_batch), axis=1) == y_batch.flatten())

            train_loss[epoch] += loss
            train_accuracy[epoch] += accuracy

        train_accuracy[epoch] /= X_train.shape[0]
        # Validation
        # No need to run the backward pass here, just run the forward pass to compute accuracy
        prediction = net.forward(X_val)
        val_accuracy[epoch] += np.sum(np.argmax(prediction, axis=1) == y_val.flatten())
        val_accuracy[epoch] /= X_val.shape[0]

        t.set_postfix(loss=train_loss[epoch], val_accuracy=val_accuracy[epoch],
                      train_accuracy=train_accuracy[epoch])

    flag = False
    with tqdm(range(epochs), desc="Epoch", position=1) as t:
        for epoch in t:
            if flag:
                inner.close()
                break
            try:
                inner = tqdm(range(TRAIN_IMAGES // batch_size), desc="Batch", leave=False, position=0)
                a = Thread(target=train)
                a.start()
                a.join()
            except KeyboardInterrupt:
                flag = True
            finally:
                net.freeze(epoch=epoch, accuracy=val_accuracy[epoch])
    inner.close()
    return train_loss, train_accuracy, val_accuracy


def Adam(input_size=32 * 32 * 3,
         num_layers=2,
         hidden_size=20,
         num_classes=10,
         epochs=100,
         batch_size=200,
         learning_rate=1e-3,
         learning_rate_decay=lambda x, y: x,
         regularization=0.1,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-7,
         load=None
         ):
    hidden_sizes = [hidden_size] * (num_layers - 1)
    # Initialize a new neural network model
    net = NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)
    if load is not None:
        net.load(load)
    for directory in glob('stored/*'):
        rmtree(directory)
    # Variables to store performance for each epoch
    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)

    Ay = np.concatenate((X_train, y_train[..., np.newaxis]), axis=1)

    def train():
        np.random.shuffle(Ay)
        features = Ay[:, :-1]
        labels = Ay[:, -1][..., np.newaxis]

        # Training
        # For each mini-batch...
        for batch in inner:
            # Create a mini-batch of training data and labels
            X_batch = features[batch_size * batch: batch_size * (batch + 1), :]
            y_batch = labels[batch_size * batch: batch_size * (batch + 1), :]

            # Run the backward pass of the model to update the weights and compute the loss
            lr = learning_rate_decay(learning_rate, epoch)
            loss = net.backward(X_batch, y_batch, lr, regularization)

            # Run the forward pass of the model to get a prediction and compute the accuracy
            accuracy = np.sum(np.argmax(net.forward(X_batch), axis=1) == y_batch.flatten())

            train_loss[epoch] += loss
            train_accuracy[epoch] += accuracy

        train_accuracy[epoch] /= X_train.shape[0]
        # Validation
        # No need to run the backward pass here, just run the forward pass to compute accuracy
        prediction = net.forward(X_val)
        val_accuracy[epoch] += np.sum(np.argmax(prediction, axis=1) == y_val.flatten())
        val_accuracy[epoch] /= X_val.shape[0]

        t.set_postfix(loss=train_loss[epoch], val_accuracy=val_accuracy[epoch],
                      train_accuracy=train_accuracy[epoch])

    flag = False
    with tqdm(range(epochs), desc="Overall", position=1) as t:
        for epoch in t:
            if flag:
                break
            try:
                inner = tqdm(range(TRAIN_IMAGES // batch_size), desc="Epoch", leave=False, position=0)
                a = Thread(target=train)
                a.start()
                a.join()
            except KeyboardInterrupt:
                flag = True
            finally:
                net.freeze(epoch=epoch, accuracy=val_accuracy[epoch])
    inner.close()
    return train_loss, train_accuracy, val_accuracy


if __name__ == '__main__':
    def decay(lr, epoch):
        if epoch < 2:
            return lr
        # return 1e-5 * np.e ** (-0.001 * epoch)
        return 5e-4


    # train_loss, train_accuracy, val_accuracy = SGD(learning_rate_decay=decay, epochs=25, learning_rate=5e-2,
    #                                                hidden_size=25, regularization=0.1, num_layers=2)
    train_loss, train_accuracy, val_accuracy = SGD(epochs=15, learning_rate=3e-2,
                                                   hidden_size=20, regularization=5e-3, num_layers=3,
                                                   load='stored/epoch26_0.484')
    plt.subplot(2, 1, 1)
    plt.plot(train_loss)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_accuracy, label='train')
    plt.plot(val_accuracy, label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()
    plt.show()
    plt.close('all')
    # input_size = 32 * 32 * 3
    # num_layers = 2
    # hidden_size = 20
    # num_classes = 10
    # hidden_sizes = [hidden_size] * (num_layers - 1)
    # net = NeuralNetwork(input_size, hidden_sizes, num_classes, num_layers)
    # print(net.forward(X_train).shape)
