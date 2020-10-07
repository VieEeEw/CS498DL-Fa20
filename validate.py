import numpy as np

from kaggle_submission import output_submission_csv
from models.neural_net import NeuralNetwork
from utils.data_process import get_CIFAR10_data
from glob import glob

TRAIN_IMAGES = 49000
VAL_IMAGES = 1000
TEST_IMAGES = 10000

data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_test, y_test = data['X_test'], data['y_test']

# load = 'stored_weights/Adam-2layer'
# load = 'stored_weights/Adam-3layer'
load = glob('stored/epoch*')
# load = 'another_stored/epoch0_0.476'

# net = NeuralNetwork(32 * 32 * 3, [0], 10, 2, load=load)
net = NeuralNetwork(32 * 32 * 3, [20] * 2, 10, 3)


def test_accuracy(lo):
    net.load(lo)
    prediction = np.argmax(net.forward(X_test), axis=1)
    print(f"Accuracy={np.sum(prediction == y_test) / y_test.shape[0]}")
    if input("Output csv?").upper() == 'Y':
        # output_submission_csv('kaggle/nn_2layer_adam_submission.csv', prediction)
        output_submission_csv('kaggle/nn_3layer_adam_submission.csv', prediction)
        exit(0)

for lo in load:
    test_accuracy(lo)
