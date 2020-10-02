from models import Logistic, Perceptron, np
from typing import List
from data_process import get_MUSHROOM_data, get_CIFAR10_data
from kaggle_submission import output_submission_csv

VALIDATION = 0.2
TRAIN_IMAGES = 40000
VAL_IMAGES = 10000


def get_acc(pred, y_test):
    return np.sum(y_test == pred) / len(y_test) * 100


MR_data = get_MUSHROOM_data(VALIDATION)
X_train_MR, y_train_MR = MR_data['X_train'], MR_data['y_train']
X_val_MR, y_val_MR = MR_data['X_val'], MR_data['y_val']
X_test_MR, y_test_MR = MR_data['X_test'], MR_data['y_test']
n_class_MR = len(np.unique(y_test_MR))

CIFAR_data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES)
X_train_CIFAR, y_train_CIFAR = CIFAR_data['X_train'], CIFAR_data['y_train']
X_val_CIFAR, y_val_CIFAR = CIFAR_data['X_val'], CIFAR_data['y_val']
X_test_CIFAR, y_test_CIFAR = CIFAR_data['X_test'], CIFAR_data['y_test']
n_class_CIFAR = len(np.unique(y_test_CIFAR))

X_train_CIFAR = np.reshape(X_train_CIFAR, (X_train_CIFAR.shape[0], -1))
X_val_CIFAR = np.reshape(X_val_CIFAR, (X_val_CIFAR.shape[0], -1))
X_test_CIFAR = np.reshape(X_test_CIFAR, (X_test_CIFAR.shape[0], -1))


def get_input(flag: List[bool], val=None, t=None, end='s', msg=''):
    if val is not None:
        t = type(val)
    ipt = input(msg)
    if ipt == end:
        return val
    try:
        return t(ipt)
    except ValueError:
        flag[0] = False


def test_logistic(learning_rate=0.4, epochs=15, decay=0.2):
    lr = Logistic(learning_rate, epochs, decay)

    flag = [True]
    while True:
        lr.train(X_train_MR, y_train_MR)
        pred_lr = lr.predict(X_train_MR)
        print('The training accuracy is given by: %f' % (get_acc(pred_lr, y_train_MR)))

        pred_lr = lr.predict(X_val_MR)
        print('The validation accuracy is given by: %f' % (get_acc(pred_lr, y_val_MR)))

        pred_lr = lr.predict(X_test_MR)
        print('The testing accuracy is given by: %f' % (get_acc(pred_lr, y_test_MR)))
        learning_rate = get_input(flag, learning_rate, end='', msg=f'{learning_rate=}')
        lr.lr = learning_rate
        lr.epochs = get_input(flag, epochs, end='', msg=f"epochs=")


def test_perceptron(lr=0.5, epochs=3, decay=0):
    flag = [True]
    percept_CIFAR = Perceptron(n_class_CIFAR, lr, epochs, decay)
    while flag[0]:
        percept_CIFAR.train(X_train_CIFAR, y_train_CIFAR)

        pred_percept = percept_CIFAR.predict(X_train_CIFAR)
        print('The training accuracy is given by: %f' % (get_acc(pred_percept, y_train_CIFAR)))

        pred_percept = percept_CIFAR.predict(X_val_CIFAR)
        print('The validation accuracy is given by: %f' % (get_acc(pred_percept, y_val_CIFAR)))

        pred_percept = percept_CIFAR.predict(X_test_CIFAR)
        print('The testing accuracy is given by: %f' % (get_acc(pred_percept, y_test_CIFAR)))

        percept_CIFAR.lr = get_input(flag, lr, end='', msg=f"{lr=}")
        percept_CIFAR.epochs = get_input(flag, epochs, end='', msg=f"{epochs=}")
    output_submission_csv('kaggle/perceptron_submission_CIFAR.csv', percept_CIFAR.predict(X_test_CIFAR))


def test_perceptron_MR(*, lr=0.5, epochs=3):
    flag = [True]
    percept_MR = Perceptron(n_class_MR, lr, epochs, 0)
    while flag[0]:
        percept_MR.train(X_train_MR, y_train_MR)

        pred_percept = percept_MR.predict(X_train_MR)
        print('The training accuracy is given by: %f' % (get_acc(pred_percept, y_train_MR)))

        pred_percept = percept_MR.predict(X_val_MR)
        print('The validation accuracy is given by: %f' % (get_acc(pred_percept, y_val_MR)))

        pred_percept = percept_MR.predict(X_test_MR)
        print('The testing accuracy is given by: %f' % (get_acc(pred_percept, y_test_MR)))
        percept_MR.lr = get_input(flag, lr, end='', msg=f"{lr=}")
        percept_MR.epochs = get_input(flag, epochs, end='', msg=f"{epochs=}")


if __name__ == '__main__':
    # print((y_train_CIFAR[:50]))
    np.random.seed(27)  # 88%, 92%
    # test_logistic(1e-3, 10, 0)
    # test_perceptron_MR(lr=1e-3, epochs=40)    # 92%
    test_perceptron(lr=1e-4, epochs=10)
