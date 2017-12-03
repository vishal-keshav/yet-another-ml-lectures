import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils import to_categorical
import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description='Multi-input end to end mnist')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--epochs', default=10, type=int, help='Number of passes thorugh total dataset')
    parser.add_argument('--l_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--l_decay', default=0.01, type=float, help='Learning decay for momentum')
    parser.add_argument('--dataset', default='mnist', type=str, help='Dataset from keras')
    args = parser.parse_args()
    return args

def load_data(dataset_name):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    else:
        print("Dataset " + dataset_name + " not found, defaulting to mnist.")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if(dataset_name == 'cifar10' or dataset_name == 'cifar100'):
        x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
        x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
        if dataset_set == 'cifar10':
            y_train = to_categorical(y_train, 10)
            y_test = to_categorical(y_test, 10)
        else:
            y_train = to_categorical(y_train, 100)
            y_test = to_categorical(y_test, 100)
    elif dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    else:
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train/225
    t_test = x_test/225
    return (x_train, y_train), (x_test, y_test)

def vector_modulo(y1, y2):
    y_ret = np.zeros(y1.shape)
    y_ret[(np.argmax(y1)+np.argmax(y2))%y1.shape[0]] = 1
    return y_ret

def convert_dataset(x,y,dataset_name):
    x1_new = np.empty((0,x.shape[1], x.shape[2], x.shape[3]))
    x2_new = np.empty((0, x.shape[1], x.shape[2], x.shape[3]))
    y_new = np.empty((0, y.shape[1]))
    for i in range(0,y.shape[0],2):
        y_new = np.append(y_new, [vector_modulo(y[i], y[i+1])], axis=0)
        x1_new = np.append(x1_new, [x[i]], axis=0)
        x2_new = np.append(x2_new, [x[i+1]], axis=0)
    return (x1_new, x2_new, y_new)

def main():
    args = argument_parser()
    (x_train, y_train), (x_test, y_test) = load_data(args.dataset)
    (x_train1, x_train2, y_train) = convert_dataset(x_train, y_train, args.dataset)
    (x_test1, x_test2, y_test) = convert_dataset(x_test, y_test, args.dataset)

if __name__ == "__main__":
    main()
