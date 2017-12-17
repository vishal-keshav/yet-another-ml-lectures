"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Image loader utility for fetching train/test data
"""
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from keras.utils import to_categorical

def load_data_simple(dataset_name):
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
