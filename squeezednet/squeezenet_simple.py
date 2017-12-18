"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Keras implementation of squeezenet with
        model design exploration parameters.
"""

import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import Utility
import Image_loader
import os
import h5py

def main():
    args = Utility.argument_parser()
    os.chdir('../../Dataset/tiny-imagenet-200/')
    data_path = os.getcwd()
    if os.path.isfile("tiny_imagenet.h5"):
        print("File found, loading")
        data_file = h5py.File('tiny_imagenet.h5', 'r')
        group_data = data_file.get('tiny_imagenet_group')
        x_train = np.array(group_data.get('x_train'))
        y_train = np.array(group_data.get('y_train'))
        x_test = np.array(group_data.get('x_test'))
        y_test = np.array(group_data.get('y_test'))
        data_file.close()
    else:
        (x_train, y_train), (x_test, y_test) = Image_loader.load_data_external("tiny_imagenet", data_path)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #model_def = define_model([28,28,1], len(np.unique(np.argmax(y_test, 1))))
    #model_def.summary()

if __name__ == "__main__":
    main()
