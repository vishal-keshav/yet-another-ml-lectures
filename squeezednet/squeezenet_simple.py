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

def main():
    args = Utility.argument_parser()
    (x_train, y_train), (x_test, y_test) = Image_loader.load_data(args.dataset)
    #model_def = define_model([28,28,1], len(np.unique(np.argmax(y_test, 1))))
    #model_def.summary()

if __name__ == "__main__":
    main()
