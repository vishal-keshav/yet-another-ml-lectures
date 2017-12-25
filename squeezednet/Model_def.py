"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Squeeze-expand defenition function and model defenition
"""

from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D, Concatenate, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from math import floor

def fire_module(input_tensor, nr_squeeze_1, nr_expand_1, nr_expand_3):
    squeeze_tensor = Conv2D(filters = nr_squeeze_1, kernel_size = (1,1), strides=(1,1),
                    activation='relu', padding='same', kernel_initializer='glorot_uniform')(input_tensor)
    expand_1_tensor = Conv2D(filters = nr_expand_1, kernel_size = (1,1), strides=(1,1),
                    activation='relu', padding='same', kernel_initializer='glorot_uniform')(squeeze_tensor)
    expand_3_tensor = Conv2D(filters = nr_expand_3, kernel_size = (3,3), strides=(1,1),
                    activation='relu', padding='same', kernel_initializer='glorot_uniform')(squeeze_tensor)
    output_tensor = Concatenate(axis=3)([expand_1_tensor, expand_3_tensor])
    return output_tensor

def define_model(input_shape, output_class, args):
    max_pooling_index = [2, 6]
    base_expand_kernels = args.base_expand
    expansion_increment = args.expansion_increment
    freq = args.freq
    pct = args.pct
    SR = args.SR
    #Model construction
    model_input = Input(shape = input_shape)
    tensor = Conv2D(filters = 96, kernel_size = (3,3), strides = (2,2), padding = 'same',
                     activation='relu', kernel_initializer='glorot_uniform')(model_input)
    tensor = MaxPooling2D(pool_size = (3,3), strides = (2,2))(tensor)
    for i in range(8):
        expand_kernels = base_expand_kernels + (expansion_increment*(i/freq))
        nr_squeeze_1 = int(SR*expand_kernels)
        nr_expand_1 = int(expand_kernels*(1.0-pct))
        nr_expand_3 = expand_kernels - nr_expand_1
        tensor = fire_module(tensor, nr_squeeze_1, nr_expand_1, nr_expand_3)
        if i in max_pooling_index:
            tensor = MaxPooling2D(pool_size = (3,3), strides = (2,2))(tensor)
    tensor = Dropout(rate = 0.5)(tensor)
    tensor = Conv2D(filters = output_class, kernel_size = (1,1), strides = (1,1),
             padding = 'valid', kernel_initializer = 'glorot_uniform')(tensor)
    global_avgpool = GlobalAveragePooling2D()(tensor)
    soft_output = Activation(activation = 'softmax')(global_avgpool)
    model = Model(inputs = model_input, outputs = soft_output)
    return model
