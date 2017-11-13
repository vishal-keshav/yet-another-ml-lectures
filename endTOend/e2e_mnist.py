import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
import argparse


class multi_input_net(object):
    def __init__(self, axis, **kwargs):
        self.axis = axis
        super(multi_input_net, self).__init__(**kwargs)

    def build(self):
        super(multi_input_net, self).build()

    def call(self, inputs, **kwargs):
        flat_output = []
        for i in range(0, len(inputs)):
            temp_out = deep_net(inputs[i])
            flat_output.append(temp_out)
        self.output = K.concatenate(flat_output, axis = self.axis)
        return self.output

    def deep_net(self,input):
        tensor = BatchNormalization()(input)
        tensor = Conv2D(64, (3,3), 1, 'valid')(tensor)
        tensor = MaxPooling2D((2,2), (2,2), 'valid')(tensor)
        tensor = Conv2D(128, (3,3), 1, 'same')(tensor)
        tensor = MaxPooling2D((2,2), (2,2), 'same')(tensor)
        out = Flatten()(tensor)
        return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Multi-input end to end mnist')
	parser.add_argument('--batch_size', default=512, type=int, help='Number of examples to be learnet at once..')
	parser.add_argument('--epochs', default=10, type=int, help='Number of full pass upon one complete test set, use 100 if training on GPU')
	parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate, use lesser than 0.01 always')
	args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = mnist.load_data

    #[TODO]Prepare dataset in triplets
    x1_input_train = None
    x2_input_train = None
    y_input_train = None

    input1 = Input(shape = [28,28,1])
    input2 = Input(shape = [28,28,1])

    flat_out = multi_input_net(axis = 1)([input1, input2])
    fc = Dense(2048)(flat_out)
    fc = Activation('relu')(fc)
    fc = Dense(28*28)(fc)

    model = Model(inputs = [input1, input2], outputs = fc)
    model.summary()
    
    model.compile(optimizer = Adam(lr=args.learning_rate), loss = 'mean_squared_error',
                  metrics = ['accuracy'])
    model.fit([x1_input_train, x2_input_train], y_input_train, batch_size = args.batch_size,
              epochs = args.epochs)
