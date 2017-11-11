import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import argparse

class block_builder(object):
	"""
	Basic block building class, each instance can be used differently
	"""
	def __init__(self, nr_channels, kernel_size, strides, padding, type):
		self.nr_channels = nr_channels
		self.k = kernel_size
		self.s = strides
		self.p = padding
		self.b_type = type

	def __call__(self, input_tensor, block_index):
		if self.b_type == 1:
			tensor = Conv2D(self.nr_channels, kernel_size=(self.k, self.k),
							strides = self.s, padding = 'same', name = 'conv' + str(block_index))(input_tensor)
			tensor = BatchNormalization(axis = 3, name = 'bn'+str(block_index))(tensor)
			tensor = Activation('relu')(tensor)
			return tensor
		elif self.b_type == 2:
			tensor = Conv2D(self.nr_channels, kernel_size=(self.k, self.k),
							strides = self.s, padding = self.p, name = 'conv' + str(block_index))(input_tensor)
			tensor = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='valid')(tensor)
			return tensor
		else:
			tensor = Conv2D(self.nr_channels, kernel_size=(self.k, self.k),
							strides = self.s, padding = self.p, name = 'conv' + str(block_index))(input_tensor)
			tensor = Activation('relu')(tensor)

def load_fashion_mnist():
	from keras.datasets import fashion_mnist
	from keras.utils import to_categorical

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	#Normalize dataset
	x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')/255
	x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')/255
	y_train = to_categorical(y_train.astype('float32'))
	y_test = to_categorical(y_test.astype('float32'))
	return (x_train, y_train), (x_test, y_test)

def define_model(input_shape, output_class):
	"""
	Stacks-up CNN blocks to complete model building
	"""
	model_input = Input(shape = input_shape)

	block_type1 = block_builder(32, 3, 1, 'valid', 1)
	block_type2 = block_builder(64, 5, 2,'valid', 2)

	block = block_type1(model_input, 1)
	block = block_type1(block, 2)
	block = block_type2(block, 3)
	block = block_type2(block, 4)

	block = Flatten()(block)
	block = Dense(1024, activation='relu')(block)
	block = Dense(128, activation='relu')(block)
	model_output = Dense(output_class, activation='softmax')(block)

	model =  Model(inputs=model_input, outputs=model_output)
	return model


def train_model(model, data, args):
	"""
	Compiles and fit the model on data with given args arguments
	"""
	(x_train, y_train), (x_test, y_test) = data
	model.compile(optimizer = Adam(lr=args.learning_rate),
				  loss = 'categorical_crossentropy',
				  metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
			  validation_data=[x_test, y_test])

def test_model(model, data):
	"""
	Test the model accuracy with already trained weights
	"""
	x_test, y_test = data
	y_prediction = model.predict(x_test, batch_size=args.batch_size)
	print(np.sum(np.argmax(y_prediction,1) == np.argmax(y_test,1))/y_test.shape[0])

if __name__ == "__main__":

	"""
	Simplified CNN model to acheive high accuracy on fashion mnist dataset
	"""

	parser = argparse.ArgumentParser(description='Arguments for fashin mnist with CNNs')
	parser.add_argument('--batch_size', default=200, type=int, help='Number of examples to be learnet at once..')
	parser.add_argument('--epochs', default=10, type=int, help='Number of full pass upon one complete test set, use 100 if training on GPU')
	parser.add_argument('--learning_rate', default=0.005, type=float, help='Learning rate, use lesser than 0.01 always')
	parser.add_argument('--is_training', default=1, type=int, help='If not training, forward pass is done on saved model')
	args = parser.parse_args()
	print(args)


	(x_train, y_train),(x_test, y_test) = load_fashion_mnist()

	model_def = define_model([28,28,1], len(np.unique(np.argmax(y_test, 1))))
	model_def.summary()

	if args.is_training:
		train_model(model_def, ((x_train, y_train),(x_test, y_test)), args)
	else:
		test_model(mode_def, (x_test, y_test))
