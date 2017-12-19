"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Keras implementation of squeezenet with
        model design exploration parameters.
"""

import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, Activation
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import TensorBoard as TB
from keras.utils.vis_utils import plot_model
from keras import backend as K

import Utility
import Image_loader
import Model_def
import os
import h5py

def load_tinyimagenet():
    os.chdir('../../Dataset/tiny-imagenet-200/')
    data_path = os.getcwd()
    if os.path.isfile("tiny_imagenet.h5"):
        data_file = h5py.File('tiny_imagenet.h5', 'r')
        group_data = data_file.get('tiny_imagenet_group')
        x_train = np.array(group_data.get('x_train'))
        y_train = np.array(group_data.get('y_train'))
        x_test = np.array(group_data.get('x_test'))
        y_test = np.array(group_data.get('y_test'))
        data_file.close()
    else:
        (x_train, y_train), (x_test, y_test) = Image_loader.load_data_external("tiny_imagenet", data_path)
    return (x_train, y_train), (x_test, y_test)

def save_model(model, model_name):
    model_json = model.to_json()
    with open(model_name+".json","w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name+".h5")
    print("Saved model in the name:" + model_name)

def save_model(model_name):
    json_file = open(model_name+".json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_name+".h5")
    return loaded_model

def train_model(model, data, args):
    (x_train, y_train), (x_test, y_test) = data
    model.compile(optimizer = Adam(lr=args.l_rate),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    tensorboard_callback = TB(log_dir = args.save, histogram_freq = 0,
                              write_graph = True)
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
			  callbacks = [tensorboard_callback], validation_data=[x_test, y_test])


def main():
    args = Utility.argument_parser()
    (x_train, y_train), (x_test, y_test) = load_tinyimagenet()
    model_def = Model_def.define_model([64,64,3], 200, args)
    model_def.summary()
    if args.plot != '0':
        plot_model(model_def, to_file=args.plot, show_shapes = False, show_layer_names = True)


if __name__ == "__main__":
    main()
