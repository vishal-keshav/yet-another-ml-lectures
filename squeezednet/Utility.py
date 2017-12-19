"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Utility functions
"""
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Squeezenet model exploration')
    parser.add_argument('--is_training', default=True, type=bool, help='Train or test?')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=10, type=int, help='Number of passes thorugh total dataset')
    parser.add_argument('--l_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--l_decay', default=0.0001, type=float, help='Learning decay for momentum')
    parser.add_argument('--dataset', default='tiny_imagenet', type=str, help='Dataset options')
    parser.add_argument('--model_name', default='model', type=str, help='Saving and retriving trained model')
    parser.add_argument('--save', default='./logs', type=str, help='Save path for tensorboard visualization')
    parser.add_argument('--plot', default='0', type=str, help='Plot the model graph')
    #Metaparameters for squeeze-net
    parser.add_argument('--base_expand', default= 128, type=int, help='Number of expansion filter in first fire module')
    parser.add_argument('--expansion_increment', default= 128, type=int, help='Increase in channel through expansion')
    parser.add_argument('--pct', default=0.5, type=float, help='Ratio of 1X1 and 3X3 expansion filters in fire module')
    parser.add_argument('--freq', default=2, type=int, help='Expansion in filters every freq fire modules')
    parser.add_argument('--SR', default=0.125, type=float, help='Squeeze ratio of squeeze and expand filters')
    args = parser.parse_args()
    return args
