"""
AUTHOR: bulletcross@gmail.com (Vishal Keshav)
TITLE : Utility functions
"""
import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Squeezenet model exploration')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--epochs', default=10, type=int, help='Number of passes thorugh total dataset')
    parser.add_argument('--l_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--l_decay', default=0.01, type=float, help='Learning decay for momentum')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset from keras')
    parser.add_argument('--save', default='./logs', type=str, help='Dataset from keras')
    args = parser.parse_args()
    return args
