import os
import sys

import torch

sys.path.append(os.path.abspath('.'))
from utils.utils import stringify

def train(config, model_def, dataset, experiment):
    print(config)
    if 'algorithm' in config.keys():
        experiment.log_metric(config['metric'], float(config['nr_epochs'])/float(config['batch_size']))

def test():
    config = {'a': 'b'}
    from model.default import model as model_def
    from dataset.default import dataset
    experiment = None
    train(config, model_def, dataset, experiment)


if __name__ == "__main__":
    test()