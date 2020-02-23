# The main run file
# command: python3 run.py --configs default --trainer default --dataset default --model default

API_KEY = ''
from comet_ml import Experiment
from comet_ml import Optimizer
import argparse
import importlib

from utils.utils import parse_comet_config
from utils.utils import get_parameters

def argument_parser():
    parser = argparse.ArgumentParser(description="sample")
    parser.add_argument('--config', default='default', type=str, help='configuration')
    parser.add_argument('--trainer', default='default', type=str, help='trainer scripts')
    parser.add_argument('--dataset', default='default', type=str, help='dataset')
    parser.add_argument('--model', default='default', type=str, help='model')
    parser.add_argument('--experiment', default=1, type=int, help='comet.ml')
    args = parser.parse_args()
    return args

def train_all(index, config_keys, configs):
    if index >= len(config_keys):
        train(**{'config': cf, 'model_def': model_def, 'dataset': dataset, 'experiment': experiment})
    else:
        if type(configs[config_keys[index]]) == list:
            for val in configs[config_keys[index]]:
                cf[config_keys[index]] = val
                train_all(index+1, config_keys, configs)
        else:
            cf[config_keys[index]] = configs[config_keys[index]]
            train_all(index+1, config_keys, configs)

def train_with_hp_search():
    train(**{'config': cf, 'model_def': model_def, 'dataset': dataset, 'experiment': experiment})

def main():
    global cf
    global model_def
    global dataset
    global train
    global experiment
    args = argument_parser()
    configs = importlib.import_module("configs."+args.config).config
    model_def = importlib.import_module("model."+args.model).model
    dataset = importlib.import_module("dataset."+args.dataset).dataset
    train = importlib.import_module("trainer."+args.trainer).train
    config_keys = list(configs.keys())
    cf = {}
    if 'algorithm' in configs.keys():
        comet_config = parse_comet_config(configs)
        opt = Optimizer(comet_config, api_key=API_KEY, 
                                      project_name=configs['project_name'])
        for exp in opt.get_experiments():
            experiment = exp
            cf = get_parameters(experiment, configs)
            train_with_hp_search()
    else:
        if args.experiment:
            experiment = Experiment(api_key=API_KEY,
                                    project_name=configs['project_name'], 
                                    workspace=configs['workspace'])
        else:
            experiment = None
        train_all(0, config_keys, configs)

main()