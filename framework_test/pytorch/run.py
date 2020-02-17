import argparse
import importlib

def argument_parser():
    parser = argparse.ArgumentParser(description="sample")
    parser.add_argument('--config', default='default', type=str, help='configuration')
    parser.add_argument('--trainer', default='default', type=str, help='trainer scripts')
    parser.add_argument('--dataset', default='default', type=str, help='dataset')
    parser.add_argument('--model', default='default', type=str, help='model')
    args = parser.parse_args()
    return args

def train_all(index, config_keys, configs):
    if index >= len(config_keys):
        train(**{'config': cf, 'model': model_def, 'dataset': dataset})
    else:
        for val in configs[config_keys[index]]:
            cf[config_keys[index]] = val
            train_all(index+1, config_keys, configs)

def main():
    global cf
    global model_def
    global dataset
    global train
    args = argument_parser()
    configs = importlib.import_module("configs."+args.config).config
    model_def = importlib.import_module("model."+args.model).model
    dataset = importlib.import_module("dataset."+args.dataset).dataset
    train = importlib.import_module("trainer."+args.trainer).train
    config_keys = list(configs.keys())
    print(config_keys)
    cf = {}
    train_all(0, config_keys, configs)

main()