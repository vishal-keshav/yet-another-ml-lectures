# This module contains random utilities. These are moduler functions that
# are used through out the project. What is the input, and what is the output
# should be clearly mentioned in order to create one utility.
import os
import numpy as np
from scipy.signal import resample
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

def folder_exists(path):
    """
    Makes folder iteratively, if it already does not exists
    There is no output.
    """
    try:
        os.makedirs(path)    
    except FileExistsError:
        pass

def generate_permulations(param_list):
    """
    param_list is a list(types of parameter) of list(possible parameter values).
    returns a list(all permulation) of list(parameter value) in same order
    as in param_list
    """
    permu = []
    def recurse(param_type_index, current_param_value, param_list):
        if param_type_index == len(param_list):
            permu.append(current_param_value)
        else:
            for val in param_list[param_type_index]:
                temp = current_param_value.copy()
                temp.append(val)
                recurse(param_type_index+1, temp, param_list)
    recurse(0, [], param_list)
    return permu

def upsample_signal(data, sampling_factor, sampler = None):
    """
    data is a time series sequence(nd_array) numpy data
    upsampling uses fourier interpolation.
    """
    return resample(data, sampling_factor*data.shape[0])

def downsample_signal(data, sampling_factor, sampler = None):
    """
    data is time series sequenced (nd_array) numpy data
    downsample just takes the average sampling_factor points.
    nr_data points should be divisible by sampling_factor
    """
    reshaped = data.reshape(data.shape[0]//sampling_factor, sampling_factor, -1)
    return reshaped.mean(axis = 1)

def generic_sampler(data, sampling_rate, sampler):
    """
    apply sampler on numpy data with sampling rate
    """
    data = data.reshape((int(data.shape[0]/sampling_rate)), sampling_rate)
    data = sampler(data, axis = 1)
    return data

def standardizer_z_score(data, verbose = False):
    """
    data is a time seriese sequence (nd_array) numpy data
    normalize the data across time series by calculating mean and variance
    print the mean and variance if verbose is true.
    Here we standardize the data with z-score normalization.
    This is supposedly work only if data has gaussian distribution.

    Other normalization procedure to explore:
    * Median normalization
    * Sigmoid normalization
    * Tanh normalization
    """
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    if verbose:
        print("mean: ", scaler.mean_, " var: ", scaler.var_)
    return scaled_data

def normalizer_min_max(data, verbose = False):
    """
    Normalize the data in range 0 to 1. Supresses the scaler variations,
    but do not assume any gaussian distribution (this assumption is with standardization)
    """
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    if verbose:
        print("min: ", scaler.data_min_, " max: ", scaler.data_max_)
    return scaled_data

def normalizer_median_quantiles(data, verbose = False):
    """
    Normalize the data in range 75th and 25th percentile
    This normalization is robust to outliers
    """
    scaler = RobustScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    if verbose:
        print("center: ", scaler.center_, " scale: ", scaler.scale_)
    return scaled_data

def stringify(vals):
    """
    return a string version of vals (a list of object implementing __str__)
    return type: 'val1_val2_val3_ ... _valn'
    """
    return '_'.join([str(e) for e in vals])

def valuefy(strings, type_cast = None):
    """
    return a list of value, type casted by type_cast list
    return type: list if values
    By default, type cast is int
    """
    vals_string = strings.split('_')
    if type_cast == None:
        type_cast = [int]*len(vals_string)
    return [t(e) for e,t in zip(vals_string, type_cast)]

def plot_data(data, graph_labels, plot_path):
    """
    Take nd array data, graph_labels (list is plot name) and path to 
    produce plot pdf
    """
    nr_data, nr_features = data.shape[0], data.shape[1]
    fig,ax = plt.subplots(nrows = nr_features, ncols = 1, figsize = (5*nr_features,10), squeeze=False)
    for i in range(nr_features):
        ax[i][0].plot(np.linspace(0, nr_data-1, nr_data), data[:,i])
        ax[i][0].set_title(graph_labels[i])
    plt.savefig(plot_path)

def plot_data_labeled(data, graph_labels, plot_path, label, pos_label = 2.0):
    """
    Take nd array data, graph_labels (list is plot name) and path to 
    produce plot pdf. Colors the label's pos_label with red (rest in green).
    """
    nr_data, nr_features = data.shape[0], data.shape[1]
    fig,ax = plt.subplots(nrows = nr_features, ncols = 1, figsize = (10,5*nr_features), squeeze=False)
    for i in range(nr_features):
        c = ['r' if e==pos_label else 'g' for e in label]
        x = np.linspace(0, nr_data-1, nr_data)
        y = data[:,i]
        lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
        colored_lines = LineCollection(lines, colors=c, linewidths=(2,))
        ax[i][0].add_collection(colored_lines)
        ax[i][0].set_title(graph_labels[i])
        ax[i][0].autoscale_view()
    plt.savefig(plot_path)

def parse_comet_config(configs):
    """
    retrives vizier compatible configuration (only descrete values are supported)
    used for creating a comet.ml configuration dictionary
    """
    params = {}
    for k in configs.keys():
        if type(configs[k]) == list:
            params[k] = {'type': 'discrete', 'values': configs[k]}
    vizier_config = {'algorithm': configs['algorithm'],
                     'parameters': params,
                     'spec': {'metric': configs['metric'], 'objective': configs['objective']}
                     }
    return vizier_config

def get_parameters(comet_experiment, configs):
    """
    Get parameters from comet_experiment
    these parameters are corresponds to hp mentioned in configs
    """
    cf = {}
    for k in configs.keys():
        if type(configs[k]) == list:
            cf[k] = comet_experiment.get_parameter(k)
        else:
            cf[k] = configs[k]
    return cf

################################TESTING#########################################
def test_generate_permulations():
    a = [[1,2], ['a', 'b'], ["String1", "String2", "String3"]]
    b = generate_permulations(a)
    print(b)

def test_upsample_signal():
    a = np.array([[1,6], [2,7], [3,8], [4,9]])
    b = downsample_signal(a, 2)
    print(b)

def test_downsample_signal():
    a = np.array([[1,6], [2,7], [3,8], [4,9]])
    b = downsample_signal(a, 2)
    print(b)

def test_folder_exists():
    path = '/home/vishal/test_folder/test'
    folder_exists(path)

def test_standardizer_z_score():
    a = np.random.multivariate_normal(mean = [12, 3], cov = [[2, 0],[0, 5]], size = 100)
    b = standardizer_z_score(a, True)
    print(b)

def test_normalizer_min_max():
    a = np.random.multivariate_normal(mean = [12, 3], cov = [[2, 0],[0, 5]], size = 100)
    b = normalizer_min_max(a, True)
    print(b)

def test_normalizer_median_quantiles():
    a = np.random.multivariate_normal(mean = [12, 3], cov = [[2, 0],[0, 5]], size = 100)
    b = normalizer_median_quantiles(a, True)
    print(b)

def test_stringify():
    a = ['abc', 34, 1.89, 'c']
    b = stringify(a)
    print(b)

def test_valuefy():
    a = '34_1.28_500_str'
    b = valuefy(a, [int, float, int, str])
    print(b)

def test_plot_data():
    a = np.random.multivariate_normal(mean = [12, 3], cov = [[2, 0],[0, 5]], size = 100)
    plot_path = '/home/vishal/test/test.pdf'
    plot_data(a, ['d1', 'd2'], plot_path)

def test_plot_data_labeled():
    plot_path = '/home/vishal/test/test.pdf'
    n = 30
    x = np.arange(n+1)
    y = np.random.randn(n+1,1)
    print(y.shape)
    annotation = [1.0, 0.0] * 15
    plot_data_labeled(y, ['graph'], plot_path, annotation)

def test_parse_comet_config():
    config={
        'project_name': 'project',
        'workspace': 'workspace',
        ############## USE LISTS FOR HP ##############
        'learning_rate': [0.0001, 0.00001],
        'nr_epochs': [100, 200],
        'batch_size': [8, 16],
        ##############################################
        'metric': 'loss',
        'objective': 'maximize',
        'algorithm': 'bayes'
    }
    vizier_config = parse_comet_config(config)
    print(vizier_config)

def test_get_parameters():
    # No test
    pass

if __name__ == "__main__":
    #test_generate_permulations()
    #test_upsample_signal()
    #test_downsample_signal()
    #test_folder_exists()
    #test_standardizer_z_score()
    #test_normalizer_min_max()
    #test_normalizer_median_quantiles()
    #test_stringify()
    #test_valuefy()
    #test_plot_data()
    #test_plot_data_labeled()
    #test_parse_comet_config()
    #test_get_parameters()
    pass