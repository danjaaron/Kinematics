import pickle 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.collections import LineCollection
import os 
import pickle_plot 
import sys
''' Concatenates all specified serial hypertune pickle files into one plot, no membatch e.g. 'g_1-100'
'''

settings = {
    'opt_name_stem': 'KinFwd',
    'dataset_name': 'MNIST',
    'model_name': 'Linear', 
    'batch_name': 'B128',
    'epoch_name': 'E1',
    'save_img': False
}


for _ in range(len(sys.argv[1:])):
    cl_in = sys.argv[1:][_]
    settings_key = list(settings.keys())[_ + 1] # increment bc KinFwd is default 
    settings[settings_key] = cl_in
    print('"{}" set to "{}" from run config'.format(settings_key, cl_in))
    if settings_key == 'save_img':
        settings[settings_key] = bool(cl_in)

print(settings)

model_name = settings['model_name']
dataset_name = settings['dataset_name']


load_path = './scp/hypertune/'

# dict to plot 
plot_dict = {}
tune_ranges = []
train_results = []
test_results = []
for h in os.listdir(load_path):
    if (model_name in h) and (dataset_name in h) and (not '-' in h) and (settings['batch_name'] in h) and (settings['epoch_name'] in h):
        with open(load_path+h, 'rb') as handle:
            a = pickle.load(handle)
            print(h)
            for k, v in a.items():
                if k == 'settings':
                    continue
                tune_ranges.append(v['g'])
                plot_dict_key = k
                if not str(v['g']) in k:
                    plot_dict_key = str(k) + str(v['g'])
                plot_dict[plot_dict_key] = {}
                if type(v['train_results']) is dict:
                    plot_dict[plot_dict_key]['train_results'] = v['train_results'][0]
                    plot_dict[plot_dict_key]['test_results'] = v['test_results'][0]
            

tune_str = '{}-{}-{}-{}'.format(int(min(tune_ranges)), int(max(tune_ranges)), settings['epoch_name'], settings['batch_name'])
print('passing: ' + tune_str)
pickle_plot.plot_training_bars(plot_dict, model_name, dataset_name, tune_str, settings['save_img'])
pickle_plot.plot_testing_bars(plot_dict, model_name, dataset_name, tune_str, settings['save_img'])