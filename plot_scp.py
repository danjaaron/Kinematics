import pickle 
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


'''
Plots and saves all pickle files in specified source directory to target dir
'''

def pickle_load(pickle_load_path, filename):
	with open(pickle_load_path + filename, 'rb') as handle:
	    a = pickle.load(handle)
	    return a 

def get_all_sources(source_path, target_path):
	onlyfiles = [f for f in listdir(source_path) if isfile(join(source_path, f)) and '.pickle' in f]
	for f in onlyfiles:
		a = pickle_load(source_path, f)
		plot_training_results(a, f, target_path)
		plot_testing_results(a, f, target_path)


def plot_training_results(a, filename, fig_save_path, save_fig = True):
	plt_legend = []
	i = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		train_results = run_dict['train_results']
		plt.plot(range(len(train_results)), train_results)
		plt_legend.append(run)
		# add g if available 
		if 'g' in run_dict.keys():
			g_vals = run_dict['g']
			plt.plot(range(len(g_vals)), g_vals)
			plt_legend.append('g')
		i += 1
	plt.title('TRAIN ' + filename)
	plt.legend(plt_legend)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	if save_fig:
		plt.savefig(fig_save_path + 'gtrain_' + filename + '.png')
	plt.show()

def plot_testing_results(a, filename, fig_save_path, save_fig = True):
	plt_legend = []
	i = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		test_results = run_dict['test_results']
		plt.plot(range(len(test_results)), test_results)
		plt_legend.append(run)
		# add g if available 
		if 'g' in run_dict.keys():
			g_vals = run_dict['g']
			plt.plot(range(len(g_vals)), g_vals)
			plt_legend.append('g')
		i += 1
	plt.title('TEST ' + filename)
	plt.legend(plt_legend)
	plt.xlabel('Iterations')
	plt.ylabel('Test Acc %')
	if save_fig:
		plt.savefig(fig_save_path + 'gtest_' + filename + '.png')
	plt.show()

if __name__ == '__main__':
	source_path = './scp/gtrain/'
	target_path = './gtrain_figs/'
	get_all_sources(source_path, target_path)