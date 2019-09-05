import pickle 
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.collections import LineCollection




def plot_training_bars(a, model_name, dataset_name, tune_range, save_img = False):
	""" Plots means with yerr = std
	"""
	
	plt_legend = []
	ys = []
	N = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		train_results = run_dict['train_results']
		plt_legend.append(run)
		ys.append(train_results)
		N += 1
	means = np.mean(ys, axis = 0)
	std = np.std(ys, axis = 0)
	plt.errorbar(range(len(means)), means, yerr = std, fmt = '.', color = 'black', ecolor = 'gray', capsize = 3)
	plt.title('Training {} {} g {}'.format(model_name, dataset_name, tune_range))
	plt.xlabel("Iterations")
	plt.ylabel("Training Loss")
	if save_img:
		plt.savefig('htrain-{}-{}-g-{}.png'.format(dataset_name, model_name, tune_range))
	plt.show()

def plot_testing_bars(a, model_name, dataset_name, tune_range, save_img = False):
	""" Plots means with yerr = std
	"""
	
	plt_legend = []
	ys = []
	N = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		test_results = run_dict['test_results']
		plt_legend.append(run)
		ys.append(test_results)
		N += 1
	print(len(np.mean(ys, axis = 0)))
	print(len(np.std(ys, axis = 0)))
	means = np.mean(ys, axis = 0)
	std = np.std(ys, axis = 0)
	plt.errorbar(range(len(means)), means, yerr = std, fmt = '.', color = 'black', ecolor = 'gray', capsize = 3)
	plt.title('Testing {} {} g {}'.format(model_name, dataset_name, tune_range))
	plt.xlabel("Iterations")
	plt.ylabel("Test Acc %")
	if save_img:
		plt.savefig('htest-{}-{}-g-{}.png'.format(dataset_name, model_name, tune_range))
	plt.show()

'''
def plot_training_results(a):
	""" Plots multiple lines
	"""
	model_name = 'Linear'
	dataset_name = 'MNIST' 
	tune_range = '1-100'
	
	plt_legend = []
	ys = []
	N = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		train_results = run_dict['train_results']
		plt_legend.append(run)
		ys.append(train_results)
		N += 1
	
	x = np.arange(N)
	# We need to set the plot limits, they will not autoscale
	fig, ax = plt.subplots()
	ax.set_xlim(0, len(ys[0]))
	ax.set_ylim(np.min([np.min(y) for y in ys]), np.max([np.max(y) for y in ys]))

	# find best performer 
	best_run, best_loss = None, float('Inf')
	for run, run_dict in a.items():
		if run == 'settings':
			continue
		train_results = run_dict['train_results']
		if train_results[-1] < best_loss:
			best_loss = train_results[-1]
			best_run = run
	print("BEST RUN: {} @ {} Loss".format(best_run, best_loss))

	# colors is sequence of rgba tuples
	# linestyle is a string or dash tuple. Legal string values are
	#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
	#          where onoffseq is an even length tuple of on and off ink in points.
	#          If linestyle is omitted, 'solid' is used
	# See :class:`matplotlib.collections.LineCollection` for more information

	# Make a sequence of x,y pairs
	line_segments = LineCollection([np.column_stack([range(len(y)), y]) for y in ys],
	                               linewidths=(0.5, 1, 1.5, 2),
	                               linestyles='solid')
	line_segments.set_array(x)
	ax.add_collection(line_segments)
	axcb = fig.colorbar(line_segments)
	axcb.set_label('Line Number')
	ax.set_title('Line Collection with mapped colors')
	plt.sci(line_segments)  # This allows interactive changing of the colormap.
	plt.show()
'''

if __name__ == '__main__':
	'''
	with open('./scp/hypertune/{}_{}_CrossEntropyLoss_KinFwd_g_{}_E1_B1000.pickle'.format(dataset_name, model_name, tune_range), 'rb') as handle:
	    a = pickle.load(handle)
	    plot_training_bars(a)
	    plot_testing_bars(a)
	'''

