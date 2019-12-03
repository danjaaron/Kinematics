import pickle, sys, os
import matplotlib
from matplotlib import pyplot as plt

def read_pickle(filename):
	with open('./results/'+filename+'.pickle', 'rb') as o:
		r = pickle.load(o)
		return r

def plot_results(results):
	legends = list(results.keys())
	args = vars(results['args'])
	data_str = args['dataset']
	model_str = args['model']

	title_settings = " {} {}".format(data_str, model_str)
	plt.title('Train Results'+title_settings)
	for r, rdict in results.items():
		if 'Kin' in r or r == 'Adam':
			plt.plot(rdict['train_results'])
			plt.ylabel('XEntropy Loss')
			plt.xlabel('Iterations [{} epochs]'.format(args['num_epochs']))
	plt.legend(legends)
	plt.show()
	plt.title('Test Results'+title_settings)
	for r, rdict in results.items():
		if 'Kin' in r or r == 'Adam':
			plt.plot(rdict['test_results'])
			plt.ylabel('Val %')
			plt.xlabel('Iterations [{} epochs]'.format(args['num_epochs']))

	plt.legend(legends)
	plt.show()


if __name__ == '__main__':
	r = read_pickle('KGM0')
	plot_results(r)