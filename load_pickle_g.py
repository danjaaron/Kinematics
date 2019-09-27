import pickle 
import matplotlib.pyplot as plt

fig_save_path = './gtrain/'
pickle_load_path = './gtrain/'
filename_version = 'dub_'

dataset = 'MNIST'
model = 'Linear'
batch_size = 'B10000'
opts = 'KbAm'
epochs = 'E1'
save_fig = True



dataset = filename_version + dataset
filename = '{}_{}_CrossEntropyLoss_{}_{}_{}'.format(dataset, model, opts, epochs, batch_size)
def plot_training_results(a):
	plt_legend = []
	i = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		train_results = run_dict['train_results']
		plt.plot(range(len(train_results)), train_results)
		plt_legend.append(run)
		i += 1
	plt.title(filename + '.pickle')
	plt.legend(plt_legend)
	plt.xlabel('Iterations')
	plt.ylabel('Loss')
	if save_fig:
		plt.savefig(fig_save_path + 'xtrain_' + filename + '.png')
	plt.show()

def plot_testing_results(a):
	plt_legend = []
	i = 0
	for run, run_dict in a.items():
		if run == 'settings':
			continue 
		test_results = run_dict['test_results']
		plt.plot(range(len(test_results)), test_results)
		plt_legend.append(run)
		i += 1
	plt.title(filename + '.pickle')
	plt.legend(plt_legend)
	plt.xlabel('Iterations')
	plt.ylabel('Test Acc %')
	if save_fig:
		plt.savefig(fig_save_path + 'xtest_' + filename + '.png')
	plt.show()

with open(pickle_load_path + filename + '.pickle', 'rb') as handle:
    a = pickle.load(handle)
    #plot_training_results(a)
    #plot_testing_results(a)
    print(a)
