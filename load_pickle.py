import pickle 
import matplotlib.pyplot as plt

dataset = 'CIFAR10'
model = 'resnet18'
batch_size = 'B128'
opts = 'AmKd'
epochs = 'E1'
save_fig = True
filename = '{}_{}_CrossEntropyLoss_{}_E1_{}'.format(dataset, model, opts, batch_size)
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
		plt.savefig('xtrain_' + filename + '.png')
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
		plt.savefig('xtest_' + filename + '.png')
	plt.show()

with open('./scp/xtrain/' + filename + '.pickle', 'rb') as handle:
    a = pickle.load(handle)
    plot_training_results(a)
    plot_testing_results(a)

