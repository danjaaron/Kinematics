from data_loader import *
import pickle 
'''
Hypertune over values of grav accel 
'''

dname = 'CIFAR100'
mname = 'resnet50'

train_dict = {}
g_range = np.arange(0.0, 101.0, 5.0)
print("HYPERTUNING G ON {} {}".format(dname, mname))
for g in g_range:
	if g == 0.:
		continue
	dl = DataLoader(dname, mname, "Kinematics")
	dl.optimizer.g = float(g)
	print(dl.optimizer.g)
	dl.train()
	train_dict[g] = list(dl.training_loss)
	print(g)
	print(train_dict)
with open('g_{}_{}.pickle'.format(dname, mname), 'wb') as handle:
    pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


