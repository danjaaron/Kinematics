from data_loader import *
import pickle 
'''
Plot hypertune values over g
'''

dname = 'CIFAR10'
mname = 'resnet18'

with open('g_{}_{}.pickle'.format(dname, mname), 'rb') as handle:
    gdict = pickle.load(handle)
    g_vals = list(gdict.keys())
    loss_vals = list(gdict.values())
    # get mean, std 
    loss_mean = np.mean(loss_vals, axis = 0)
    loss_std = np.std(loss_vals, axis = 0)
    assert(loss_mean.size == len(loss_vals[0]))
    assert(loss_mean.size == loss_std.size)
    # plot 
    plt.errorbar(list(range(len(loss_mean))), loss_mean, yerr = loss_std)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Hypertune (g) {} {}".format(dname, mname))
    plt.show()