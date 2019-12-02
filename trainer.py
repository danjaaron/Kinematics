"""
xtrains kinematics models from kin_xopt
"""

# TODO: ability to use torch optimizers

from functools import reduce 
import operator 
import argparse
import torch, torchvision, time, inspect, pickle
from torchvision import transforms
import sys, os
import kinematics
import copy
from copy import deepcopy

class XTrainer:
    """ Cross-trains EVERY optimization algorithm on the specific model type
    """
    def __init__(self, loss_criterion, model_list, optimizer_list, optimizer_names, train_loader, test_loader, num_epochs = 1, flatten_dim = None):
        # global settings 
        self.flatten_dim = flatten_dim
        self.num_epochs = num_epochs
        self.optimizer_names = optimizer_names
        # device configuration
        self.device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # init instance objects
        self.model_list = [m.to(self.device) for m in model_list]
        self.optimizer_list = optimizer_list
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_criterion = loss_criterion
        # get optimizer step parameters
        self.optimizer_parameters = [list(dict(inspect.signature(o.step).parameters).keys()) for o in self.optimizer_list]
        # init dicts to store results
        self.train_results, self.test_results = {}, {}
        for opt_ind in range(len(optimizer_list)):
            self.train_results[opt_ind] = []
            self.test_results[opt_ind] = []
    
    def train(self, print_every):
        """ Trains and tests model according to epochs, storing test loss and training accuracy at each step
        """
        # set printing settings
        total_step = len(self.train_loader)
        self.print_every = print_every #int(round(total_step / 10.))
        # train
        for epoch in range(self.num_epochs):
            for i, (data, target) in enumerate(self.train_loader): 
                
                # move all to device
                data = data.to(self.device)
                if self.flatten_dim != None:
                    data = data.reshape(-1, self.flatten_dim).to(self.device)
                target = target.to(self.device)

                # iterate over optimizers and models
                for opt_ind in range(len(self.optimizer_list)):
                    # select opt and model
                    self.optimizer = self.optimizer_list[opt_ind]
                    self.model = self.model_list[opt_ind]
                    # create closure function
                    def closure():
                        outputs = self.model(data)
                        loss = self.loss_criterion(outputs, target)
                        return loss
                    # forward pass
                    loss = closure()
                    train_res, test_res = float(loss.item()), float(self.test())
                    self.train_results[opt_ind].append(train_res)
                    self.test_results[opt_ind].append(test_res)
                    # backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    # optimize with appropriate arguments
                    optimizer_parameters = self.optimizer_parameters[opt_ind]
                    if optimizer_parameters == ['closure', 'model']:
                        self.optimizer.step(closure, self.model)
                    elif optimizer_parameters == ['closure']:
                        self.optimizer.step(closure)
                    else:
                        self.optimizer.step()

                # print results
                if (i+1) % self.print_every == 0:
                    # make a minidict for printing
                    train_printdict = {self.optimizer_names[k]: v[-1] for k, v in self.train_results.items()}
                    test_printdict = {self.optimizer_names[k]: v[-1] for k, v in self.test_results.items()}
                    print(self.optimizer_names)
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: '.format(epoch+1, self.num_epochs, i+1, total_step) + str(train_printdict) + ', Val: ' + str(test_printdict))
        
        # record total step for epoch axis labeling
        self.total_step = total_step     
        
        return self.train_results, self.test_results

    def test(self):
        """ Returns test accuracy as a percentage over test loader
        """
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in self.test_loader:
                # move all to device
                data = data.to(self.device)
                if self.flatten_dim != None:
                    data = data.reshape(-1, self.flatten_dim).to(self.device)
                labels = labels.to(self.device)
                # get predictions
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # score predictions
                correct += (predicted == labels).sum().item()
            # compute test accuracy percentage
            test_accuracy = 100. * correct / total
            return test_accuracy
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Compare Kinematics algorithms')
    parser.add_argument('kin_versions')
    parser.add_argument('-d', '--dataset', default = 'MNIST')
    parser.add_argument('-m', '--model', default = 'Linear')
    parser.add_argument('-e', '--num_epochs', type = int, default = 1)
    parser.add_argument('-b', '--batch_size', type = int, default = 128)
    parser.add_argument('-p', '--print_every', type = int, default = 1)
    args = parser.parse_args()

    # init loss criterion
    criterion = getattr(torch.nn, 'CrossEntropyLoss')()

    # get model, optimizer and data classes
    model = [getattr(module, args.model) for module in [torch.nn, torchvision.models] if hasattr(module, args.model)][0]
    kin_classes = [m[1] for m in inspect.getmembers(sys.modules['kinematics'], inspect.isclass) if hasattr(m[1], 'alias') and m[1].alias in args.kin_versions.upper()]
    kin_classes.append(torch.optim.Adam)
    dataset = getattr(torchvision.datasets, args.dataset)
    assert(model and kin_classes)

    # create dataloaders
    train_dataset = dataset(root = './datasets/', train = True, download = True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size, 
                                            shuffle=True)
    test_dataset = dataset(root = './datasets/', train = False, download = True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=args.batch_size, 
                                            shuffle=True)
    num_classes = len(train_dataset.classes)

    # create models
    model_init = lambda x: x()
    flatten_dim = None
    if args.model == 'Linear':
        in_features = reduce(operator.mul, train_dataset.data.shape[1:])
        flatten_dim = in_features
        model_init = lambda x: x(in_features, num_classes)
    model_list = [model_init(model) for _ in kin_classes]

    # create optimizers
    opt_list = [k(model_list[k_idx].parameters()) for k_idx, k in enumerate(kin_classes)]
    opt_names = [type(o).__name__ for o in opt_list]

    # init trainer 
    xTrainer = XTrainer(criterion, model_list, opt_list, opt_names, train_loader, test_loader, num_epochs = args.num_epochs, flatten_dim = flatten_dim)

    # get results 
    train_results, test_results = xTrainer.train(args.print_every)

    # prepare results for pickling 
    results_to_save = {opt_names[o]: {'train_results': train_results[o], 'test_results': test_results[o]} for o in range(len(model_list))}
    results_to_save['args'] = args
    results_to_save['iters_per_epoch'] = xTrainer.total_step
    
    # pickle results 
    pickle_path = './results/'
    file_string = str(args.kin_versions)
    pickle_filename = pickle_path + file_string + str(len([_ for _ in os.listdir(pickle_path) if file_string == _.split('.')[0][:-1]])) + '.pickle'
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(results_to_save, handle, protocol = pickle.HIGHEST_PROTOCOL)

    