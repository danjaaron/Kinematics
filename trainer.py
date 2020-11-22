"""
xtrains kinematics models from kin_xopt
TODO: print output in a refreshing screen rather than scrolling
TODO progress bar

#TODO run fig1 IMDB command and download imdb dataset
"""

import pandas as pd

import argparse
import os
import sys
from datetime import datetime

import inspect
import pickle
import torch
import torchvision
from torchvision import transforms
import torchnlp
from torchnlp import datasets
import functools
from functools import reduce
import operator

import kinematics


class XTrainer:
    """ Cross-trains EVERY optimization algorithm on the specific model type
    """
    def __init__(self, loss_criterion, model_list, optimizer_list, train_loader, test_loader, args, output_filename,
                 num_epochs=1, flatten_dim=None):

        self.output_filename = output_filename
        self.epoch, self.iter = 0, 0
        self.total_step = 0
        self.batch = 0

        # global settings 
        self.flatten_dim = flatten_dim
        self.num_epochs = num_epochs
        self.optimizer_names = [type(o).__name__ for o in optimizer_list]
        self.args = args
        # device configuration
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
        # init instance objects
        self.model_list = [m.to(self.device) for m in model_list]
        self.optimizer_list = optimizer_list
        self.num_optimizers = len(self.optimizer_list)
        self.train_loader = train_loader
        self.num_batches = len(self.train_loader)
        self.test_loader = test_loader
        self.loss_criterion = loss_criterion
        # get optimizer step parameters
        self.optimizer_parameters = [list(dict(inspect.signature(o.step).parameters).keys()) for o in
                                     self.optimizer_list]
        # init dicts to store results
        self.train_results, self.test_results, self.gravity_results = {}, {}, {}
        self.log_dicts = [self.train_results, self.test_results, self.gravity_results]
        self.log_dicts_names = ['train_loss', 'test_acc', 'gravity']
        for opt_ind in range(len(optimizer_list)):
            for log_dict in self.log_dicts:
                log_dict[opt_ind] = []
        self.results_df = pd.DataFrame()

    def train(self):
        """ Trains and tests model according to epochs, storing train loss and test loss at each step
            .... checkpoints on every print
        """

        # epoch loop
        while self.epoch <= self.args.num_epochs:

            # batch loop
            for batch_num, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                if self.flatten_dim is not None:
                    data = data.reshape(-1, self.flatten_dim).to(self.device)
                target = target.to(self.device)

                # optimizer loop
                for opt_ind in range(len(self.optimizer_list)):

                    # select opt and model
                    self.optimizer = self.optimizer_list[opt_ind]
                    self.optimizer_name = self.optimizer_names[opt_ind]
                    self.model = self.model_list[opt_ind]
                    self.opt_ind = opt_ind

                    # create closure function
                    def closure():
                        outputs = self.model(data)
                        loss = self.loss_criterion(outputs, target)
                        return loss

                    # forward pass
                    self.loss = closure()
                    self.loss.backward()

                    # optimizer step
                    optimizer_parameters = self.optimizer_parameters[opt_ind]
                    if optimizer_parameters == ['closure', 'model']:
                        self.optimizer.step(closure, self.model)
                    elif optimizer_parameters == ['closure']:
                        self.optimizer.step(closure)
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                    # optimizer checkpoint
                    opt_bool = bool(self.args.verbosity == 'opt')
                    self.checkpoint(_update=True, _print=opt_bool, _export=opt_bool)

                # batch checkpoint
                batch_bool = bool(self.args.verbosity == 'batch')
                self.checkpoint(_update=False, _print=batch_bool, _export=batch_bool)
                self.batch += 1

            # epoch checkpoint
            epoch_bool = bool(self.args.verbosity == 'epoch')
            self.checkpoint(_update=False, _print=epoch_bool, _export=epoch_bool)
            self.epoch += 1
            self.batch = 0

        # record total step for epoch axis labeling
        self.total_step = self.total_step

        return self.train_results, self.test_results



    def checkpoint(self, _update=False, _print=True, _export=False):
        """ Saves a log of current results
        """
        if _update:
            # new pandas way
            g = (hasattr(self.optimizer, 'g') and float(self.optimizer.g))
            self.results_df = self.results_df.append({
                'batch': self.batch,
                'epoch': self.epoch,
                'optimizer': self.optimizer_name,
                'train_loss': float(self.loss.item()),
                'test_loss': float(self.test()),
                'gravity':  g
            }, ignore_index=True)

        if _print:
            print(self.results_df[-self.num_optimizers:])

        if _export:
            print('exporting')
            # save results
            self.results_df.to_csv(self.output_filename)
            # save optimizer configuration
            config_file = self.output_filename[:-4] + '.pickle'
            pickle.dump(self.optimizer_list, open(config_file, 'wb+'))
            # confirm
            print('. exported to ' + config_file[:-7])


    def test(self):
        """ Returns average test loss
        """

        with torch.no_grad():
            loss_list = []
            for data, labels in self.test_loader:
                # move all to device
                data = data.to(self.device)
                if self.flatten_dim != None:
                    data = data.reshape(-1, self.flatten_dim).to(self.device)
                labels = labels.to(self.device)
                # get predictions
                outputs = self.model(data)
                loss_list.append(self.loss_criterion(outputs, labels).item())
            # compute avg test loss
            test_loss = sum(loss_list)/len(loss_list)
            return test_loss


def setup_results():
    """ setup a directory for the results """
    # get results number
    num_results = len([k for k in os.listdir('./results/') if 'results' in k])
    new_results_dir = './results/results{}/'.format(num_results)
    os.mkdir(new_results_dir)
    args_path = new_results_dir+'config.txt'
    with open(args_path, 'w+') as f:
        f.write(str(datetime.now())+"\n")
        for sys_arg in sys.argv:
            f.write("... {}\n".format(sys_arg))
    print("... dumped args to {}".format(args_path))
    return new_results_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compare Kinematics algorithms')
    # parser.add_argument('kin_versions')

    # command line specification
    parser.add_argument('-d', '--dataset', default='MNIST')
    parser.add_argument('-m', '--model', default='Linear')
    parser.add_argument('-e', '--num_epochs', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-x', '--comparison_list', nargs='*') # torch optimizers to compare to
    # logging options
    parser.add_argument('-v', '--verbosity', choices=['opt', 'batch', 'epoch'], default='epoch')
    # parser.add_argument('--output_file', nargs='?', type=argparse.FileType('w'), default='./results/test.csv')
    # loss criterion
    parser.add_argument('-l', '--loss_criterion', choices=['CrossEntropyLoss', 'NLLLoss'], default='CrossEntropyLoss')

    args = parser.parse_args()
    print(args)

    results_dir = setup_results()
    # args['output_file'] = results_dir+'results.csv'

    # extract optimizers to compare against
    xopt_names = args.comparison_list
    # get optimizer classes
    xopts = [getattr(torch.optim, x_name) for x_name in xopt_names]
    xopts.append(kinematics.Kin)

    # setup
    # criterion = getattr(torch.nn, 'CrossEntropyLoss')()
    criterion = getattr(torch.nn, args.loss_criterion)()

    model_class = [getattr(module, args.model) for module in [torch.nn, torchvision.models] if hasattr(module, args.model)][0]
    dataset = None
    _src = None
    for dataset_source in [torchvision.datasets, torchnlp.datasets]:
        _src = dataset_source
        try:
            # dataset = getattr(torchvision.datasets, args.dataset)
            dataset = getattr(dataset_source, args.dataset)
        except:
            pass
        if dataset:
            break
    assert dataset, 'FATAL: could not find dataset source for {}'.format(args.dataset)
    print('OK: found dataset {}'.format(dataset))

    _src_from_torchvision = 'torchvision' in str(_src)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create dataloaders
    if _src_from_torchvision:
        train_dataset = dataset(root='./datasets/', train=True, download=True, transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
        test_dataset = dataset(root='./datasets/', train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True)
    else:
        train_dataset = dataset(train=True)
        test_dataset = dataset(train=False)
        print(dir(train_dataset))


    num_optimizers = len(xopts)
    flatten_dim = None
    if args.model == 'Linear':
        in_features = reduce(operator.mul, train_dataset.data.shape[1:])
        flatten_dim = in_features
        num_classes = len(train_dataset.classes)
        model_list = [model_class(in_features, num_classes).to(device) for _ in range(num_optimizers)]
    else:
        model_list = [model_class().to(device) for _ in range(num_optimizers)]
    opt_list = []
    for opt_ind, opt_class in enumerate(xopts):
        optimizer = opt_class(model_list[opt_ind].parameters())
        opt_list.append(optimizer)

    # train
    output_filename = results_dir+'results.csv'
    xTrainer = XTrainer(criterion, model_list, opt_list, train_loader, test_loader, args,
                        num_epochs=args.num_epochs, flatten_dim=flatten_dim, output_filename=output_filename)
    xTrainer.train()