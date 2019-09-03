"""
Xtrainer.py -- cross-trains multiple optimization algorithms in parallel (on a single model type) for comparative results
    *** implement and investigate past (psuedo-)parallel hyper-param tuning 
"""

import torch, torchvision, time, inspect, kinematics, pickle
from torchvision import transforms

class XTrainer:
    """ Cross-trains EVERY optimization algorithm on the specific model type
    """
    def __init__(self, model_list, loss_criterion, optimizer_list, optimizer_names, train_loader, test_loader, num_epochs = 1, flatten_dim = None):
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
    
    def train(self):
        """ Trains and tests model according to epochs, storing test loss and training accuracy at each step
        """
        # set printing settings
        total_step = len(self.train_loader)
        self.print_every = 1 #int(round(total_step / 10.))
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
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: '.format(epoch+1, self.num_epochs, i+1, total_step) + str(train_printdict) + ', Val: ' + str(test_printdict))
                           
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
    
    """ Example of crosstrainer usage with torchvision models and datasets
        (!) TODO: implement parallel hypertuning -- configure tons of opt algos with diff hyperparam settings (** has this been done before?)
            ... will need to change filename saving, etc.
    """

    settings_dict = {
        "batch_size": 1000,
        "num_epochs": 1,
        "shuffle_train": True,
        "flatten_dim": 784,
        "num_classes": 10,
        "dataset": 'MNIST',
        "model": 'Linear',
        "loss": 'CrossEntropyLoss',
        "optimizers": ['Adam', 'KinFwd'] #, 'Adadelta', 'RMSprop'] 

    }

    # init datasets
    train_dataset = getattr(torchvision.datasets, settings_dict['dataset'])(root='../../data', 
                                               train=True, 
                                               transform=transforms.ToTensor(), 
                                               download=True)

    test_dataset = getattr(torchvision.datasets, settings_dict['dataset'])(root='../../data', 
                                              train=False, 
                                              transform=transforms.ToTensor())

    # init data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=settings_dict['batch_size'], 
                                               shuffle=settings_dict['shuffle_train'])

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=settings_dict['batch_size'], 
                                              shuffle=False)

    # init loss criterion
    criterion = getattr(torch.nn, settings_dict['loss'])()

    # init models
    if hasattr(torchvision.models, settings_dict['model']):
        # torchvision model, e.g. resnet 18
        model_list = [getattr(torchvision.models, settings_dict['model'])().to('cuda') for _ in settings_dict['optimizers']]
    elif hasattr(torch.nn, settings_dict['model']):
        # torch.nn layer, e.g. Linear
        model_list = [getattr(torch.nn, settings_dict['model'])(settings_dict['flatten_dim'], settings_dict['num_classes']).to('cuda') for _ in settings_dict['optimizers']]
    else:
        # model not implemented
        raise NotImplementedError
    
    # init optimizers and names
    opt_save_str = '' # stem for the pickle filename string which identifies tested opt algos
    optimizer_list, optimizer_names = [], []
    for opt_ind in range(len(settings_dict['optimizers'])):
        # get opt model and name
        opt_model = model_list[opt_ind]
        opt_name = settings_dict['optimizers'][opt_ind]
        opt_save_str = opt_save_str + (opt_name[0].upper() + opt_name[-1].lower())
        optimizer_names.append(opt_name)
        # get opt method
        if hasattr(torch.optim, opt_name):
            opt_attr = getattr(torch.optim, opt_name)
        elif hasattr(kinematics, opt_name):
            opt_attr = getattr(kinematics, opt_name)
        else:
            raise NotImplementedError
        # init optimizer object
        optimizer_list.append(opt_attr(opt_model.parameters()))

    # init trainer 
    xTrainer = XTrainer(model_list, criterion, optimizer_list, optimizer_names, train_loader, test_loader, flatten_dim = settings_dict['flatten_dim'], num_epochs = settings_dict['num_epochs'])

    # get results 
    train_results, test_results = xTrainer.train()

    # prepare results for pickling 
    results_to_save = {optimizer_names[o]: {'train_results': train_results[o], 'test_results': test_results[o]} for o in range(len(model_list))}
    results_to_save['settings'] = settings_dict
    
    # pickle results 
    pickle_path = './xtrain_results/'
    pickle_filename = pickle_path + '{}_{}_{}_{}_E{}_B{}.pickle'.format(settings_dict['dataset'], settings_dict['model'], settings_dict['loss'], opt_save_str, settings_dict['num_epochs'], settings_dict['batch_size'])
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(results_to_save, handle, protocol = pickle.HIGHEST_PROTOCOL)
