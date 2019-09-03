"""
hypertune_kinfwd.py -- implements xtrainer.py to hypertune "g" for kinfwd algorithm 
... differs from xtrainer __main__ only in optimizer and model initiation (creates a new model / optimizer for each tune value)
"""

from xtrainer import * 
import numpy as np

if __name__ == '__main__':
    torch.cuda.empty_cache()
    results_path = './hypertune_results/'
    

    settings_dict = {
        "batch_size": 1000,
        "num_epochs": 1,
        "shuffle_train": True,
        "flatten_dim": None,
        "num_classes": 10,
        "dataset": 'CIFAR10',
        "model": 'resnet18',
        "loss": 'CrossEntropyLoss',
        "optimizers": ['KinFwd'],
        "tune_param": 'g',
        "tune_range": list(np.arange(31.0, 100.1, 1.0))
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
        model_list = [getattr(torchvision.models, settings_dict['model'])().to('cuda') for _ in settings_dict['tune_range']]
    elif hasattr(torch.nn, settings_dict['model']):
        # torch.nn layer, e.g. Linear
        model_list = [getattr(torch.nn, settings_dict['model'])(settings_dict['flatten_dim'], settings_dict['num_classes']).to('cuda') for _ in settings_dict['tune_range']]
    else:
        # model not implemented
        raise NotImplementedError
    
    # init optimizers and names
    optimizer_list, optimizer_names = [], []
    opt_name = settings_dict['optimizers'][0]
    for tune_ind in range(len(settings_dict['tune_range'])):
        tune_value = settings_dict['tune_range'][tune_ind]
        # get opt model and name
        opt_model = model_list[tune_ind]
        optimizer_names.append(opt_name + '{:.1f}'.format(tune_value))
        # get opt method
        if hasattr(torch.optim, opt_name):
            opt_attr = getattr(torch.optim, opt_name)
        elif hasattr(kinematics, opt_name):
            opt_attr = getattr(kinematics, opt_name)
        else:
            raise NotImplementedError
        # init optimizer object
        opt_obj = opt_attr(opt_model.parameters())
        # set tuned value
        param_to_tune = settings_dict['tune_param']
        assert(hasattr(opt_obj, param_to_tune))
        setattr(opt_obj, param_to_tune, tune_value)
        # add tuned optimizer to list
        optimizer_list.append(opt_obj)

    print('initialized successfully')
    print(settings_dict)

    # init trainer 
    xTrainer = XTrainer(model_list, criterion, optimizer_list, optimizer_names, train_loader, test_loader, flatten_dim = settings_dict['flatten_dim'], num_epochs = settings_dict['num_epochs'])
    # get results 
    train_results, test_results = xTrainer.train()

    # prepare results for pickling 
    results_to_save = {optimizer_names[o]: {'train_results': train_results[o], 'test_results': test_results[o], settings_dict['tune_param']: settings_dict['tune_range'][o]} for o in range(len(model_list))}
    results_to_save['settings'] = settings_dict
    
    # pickle results 
    pickle_filename = results_path + '{}_{}_{}_{}_{}_{}-{}_E{}_B{}.pickle'.format(settings_dict['dataset'], settings_dict['model'], settings_dict['loss'], opt_name, settings_dict['tune_param'], int(settings_dict['tune_range'][0]), int(settings_dict['tune_range'][-1]), settings_dict['num_epochs'], settings_dict['batch_size'])
    with open(pickle_filename, 'wb') as handle:
        pickle.dump(results_to_save, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
