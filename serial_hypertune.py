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
        "flatten_dim": 784,
        "num_classes": 10,
        "dataset": 'MNIST',
        "model": 'Linear',
        "loss": 'CrossEntropyLoss',
        "optimizers": ['KinFwd'],
        "tune_param": 'g',
        "tune_range": list(np.arange(1.0, 2.1, 1.0))
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

    # get model function 
    if hasattr(torchvision.models, settings_dict['model']):
        # torchvision model, e.g. resnet 18
        model_attr = getattr(torchvision.models, settings_dict['model']) #call: ()
    elif hasattr(torch.nn, settings_dict['model']):
        # torch.nn layer, e.g. Linear
        model_attr = getattr(torch.nn, settings_dict['model']) #call: (settings_dict['flatten_dim'], settings_dict['num_classes'])
    else:
        # model not implemented
        raise NotImplementedError

    # get optimizer function 
    opt_name = settings_dict['optimizers'][0]
    if hasattr(torch.optim, opt_name):
        opt_attr = getattr(torch.optim, opt_name)
    elif hasattr(kinematics, opt_name):
        opt_attr = getattr(kinematics, opt_name)
    else:
        raise NotImplementedError

    # iterate over tune parameter values
    for tune_ind in range(len(settings_dict['tune_range'])):
        tune_value = settings_dict['tune_range'][tune_ind]
        print("TUNING {}".format(tune_value))
        assert(int(tune_value) != 0) # for results filename
        # init model
        if settings_dict['flatten_dim'] == None:
            model = model_attr()
        else:
            model = model_attr(settings_dict['flatten_dim'], settings_dict['num_classes'])
        # init opt 
        opt = opt_attr(model.parameters())
        # set tuned value
        param_to_tune = settings_dict['tune_param']
        assert(hasattr(opt, param_to_tune))
        setattr(opt, param_to_tune, tune_value)
        # init trainer 
        xTrainer = XTrainer([model], criterion, [opt], settings_dict['optimizers'], train_loader, test_loader, flatten_dim = settings_dict['flatten_dim'], num_epochs = settings_dict['num_epochs'])
        # get results 
        train_results, test_results = xTrainer.train()
        # prepare results for pickling 
        results_to_save = {opt_name: {'train_results': train_results, 'test_results': test_results, settings_dict['tune_param']: tune_value}}
        results_to_save['settings'] = settings_dict
        # pickle results 
        pickle_filename = results_path + '{}_{}_{}_{}_{}_{}_E{}_B{}.pickle'.format(settings_dict['dataset'], settings_dict['model'], settings_dict['loss'], opt_name, settings_dict['tune_param'], int(tune_value), settings_dict['num_epochs'], settings_dict['batch_size'])
        with open(pickle_filename, 'wb') as handle:
            pickle.dump(results_to_save, handle, protocol = pickle.HIGHEST_PROTOCOL)

        

    print('initialized successfully')
    print(settings_dict)

    

    
    
    
    
