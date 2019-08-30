import torch 
import torch.nn as nn 
from models import *
import matplotlib.pyplot as plt 
import time 
import numpy as np 
import torchvision.models as tmodels
from torchvision import transforms
import kinfwd_optim
import scipy 
# torchvision models: https://pytorch.org/docs/stable/torchvision/models.html
# tutorial with models: https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

class DataLoader:
    """
    Loads given dataset with specified model, optimizer, and loss
    Args: *_name are strings 
    """
    def __init__(self, dataset_name, model_name, opt_name, loss_name = 'CrossEntropyLoss'):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("begin initialization")
        # Set supported arguments
        self.supported_datasets = dir(torchvision.datasets)
        self.supported_losses = ['CrossEntropyLoss']
        self.supported_models = ['NN', 'Conv', 'AlexNet'] + list(dir(torchvision.models))
        self.supported_optims = ['KinFwdAdj', 'Kinematics', 'Adam'] #, 'SGD'] #, 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'AdamW']
        # Universal settings 
        self.num_epochs = 1
        self.batch_size = 1000 #5000
        self.shuffle_train = False
        self.num_trials = 30
        self.print_every = 10

        self.transform = transform = transforms.Compose([
            transforms.RandomCrop(36, padding=4),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        

        # Check args 
        # assert(dataset_name in self.supported_datasets)
        # assert(loss_name in self.supported_losses)
        # assert(opt_name in self.supported_optims)
        # assert(model_name in self.supported_models)

        # Store names 
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.opt_name = opt_name
        self.loss_name = loss_name
        # Plotting objects 
        self.training_loss = []
        self.val_acc = [] # for plotting val acc throughout training 
        self.test_accuracy = None 
        self.training_dict = {}
        self.testing_dict = {}
        self.time_dict = {}
        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'

        # load specific data 
        if dataset_name == 'CIFAR10':
            self.load_CIFAR10()
        elif dataset_name == 'CIFAR100':
            self.load_CIFAR100()
        elif dataset_name == 'MNIST':
            self.load_MNIST()
        elif dataset_name == 'ImageNet':
            self.load_ImageNet()
        else:
            raise NotImplementedError
        # load loss
        self.criterion = nn.CrossEntropyLoss()
        # load model 
        if model_name == 'NN':
            self.model = NeuralNet(self.input_size, self.hidden_size, self.num_classes).to(self.device)
        elif model_name == 'Conv':
            self.model = ConvNet(in_channels = self.input_channels, conv_dim = self.conv_dim, num_classes = self.num_classes).to(self.device)
        elif model_name == 'AlexNet':
            self.model = tmodels.alexnet()
        elif model_name == 'AlexNet100':
            self.model = AlexNet100().to(self.device)
        elif model_name == 'resnet18':
            self.model = tmodels.resnet18().to(self.device)
        elif model_name == 'resnet50':
            self.model = tmodels.resnet50().to(self.device)
        elif model_name == 'Linear':
            self.model = nn.Linear(self.input_size, self.num_classes)
        else:
            raise NotImplementedError
        # load optimizer 
        if opt_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        elif opt_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate)
        elif opt_name == 'Kinematics':
            #self.optimizer = kinematics_optim.Kinematics(self.model.parameters())
            self.optimizer = kinfwd_optim.KinFwd(self.model.parameters())
        elif opt_name == 'KinAdj':
            self.optimizer = kinadj_optim.KinAdj(self.model.parameters())
        elif opt_name == 'KinFwdAdj':
            self.optimizer = kinfwdadj_optim.KinFwdAdj(self.model.parameters())
        elif opt_name == 'Adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters())
        elif opt_name == 'Adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters())
        elif opt_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters())
        elif opt_name == 'Adamax':
            self.optimizer = torch.optim.Adamax(self.model.parameters())
        elif opt_name == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters())
        else:
            raise NotImplementedError
        print("completed initialization")

    def load_CIFAR10(self):
        # Hyper parameters
        self.input_size = 32*32*3
        self.hidden_size = None 
        self.num_classes = 10
        self.learning_rate = 0.001
        self.input_channels = 3
        self.conv_dim = 2048
        self.dataset = 'CIFAR10'

        # MNIST dataset
        self.train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                   train=True, 
                                                   transform=transforms.ToTensor(),
                                                   download=True)

        self.test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                                  train=False, 
                                                  transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size, 
                                                   shuffle=self.shuffle_train)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.batch_size, 
                                                  shuffle=False)
    def load_ImageNet(self):
        # Hyper parameters
        self.input_size = 32*32*3
        self.hidden_size = None 
        self.num_classes = 10
        self.learning_rate = 0.001
        self.input_channels = 3
        self.conv_dim = 2048
        self.dataset = 'ImageNet'

        # MNIST dataset
        self.train_dataset = torchvision.datasets.ImageNet(root='../../data/',
                                                   train=True, 
                                                   transform=self.transform,
                                                   download=True)

        self.test_dataset = torchvision.datasets.ImageNet(root='../../data/',
                                                  train=False, 
                                                  transform=self.transform)

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size, 
                                                   shuffle=self.shuffle_train)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.batch_size, 
                                                  shuffle=False)

    def load_CIFAR100(self):
        # Hyper parameters
        self.input_size = 32*32*3
        self.hidden_size = 500
        self.num_classes = 100
        self.learning_rate = 0.001
        self.input_channels = 3
        self.conv_dim = 2048

        # MNIST dataset
        self.train_dataset = torchvision.datasets.CIFAR100(root='../../data/',
                                                   train=True, 
                                                   transform=self.transform,
                                                   download=True)

        self.test_dataset = torchvision.datasets.CIFAR100(root='../../data/',
                                                  train=False, 
                                                  transform=self.transform)

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.batch_size, 
                                                   shuffle=self.shuffle_train)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.batch_size, 
                                                  shuffle=False)

    def load_MNIST(self):
        # Hyper-parameters 
        self.input_size = 784
        self.hidden_size = 500
        self.num_classes = 10
        self.learning_rate = 0.001
        self.input_channels = 1
        self.conv_dim = 7*7*32
        self.dataset = 'MNIST'

        # MNIST dataset 
        self.train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                                   train=True, 
                                                   transform=transforms.ToTensor(),  
                                                   download=True)

        self.test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                                  train=False, 
                                                  transform=transforms.ToTensor())

        # Data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                   batch_size=self.batch_size, 
                                                   shuffle=self.shuffle_train)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                  batch_size=self.batch_size, 
                                                  shuffle=False)

    def train(self):
        old_model_state_dict = self.model.state_dict()
        # Train the model
        total_step = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                # Move tensors to the configured device
                
                '''
                if self.input_size is None:
                    images = images.to(self.device)
                else:
                    images = images.reshape(-1, self.input_size).to(self.device)
                '''

                
                old_optimizer_state_dict = self.optimizer.state_dict()

                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.dataset_name == 'MNIST':
                    images = images.reshape(-1, self.input_size).to(self.device)
                
                # Forward pass
                def closure():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    return loss

                loss = closure()
                self.training_loss.append(float(loss.item()))
                self.val_acc.append(float(self.test()))
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()

                if self.opt_name == 'Kinematics' or self.opt_name == 'KinFwdAdj':
                    # print("Pre: ", closure().item())
                    # torch.save(self.model.state_dict(), "./model_save")
                    self.optimizer.step(closure, self.model)
                    #self.model.load_state_dict(old_model_state_dict)
                    # self.model.load_state_dict(torch.load("./model_save"))
                    # print("Post: ", closure().item())
                elif self.opt_name == 'KinAdj':
                    self.optimizer.step(closure)
                else:
                    self.optimizer.step()

                if (i+1) % self.print_every == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Val: {}' 
                           .format(epoch+1, self.num_epochs, i+1, total_step, loss.item(), self.test(verbose = True))) 
                    
                # print("POST-STEP LOSS: ", loss.item())

    def test(self, verbose = False):
        # Test the model
        # In test phase, we don't need to compute gradients (for memory efficiency)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)

                if self.dataset_name == 'MNIST':
                    images = images.reshape(-1, self.input_size).to(self.device)
                '''
                if self.input_size is None:
                    images = images.to(self.device)
                else:
                    images = images.reshape(-1, self.input_size).to(self.device)
                '''
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            self.test_accuracy = 100 * correct / total
            if verbose:
                print('Accuracy of the network on the 10000 test images: {} %'.format(self.test_accuracy))
        return self.test_accuracy

    def plotTrain(self):
        """Plot the training loss against iterations
        """
        if not self.training_loss:
            self.train()
            # self.test()
        assert(self.training_loss)
        plt.plot(self.training_loss)
        plt.title("Training Loss of {} on {} ({} model)".format(self.opt_name, self.dataset_name, self.model_name))
        plt.ylabel("{}".format(self.loss_name))
        plt.xlabel("Iterations ({} epochs of {} batch size)".format(self.num_epochs, self.batch_size))
        plt.show()

    def plotTest(self):
        """Plot the validation accuracy against iterations
        """
        if not self.val_acc:
            self.train()
            # self.test()
        assert(self.val_acc)
        plt.plot(self.val_acc)
        plt.title("Val Accuracy of {} on {} ({} model)".format(self.opt_name, self.dataset_name, self.model_name))
        plt.ylabel("Accuracy %")
        plt.xlabel("Iterations ({} epochs of {} batch size)".format(self.num_epochs, self.batch_size))
        plt.show()

    def multitrain(self):
        """Trains all supported optimization algorithms independently for a specific number of trials each, then averages
        """
        training_dict = {}
        testing_dict = {}
        time_dict = {}
        # initialize
        for opt in self.supported_optims:
            self.training_dict[opt] = []
            self.testing_dict[opt] = []
            self.time_dict[opt] = []
        for opt in self.supported_optims:
            # check if algo is supported
            assert(opt in self.supported_optims)
            print("multitrain on {}".format(opt))
            # reset and train model
            self.__init__(self.dataset_name, self.model_name, opt)
            start = time.time()
            self.train()
            end = time.time()
            self.test()
            # record train and test results 
            training_dict[opt] = self.training_loss
            testing_dict[opt] = self.val_acc
            time_dict[opt] = end - start
        # record final results 
        self.training_dict = training_dict
        self.testing_dict = testing_dict
        self.time_dict = time_dict

    def plotMultitrain(self, loss_fig = None, val_fig = None):
        """Plots multitrain results
            loss_fig / val_fig for saving file name
        """
        if not self.training_dict:
            self.multitrain()
        assert(self.training_dict)
        assert(self.testing_dict)

        # plot losses
        plt.title("Comparative Loss on {} ({} model)".format(self.dataset_name, self.model_name))
        legend = []
        for opt, train in self.training_dict.items():
            plt.plot(train)
            legend.append(opt)
        plt.legend(legend)
        plt.ylabel("{}".format(self.loss_name))
        plt.xlabel("Iterations (epochs: {}, batch size: {})".format(self.num_epochs, self.batch_size))
        
        if loss_fig is None:
            plt.show()
        else:
            print("Saved loss figure at {}".format(loss_fig))
            plt.savefig(loss_fig)


        # plot val acc
        plt.title("Comparative Val Acc on {} ({} model)".format(self.dataset_name, self.model_name))
        legend = []
        for opt, val in self.testing_dict.items():
            plt.plot(val)
            legend.append(opt)
        plt.legend(legend)
        plt.ylabel("Accuracy %")
        plt.xlabel("Iterations (epochs: {}, batch size: {})".format(self.num_epochs, self.batch_size))

        if val_fig is None:
            plt.show()
        else:
            print("Saved val figure at {}".format(val_fig))
            plt.savefig(val_fig)






if __name__ == "__main__":
    dname, mname, oname = 'CIFAR10', 'resnet18', 'KinFwdAdj'
    # print("START {} {} {}".format(dname, mname, oname))
    # dl = DataLoader(dname, mname, oname)
    # # dl.plotMultitrain()
    loss_fig_name = './figures/{}_{}_{}.png'.format(oname, mname, dname)
    val_fig_name = './figures/Val_{}_{}_{}.png'.format(oname, mname, dname)
    dl = DataLoader(dname, mname, oname)
    # dl.plotMultitrain(loss_fig_name, val_fig_name)
    dl.plotMultitrain()
