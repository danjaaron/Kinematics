import torch, torchvision, time, inspect, kinfwd_optim
from torchvision import transforms
"""
Trains and tests a specific model / optim / dataset combination
8/31/19 successor to all past Kinematics github master / data_loader files
"""

class Trainer:
    def __init__(self, model, loss_criterion, optimizer, train_loader, test_loader, num_epochs = 1):
        # global settings 
        self.num_epochs = num_epochs
        # device configuration
        self.device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # init instance objects
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        # self.loss_criterion = loss_criterion
        self.loss_criterion = loss_criterion
        # get optimizer step parameters
        self.optimizer_parameters = list(dict(inspect.signature(self.optimizer.step).parameters).keys())
        # init lists to store results 
        self.train_results, self.test_results = [], []
    
    def train(self):
        """ Trains and tests model according to epochs, storing test loss and training accuracy at each step
        """
        if 'MNIST' in str(self.train_loader.dataset):
            mnist_bool = True
        # set printing settings
        total_step = len(self.train_loader)
        self.print_every = 1 #int(round(total_step / 10.))
        # train
        for epoch in range(self.num_epochs):
            for i, (data, target) in enumerate(self.train_loader):  

                # move all to device
                data = data.to(self.device)
                if mnist_bool:
                    data = data.reshape(-1, 784).to(self.device)
                target = target.to(self.device)

                def closure():
                    outputs = self.model(data)

                    loss = self.loss_criterion(outputs, target)
                    return loss

                # forward pass
                loss = closure()
                train_res, test_res = float(loss.item()), float(self.test())
                self.train_results.append(train_res)
                self.test_results.append(test_res)
                
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # optimize with appropriate arguments
                print(self.optimizer_parameters)
                if self.optimizer_parameters == ['closure', 'model']:
                    self.optimizer.step(closure, self.model)
                elif self.optimizer_parameters == ['closure']:
                    self.optimizer.step(closure)
                else:
                    self.optimizer.step()

                # print results
                if (i+1) % self.print_every == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Val: {}' 
                           .format(epoch+1, self.num_epochs, i+1, total_step, train_res, test_res))
        return self.train_results, self.test_results

    def test(self):
        """ Returns test accuracy as a percentage over test loader
        """
        if 'MNIST' in str(self.train_loader.dataset):
            mnist_bool = True
        with torch.no_grad():
            correct = 0
            total = 0
            for data, labels in self.test_loader:
                # move all to device
                data = data.to(self.device)
                if mnist_bool:
                    data = data.reshape(-1, 784).to(self.device)
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
    
    """ Example of trainer usage on MNIST 
    """

    batch_size = 10
    shuffle_train = True

    train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                               train=True, 
                                               transform=transforms.Compose([
                                                transforms.ToTensor()
                                                ]),  
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                              train=False, 
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle_train)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False)

    # model and loss
    model = torch.nn.Linear(784, 10)
    criterion = torch.nn.CrossEntropyLoss()

    # Adam optimizer
    opt = torch.optim.Adam(model.parameters())

    # Kinematics optimizer
    # opt = kinfwd_optim.KinFwd(model.parameters())

    # init trainer 
    testTrainer = Trainer(model, criterion, opt, train_loader, test_loader)

    # get results 
    train_results, test_results = testTrainer.train()