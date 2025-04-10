import Custom_Optimizers
import torch
import torchvision
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch.optim as optim

import torch
import torchvision
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms

def build_dataset_CIFAR100(minibatch_size, repeat_times = 1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Mean and std for CIFAR-100
    ])
    # Load the original dataset
    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )

    # Duplicate the dataset by concatenating it with itself
    duplicated_dataset = ConcatDataset([dataset] * repeat_times)

    # Create a DataLoader from the duplicated dataset
    loader = DataLoader(duplicated_dataset, batch_size=minibatch_size, shuffle=True)
    
    return loader

def build_test_dataset_CIFAR100(batch_size_test):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
    ])

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform
        ),
        batch_size=batch_size_test, shuffle=False  # No need to shuffle test data
    )
    return test_loader

def save_data(name, L_sizes, next_L, loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, s_min, s_plus, mb, lrb, lr, 
              max_L, min_L, optimizer, network, max_lr, min_lr):
    max_len = max(len(L_sizes), len(next_L), len(loss_t), len(loss_v), len(acc_t), len(acc_v), len(lr_m), 
                  len(s_min), len(s_plus), len(mb), len(lrb), len(lr_std), len(lr), len(max_L), len(min_L),
                  len(optimizer), len(network), len(max_lr), len(min_lr))
    L_sizes += [''] * (max_len - len(L_sizes)) 
    next_L += [''] * (max_len - len(next_L))
    loss_t += [''] * (max_len - len(loss_t))
    loss_v += [''] * (max_len - len(loss_v))
    acc_t += [''] * (max_len - len(acc_t))
    acc_v += [''] * (max_len - len(acc_v))
    lr_m += [''] * (max_len - len(lr_m))
    lr_std += [''] * (max_len - len(lr_std))
    s_min += [''] * (max_len - len(s_min))
    s_plus += [''] * (max_len - len(s_plus))
    mb += [''] * (max_len - len(mb))
    lrb += [''] * (max_len - len(lrb))
    lr += [''] * (max_len - len(lr))
    max_L += [''] * (max_len - len(max_L))
    min_L += [''] * (max_len - len(min_L))
    optimizer += [''] * (max_len - len(optimizer))
    network += [''] * (max_len - len(network))
    max_lr += [''] * (max_len - len(max_lr))
    min_lr += [''] * (max_len - len(min_lr))

    with open(f"{name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["L_sizes", "next_L", "loss_t", "loss_v", "acc_t", "acc_v", "lr_mean", "lr_std", "s_min", 
                         "s_plus", "mb_size", "lrb_size", "lr", "max_L", "min_L", "optimizer", "network", "max_lr",
                         "min_lr"])  # Column headers
        for i in range(max_len):
            writer.writerow([L_sizes[i], next_L[i], loss_t[i], loss_v[i], acc_t[i], acc_v[i], lr_m[i], lr_std[i], 
                             s_min[i], s_plus[i], mb[i], lrb[i], lr[i], max_L[i], min_L[i], optimizer[i], network[i],
                             max_lr[i], min_lr[i]])    

def validate(network, test_loader, device, return_loss_acc = False, print_stuff = True):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= (len(test_loader.dataset)/len(data))
    if print_stuff == True:
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    if return_loss_acc == True:
        return test_loss, 100 * correct / len(test_loader.dataset)

def int_to_list(x):
    if type(x) == int or type(x) == float:
        x = [x]
        return x
    else:
        return x

class SWEEP():
    def __init__(self, dataset = 'CIFAR100', M = 10, L = 50000, lr = 0.01, epochs = 10, optimizer = 'SGDUPD',
                 save_results = True, repeat_dataset = 1, batch_size_test = 10000, eta_m = 0.7375, eta_p = 1.2, 
                 min_lr = 1e-6, max_lr = 0.1, network = 'CNN', s_plus = 0, s_min = 0, name = 'name', plot = False):
        self.dataset = dataset  #Which dataset to use
        self.repeat_dataset = repeat_dataset #How many times to repeat the data set
        self.batch_size_test = batch_size_test
        self.save_results = save_results #Save results True or False

        #possible parameters to sweep:
        self.M = int_to_list(M) #Mini-batch M
        if optimizer != 'SGD' and optimizer != 'ADAM':
            self.L = int_to_list(L) #Mini-batch L
        else:
            self.L = [0]
        self.lr = int_to_list(lr) #Learning rate

        self.s_plus = s_plus #Still needs to be added
        self.s_min = s_min #Still needs to be added

        self.epochs = epochs #Number of epochs to run
        self.optimizer = optimizer
        self.eta_m, self.eta_p = eta_m, eta_p
        self.min_lr, self.max_lr = min_lr, max_lr
        self.network = network
        self.max_L = 100000000000000
        self.min_L = int(max(self.M) * 2)
        
        self.L_update_type = 'lr based'
        self.L_update_step = 'round nearest integer'

        self.name = name
        self.plot = plot
        self.n_sweep = 0
        self.n_run = 0
        self.sweep_par = self.sweep_parameters()
        

    def sweep(self):
        for i in range(self.n_sweeps):
            self.n_run = i+1
            [M, L, lr] = self.sweep_par[i,:] 
            device = 'cuda' if torch.cuda.is_available() else 'cpu' #Might be put in Sweep
            network = self.network_type(device)
            test_loader = self.build_test_dataset()
            loader = self.build_dataset(int(M))
            print('Run =', i+1)
            print('M =', M)
            print('L =', L)
            print('lr =', lr)
            print('Optimizer =', self.optimizer)
            self.run(int(M), int(L), lr, network, device, loader, test_loader)


    def run(self, M, L, lr, network, device, loader, test_loader): #Add optimizer as sweep parameter
        optimizer = self.optimizer_type(network, M, L, lr)
        loss_val, acc_val = [], []
        loss_train, acc_train = [], []
        ps = 10000 #print step
        for epoch in range(1, self.epochs+1):
            network.train()
            loss_3000 = 0
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = network(data)
                loss = F.nll_loss(output, target)
                loss_3000 += loss
                loss.backward()
                optimizer.step()
                if ((batch_idx+1)*len(data))%ps==0:
                    if M<ps:
                        loss_3000 /= (ps/len(data))
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                    len(loader.dataset), 100. * batch_idx / len(loader), loss_3000.item()))
                    loss_3000 = 0
            network.eval()
            loss_v, acc_v = validate(network, test_loader, device, True)
            loss_val.append(loss_v)
            acc_val.append(float(acc_v))
            loss_t, acc_t = validate(network, loader, device, True)
            loss_train.append(loss_t)
            acc_train.append(float(acc_t))
        if self.optimizer != 'SGD' and self.optimizer != 'ADAM':
            lr_mean = optimizer.lr_mean
            lr_std = optimizer.lr_std
        else: 
            lr_mean, lr_std = 0, 0

        self.save_data(self.name+str(self.n_run), loss_train, loss_val, acc_train, acc_val, lr_mean, lr_std, self.s_min, self.s_plus, M, L, lr, optimizer)
        if self.plot == True:
            plt.plot(acc_val, label = 'Val')
            plt.plot(acc_train, label = 'Train')
            plt.legend()
            plt.show()

    def sweep_parameters(self):
        n_sweeps_lr = 1
        n_sweeps_L = len(self.lr)
        n_sweeps_M = len(self.L) * n_sweeps_L
        n_sweeps = len(self.M) * n_sweeps_M #number of sweeps
        self.n_sweeps = n_sweeps
        par = np.zeros((n_sweeps, 3))
        for i in range(len(self.M)):
            par[i*n_sweeps_M:(i+1)*n_sweeps_M, 0] = np.ones(n_sweeps_M)*self.M[i]
        for k in range(len(self.M)): 
            l = len(self.L)*len(self.lr)
            for i in range(len(self.L)):
                par[(i)*n_sweeps_L+k*l:(i+1)*n_sweeps_L+k*l, 1] = np.ones(n_sweeps_L)*self.L[i]
        for k in range(len(self.M)*len(self.L)): #Number of sections
            l = len(self.lr) #Length of a section
            for i in range(len(self.lr)):
                par[i*n_sweeps_lr+k*l:(i+1)*n_sweeps_lr+k*l, 2] = np.ones(n_sweeps_lr)*self.lr[i]
        return par

    def network_type(self, device):
        random_seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(random_seed)
        if self.network == 'CNN':
            network = Custom_Optimizers.CNN(num_channels = 3, num_classes = 100)
        elif self.network == 'ResNet9':
            network = Custom_Optimizers.ResNet9(in_channels=3, num_classes=100, drop_rate=0) # Default Dropout is 0.2
        else:
            print('ERROR: Wrong network type. Try CNN, ResNet9. You gave ', self.network)
        network.to(device)
        return network

    def optimizer_type(self, network, M, L, lr): #Need to add optimizers as parameter
        if self.optimizer == 'SRPROP':
            return Custom_Optimizers.SRPROP(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True)
        elif self.optimizer == 'SRPROPL':
            return Custom_Optimizers.SRPROPL(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True, L_update_type = self.L_update_type, L_update_step = self.L_update_step, s_plus = self.s_plus, s_min = self.s_min, max_L = self.max_L, min_L = self.min_L)
        elif self.optimizer == 'ADAMUPD':
            return Custom_Optimizers.ADAMUPD(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True)
        elif self.optimizer == 'ADAMUPDL':
            return Custom_Optimizers.ADAMUPDL(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True, L_update_type = self.L_update_type, L_update_step = self.L_update_step, s_plus = self.s_plus, s_min = self.s_min, max_L = self.max_L, min_L = self.min_L)
        elif self.optimizer == 'SGDUPD':
            return Custom_Optimizers.SGDUPD(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True)
        elif self.optimizer == 'SGDUPDL' or self.optimizer == 'SGDUpdL':
            return Custom_Optimizers.SGDUPDL(network.parameters(), M=M, L=L, lr=lr, etas=(self.eta_m, self.eta_p), lr_limits=(self.min_lr, self.max_lr), track_lr=True, L_update_type = self.L_update_type, L_update_step = self.L_update_step, s_plus = self.s_plus, s_min = self.s_min, max_L = self.max_L, min_L = self.min_L)
        elif self.optimizer == 'SGD':
            return optim.SGD(network.parameters(), lr=lr, momentum=0, weight_decay=0, dampening=0)
        elif self.optimizer == 'ADAM':
            return optim.Adam(network.parameters(), lr=lr)
        else:
            print('ERROR: Wrong optimizer type. Try: SRPROP, SRPROPL, ADAMUPD, ADAMUPDL, ADAM, SGDUPD, SGDUPDL, SGD. You gave ', self.optimizer)

    def save_data(self, name, loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, s_min, s_plus, mb, lrb, lr, optimizer):
        if self.save_results == True:
            if self.optimizer == 'SGDUPD' or self.optimizer == 'ADAMUPD' or self.optimizer == 'SRPROP':
                save_data(name, ['empty'], ['empty'], loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, ['empty'], ['empty'], [mb], [lrb], [lr], 
                            ['empty'], ['empty'], [self.optimizer], [self.network], [self.max_lr], [self.min_lr])
            elif self.optimizer == 'SGDUPDL' or self.optimizer == 'ADAMUPDL' or self.optimizer == 'SRPROPL':
                save_data(name, optimizer.L_sizes, optimizer.next_L, loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, [s_min], [s_plus], [mb], [lrb], [lr], 
                            [self.max_L], [self.min_L], [self.optimizer], [self.network], [self.max_lr], [self.min_lr])
            elif self.optimizer == 'SGD' or self.optimizer == 'ADAM':
                save_data(name, ['empty'], ['empty'], loss_t, loss_v, acc_t, acc_v, ['empty'], ['empty'], ['empty'], ['empty'], [mb], ['empty'], [lr], 
                            ['empty'], ['empty'], [self.optimizer], [self.network], ['empty'], ['empty'])


    def build_dataset(self, M):
        if self.dataset == 'CIFAR100':
            return build_dataset_CIFAR100(M, self.repeat_dataset)
        else:
            print('ERROR: Invalid dataset, try CIFAR100')

    def build_test_dataset(self):
        if self.dataset == 'CIFAR100':
            return build_test_dataset_CIFAR100(self.batch_size_test)
        else:
            print('ERROR: Invalid dataset, try CIFAR100')
    
    