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

def build_dataset(minibatch_size, repeat_times = 1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # Mean and std for CIFAR-100
    ])
    """
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        ),
        batch_size=minibatch_size, shuffle=True
    )
    """
    # Load the original dataset
    dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform
    )

    # Duplicate the dataset by concatenating it with itself
    duplicated_dataset = ConcatDataset([dataset] * repeat_times)

    # Create a DataLoader from the duplicated dataset
    loader = DataLoader(duplicated_dataset, batch_size=minibatch_size, shuffle=True)
    
    return loader

def build_test_dataset(batch_size_test):
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


def plot_results(lr_mean, loss_val, loss_train, acc_val, acc_train, L_sizes, next_L):
    fig, ax = plt.subplots(2, 3, figsize = [12,5])

    ax[0,0].plot(lr_mean)
    ax[0,0].set_ylabel('lr mean')
    ax[0,0].set_xlabel('lr updates')
    ax[1,0].plot(L_sizes)
    ax[1,0].set_ylabel('L')
    ax[1,0].set_xlabel('lr updates')
    ax[0,1].plot(np.array(next_L[0:len(lr_mean)])/50000, lr_mean)
    ax[1,1].plot(np.array(next_L[0:len(L_sizes)])/50000, L_sizes)
    ax[0,1].set_ylabel('lr mean')
    ax[0,1].set_xlabel('epoch')
    ax[1,1].set_ylabel('L')
    ax[1,1].set_xlabel('epoch')
    ax[0,2].plot(np.linspace(1,len(loss_val),len(loss_val)), loss_val, label = 'Val')
    if len(loss_train) != 0:
        ax[0,2].plot(np.linspace(1,len(loss_train),len(loss_train)),loss_train, label = 'Train')
    ax[0,2].set_xlabel('epoch')
    ax[0,2].set_ylabel('Loss')
    ax[0,2].legend()
    ax[1,2].plot(np.linspace(1,len(acc_val),len(acc_val)), acc_val, label = 'Val')
    if len(acc_train) != 0:
        ax[1,2].plot(np.linspace(1,len(acc_train),len(acc_train)), acc_train, label = 'Train')
    ax[1,2].set_xlabel('epoch')
    ax[1,2].set_ylabel('Accuracy')
    ax[1,2].legend()
    plt.tight_layout()

def save_data(name, L_sizes, next_L, loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, s_min, s_plus, mb, lrb, lr, 
              max_L, min_L):
    max_len = max(len(L_sizes), len(next_L), len(loss_t), len(loss_v), len(acc_t), len(acc_v), len(lr_m), 
                  len(s_min), len(s_plus), len(mb), len(lrb), len(lr_std), len(lr), len(max_L), len(min_L))
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

    with open(f"{name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["L_sizes", "next_L", "loss_t", "loss_v", "acc_t", "acc_v", "lr_mean", "lr_std", "s_min", 
                         "s_plus", "mb_size", "lrb_size", "lr", "max_L", "min_L"])  # Column headers
        for i in range(max_len):
            writer.writerow([L_sizes[i], next_L[i], loss_t[i], loss_v[i], acc_t[i], acc_v[i], lr_m[i], lr_std[i], 
                             s_min[i], s_plus[i], mb[i], lrb[i], lr[i], max_L[i], min_L[i]])    

def save_data_no_L_update(name, loss_t, loss_v, acc_t, acc_v, lr_m, lr_std, mb, lrb, lr):
    max_len = max(len(loss_t), len(loss_v), len(acc_t), len(acc_v), len(lr_m), len(mb), len(lrb), len(lr_std), len(lr))
    loss_t += [''] * (max_len - len(loss_t))
    loss_v += [''] * (max_len - len(loss_v))
    acc_t += [''] * (max_len - len(acc_t))
    acc_v += [''] * (max_len - len(acc_v))
    lr_m += [''] * (max_len - len(lr_m))
    lr_std += [''] * (max_len - len(lr_std))
    mb += [''] * (max_len - len(mb))
    lrb += [''] * (max_len - len(lrb))
    lr += [''] * (max_len - len(lr))

    with open(f"{name}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["loss_t", "loss_v", "acc_t", "acc_v", "lr_mean", "lr_std", 
                         "mb_size", "lrb_size", "lr"])  # Column headers
        for i in range(max_len):
            writer.writerow([loss_t[i], loss_v[i], acc_t[i], acc_v[i], lr_m[i], lr_std[i], 
                              mb[i], lrb[i], lr[i]])                                                 

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

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    random_seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(random_seed)
    #print(random_seed)
    #seed = 1234
    #torch.manual_seed(seed)
    

    eta_m, eta_p = 0.7375, 1.2   
    min_lr, max_lr = 1e-6, 0.1

    mb_size = 10
    lrb_size= 50000*51
    learning_rate=0.01
    test_batch_size = 10000
    epochs = 50
    s_plus = 0
    s_min = 0
    max_L = 50000*52
    #min_L = mb_size*10
    min_L = 0

    L_update_type = 'lr based'
    #L_update_step = 'round up'
    L_update_step = 'round nearest integer'
    #L_update_step = '1 step'
 

    loader = build_dataset(mb_size)
    #print(type(loader))
    test_loader = build_test_dataset(test_batch_size)

    network = Custom_Optimizers.CNN(num_channels = 3, num_classes = 100)
    #network = Custom_Optimizers.ResNet9(in_channels=3, num_classes=100, drop_rate=0) # Default Dropout is 0.2
    network.to(device)
    
    #optimizer = optim.Adam()
    #optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True)
    #optimizer = Custom_Optimizers.SRPROPL(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True, L_update_type = L_update_type, L_update_step = L_update_step, s_plus = s_plus, s_min = s_min, max_L = max_L, min_L = min_L)
    #optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True)
    #optimizer = Custom_Optimizers.ADAMUPDL(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True, L_update_type = L_update_type, L_update_step = L_update_step, s_plus = s_plus, s_min = s_min, max_L = max_L, min_L = min_L)
    #optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True)
    optimizer = Custom_Optimizers.SGDUPDL(network.parameters(), M=mb_size, L=lrb_size, lr=learning_rate, etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), track_lr=True, L_update_type = L_update_type, L_update_step = L_update_step, s_plus = s_plus, s_min = s_min, max_L = max_L, min_L = min_L)
    loss_val, acc_val = [], []
    loss_train, acc_train = [], []
    ps = 10000 #print step
    for epoch in range(1, epochs+1):
        network.train()
        loss_3000 = 0
        for batch_idx, (data, target) in enumerate(loader):
            #print(enumerate(loader))
            #print(data.size())
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss_3000 += loss
            loss.backward()
            optimizer.step()
            if ((batch_idx+1)*len(data))%ps==0:
                if mb_size<ps:
                    loss_3000 /= (ps/len(data))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                len(loader.dataset), 100. * batch_idx / len(loader), loss_3000.item()))
                loss_3000 = 0
            """
            print(f"Final Learning Rate Mean: {optimizer.lr_mean}")
            print(f"Final Learning Rate Std: {optimizer.lr_std}")
            print(f"Number of LR Updates: {optimizer.lr_counter}")
            """
        
        network.eval()
        loss_v, acc_v = validate(network, test_loader, device, True)
        loss_val.append(loss_v)
        acc_val.append(float(acc_v))
        loss_t, acc_t = validate(network, loader, device, True)
        loss_train.append(loss_t)
        acc_train.append(float(acc_t))

  
    lr_mean = optimizer.lr_mean
    lr_std = optimizer.lr_std
    lr_steps = optimizer.lr_counter

    
    #"""
    #print(f"Final Learning Rate Mean: {lr_mean}")
    #print(f"Final Learning Rate Std: {lr_std}")
    #print(f"Number of LR Updates: {lr_steps}")
    #"""
    """
    print('L = ', optimizer.L_sizes)
    print('next_L = ', optimizer.next_L)
    print('loss_t = ', loss_train)
    print('loss_v = ', loss_val)
    print('acc_t = ', acc_train)
    print('acc_v = ', acc_val)
    print('lr_mean = ', lr_mean)
    """
    #print(optimizer.next_L)
    
    plot_results(lr_mean, loss_val, loss_train, acc_val, acc_train, optimizer.L_sizes, optimizer.next_L)
    plt.show()
    #save_data('ADAM_L_1_0', optimizer.L_sizes, optimizer.next_L, loss_train, loss_val, acc_train, acc_val, lr_mean, lr_std,
     #        [s_min], [s_plus], [mb_size], [lrb_size], [learning_rate], [max_L], [min_L])
    save_data_no_L_update('SGD_L_1_0', loss_train, loss_val, acc_train, acc_val, lr_mean, 
              lr_std,[mb_size], [lrb_size], [learning_rate])

if __name__ == '__main__':
    main()