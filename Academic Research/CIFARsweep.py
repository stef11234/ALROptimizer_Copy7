import sys
#sys.path.append("..")
#sys.path.append(r"/home/avesta/m25_schijfsi/Thesis/ALROptimizer_Copy5")
sys.path.append(r"C:\Thesis\ALROptimizer_Copy5")
import Custom_Optimizers
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import wandb
import csv

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
           
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb.login()

random_seed = random.randint(0, 2**32 - 1)
torch.manual_seed(random_seed)

eta_m, eta_p = 0.7375, 1.2
min_lr, max_lr = 1e-6, 0.1
max_L = 1000000000000

# Data augmentation transform
mean = (0.5071, 0.4866, 0.4409)  # (129.3, 124.1, 112.4) 
std = (0.2673, 0.2564, 0.2762)  # (68.2, 65.4, 70.4)

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert images to tensors
    torchvision.transforms.Normalize(mean, std),  # Normalize all pixel values
    torchvision.transforms.RandomHorizontalFlip(p=0.5),    # Horizontally flip augmented data with 50% chance of flipping
    torchvision.transforms.Pad(padding=4, padding_mode='reflect'),    # Pad with 4 pixels using reflection
    torchvision.transforms.RandomCrop(size=(32, 32))])    # Randomly centered crop back to original size

def build_dataset(minibatch_size):
    loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data',train=True, download=False,
                                        transform=transform), batch_size=minibatch_size, shuffle=True)
    return loader

def build_network():
    #network = Custom_Optimizers.ResNet9(in_channels=3, num_classes=100, drop_rate=0) # Default Dropout is 0.2
    network = Custom_Optimizers.CNN(num_channels=3, num_classes=100)
    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)  # Parallelise data with two GPUs
        print("Using", torch.cuda.device_count(), "GPUs.")
    
    return network.to(device)

def build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr):
    if config["optimizer"] == "S-Rprop":
        optimizer = Custom_Optimizers.SRPROP(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                              weight_decay=0, track_lr=False)
    elif config["optimizer"] == "SGD-Upd":
        optimizer = Custom_Optimizers.SGDUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                             lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                             weight_decay=0, track_lr=False)
    elif config["optimizer"] == "Adam-Upd":
        optimizer = Custom_Optimizers.ADAMUPD(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=0, track_lr=False)
    elif config["optimizer"] == "Adam-UpdL":
        optimizer = Custom_Optimizers.ADAMUPDL(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], s_plus = config["s_plus"], s_min = config["s_min"],\
                                                L_update_type = config["L_update_type"], L_update_step = config["L_update_step"],\
                                                min_L = config["min_L"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=0, track_lr=True, max_L = max_L)
    elif config["optimizer"] == "SRpropL":
        optimizer = Custom_Optimizers.SRPROPL(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], s_plus = config["s_plus"], s_min = config["s_min"],\
                                                L_update_type = config["L_update_type"], L_update_step = config["L_update_step"],\
                                                min_L = config["min_L"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=0, track_lr=True, max_L = max_L)
    elif config["optimizer"] == "SGD-UpdL":
        optimizer = Custom_Optimizers.SGDUPDL(network.parameters(), M=config["minibatch_size"], L=config["lr_batch_size"], \
                                              lr=config["learning_rate"], s_plus = config["s_plus"], s_min = config["s_min"],\
                                                L_update_type = config["L_update_type"], L_update_step = config["L_update_step"],\
                                                min_L = config["min_L"], etas=(eta_m, eta_p), lr_limits=(min_lr, max_lr), \
                                                weight_decay=0, track_lr=True, max_L = max_L)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "SGD+M":
        optimizer = optim.SGD(network.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=0, dampening=0)
    else:
        print("Optimizer type not recognised")
    return optimizer

def train_epoch(network, loader, optimizer, epoch, min_loss, max_acc):
    network.train()
    training_loss=0
    correct=0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss.backward()
        training_loss += loss.item()
        optimizer.step()
        if (batch_idx+1)%1000==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx+1) * len(data),
                  len(loader.dataset), 100. * batch_idx / len(loader), loss.item()))
    training_loss /= (len(loader.dataset)/len(data))
    training_acc = 100. * correct / len(loader.dataset)
    if training_loss < min_loss:
                min_loss = training_loss
    if training_acc > max_acc:
        max_acc = training_acc
    wandb.log({"Training loss": training_loss, "Training accuracy": 100. * correct / len(loader.dataset)})
    return min_loss, max_acc, training_loss, training_acc

run = 0  # Initialize globally

def train(config=None):
    global run  # Declare `run` as global before using it

    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        loss_t = []
        acc_t = []
        config = wandb.config
    
        network = build_network()

        optimizer = build_optimizer(network, config, eta_m, eta_p, min_lr, max_lr)

        min_loss = float('inf')
        max_acc = -float('inf')

        for epoch in range(1, config["epochs"]+1):
            loader = build_dataset(config["minibatch_size"])
            min_loss, max_acc, train_loss, train_acc = train_epoch(network, loader, optimizer, epoch, min_loss, max_acc)
            #print(train_acc, type(train_acc))
            loss_t.append(float(train_loss))
            acc_t.append(float(train_acc))

        wandb.log({"Training loss": min_loss, "Training accuracy": max_acc})
    run += 1
    #save_data(f'SGD_av_L_run{run}', optimizer.L_sizes, optimizer.next_L, loss_t, ["empty"], acc_t, ["empty"], optimizer.lr_mean, 
    #          optimizer.lr_std, [config.s_min], [config.s_plus], [optimizer.M], [optimizer.L], [config.learning_rate], 
    #          [max_L], [config.min_L])
    save_data_no_L_update(f'SGD_high_L_3_{run}', loss_t, ["empty"], acc_t, ["empty"], optimizer.lr_mean, 
              optimizer.lr_std,[optimizer.M], [optimizer.L], [config.learning_rate])
    #print(optimizer.next_L)

    #save_data('hey', )

###### -----------------------------------------------------------------------
######                   For hyper-parameter sweeps using training only

sweep_config = {
    "name": "SGD find high L 3",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Training accuracy"},
    "parameters": {
        "optimizer": {"value": "SGD-Upd"}, 
        "learning_rate": {"values": [1e-2]},
        "minibatch_size": {"value": 10},
        "lr_batch_size": {"values": [50000*51, 10000, 20000,30000,50000,50000*2, 50000*4,50000*10, 50000*25]},                          # Change to = 0 if algorithm does not use lr_batch_size
        #"s_plus": {"values": [1, 10,100]},
        #"s_min": {"values": [0, 's_plus/10','s_plus/2', 's_plus']},
        #"min_L": {"values": [0]},
        "epochs": {"value": 50},
        #"L_update_type": {"value": "lr based"},
        #"L_update_step": {"value": "round nearest integer"},
    },
}

sweep_id = wandb.sweep(sweep=sweep_config, project="CIFAR-100")

wandb.agent(sweep_id, train)