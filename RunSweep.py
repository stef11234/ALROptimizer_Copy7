from Sweep import SWEEP

M = [5,10,20,50,100,200,500]
L = [5000, 10000, 25000, 50000]
lr = [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.1]
epochs = 10
network = 'CNN'
#network = 'ResNet9'
save_results = True
optimizer = 'SGD' 
name = 'SGD_2_'

s = SWEEP(M = M, L = L, lr = lr, epochs = epochs, network = network, save_results = save_results, optimizer = optimizer, name = name)

p = s.sweep_parameters()
print(p)
s.sweep()
