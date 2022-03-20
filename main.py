import torch
import torch.nn.functional as F
import numpy as np
import random
import os
from VRF import VRF
from utils import MinMaxScale
from utils import make_Tensor
from datetime import date

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Determine to use CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# VRF parameters
featureDim = 15
InterDim = 10
Numcol = 14

# Initialize Hyperparameters
learning_rate = 1e-4
num_epochs = 10000
 
dir_prefix = "data/"

X_A = np.load(dir_prefix + "X_A" + ".npy")
y_A = np.load(dir_prefix + "Y_A" + ".npy")
_, _, _, _, X_A, y_A = MinMaxScale(X_A, y_A)
X_B = np.load(dir_prefix + "X_B" + ".npy")
y_B = np.load(dir_prefix + "Y_B" + ".npy")
_, _, _, _, X_B, y_B = MinMaxScale(X_B, y_B)
X_C = np.load(dir_prefix + "X_C" + ".npy")
y_C = np.load(dir_prefix + "Y_C" + ".npy")
_, _, _, _, X_C, y_C = MinMaxScale(X_C, y_C)
X_D = np.load(dir_prefix + "X_D" + ".npy")
y_D = np.load(dir_prefix + "Y_D" + ".npy")
_, _, _, _, X_D, y_D = MinMaxScale(X_D, y_D)
X_E = np.load(dir_prefix + "X_E" + ".npy")
y_E = np.load(dir_prefix + "Y_E" + ".npy")
_, _, _, _, X_E, y_E = MinMaxScale(X_E, y_E)
X_F = np.load(dir_prefix + "X_F" + ".npy")
y_F = np.load(dir_prefix + "Y_F" + ".npy")
_, _, _, _, X_F, y_F = MinMaxScale(X_F, y_F)
X_G = np.load(dir_prefix + "X_G" + ".npy")
y_G = np.load(dir_prefix + "Y_G" + ".npy")
_, _, _, _, X_G, y_G = MinMaxScale(X_G, y_G)

X = np.concatenate((X_A, X_B, X_C, X_D, X_E, X_F, X_G), axis = 1)
y = np.concatenate((y_A, y_B, y_C, y_D, y_E, y_F, y_G), axis = 1)

# Train Data 
X_train = X[:int(0.8*len(X))] 
X_test = X[int(0.8*len(X)):]

# Test Data 
y_train = y[:int(0.8*len(X))] 
y_test = y[int(0.8*len(X)):] 

X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)

# Initialize the network and the Adam optimizer
net = VRF(Numcol=Numcol, InterDim=InterDim, featureDim=featureDim).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training the network for a given number of epochs
# The loss after every 100 epochs is printed
for epoch in range(num_epochs):
    for idx, data in enumerate(X_train, 0):
        x = data
        x = x.view(X.shape[1], -1)
        x = x.to(device)
        y = y_train[idx][:1,:]
        y = y.view(y.shape[1], -1)
        y = y.to(device)
        
        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar, mu_y, var_log_y, pz_mean = net(x)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        label_loss = 0.5*torch.sum(var_log_y + (1/var_log_y.exp())*((y - mu_y).pow(2)))
        kl_divergence = 0.5*torch.sum(1 + logVar - (mu-pz_mean).pow(2) - logVar.exp())
        loss = torch.mean(F.mse_loss(out, x) + label_loss - kl_divergence)
        
        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 0:
        print(label_loss)
        print('Epoch {}: Loss {}'.format(epoch, loss))
    if epoch % 1000 == 999:
        torch.save({
            'model' : net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'loss' : loss
        }, "all_data" + str(epoch+1) + ".tar")
