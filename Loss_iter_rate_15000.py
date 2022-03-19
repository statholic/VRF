#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.manifold import MDS
from datetime import date
import os


# In[2]:


"""
Make pyplot Ok
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
A Variational Autoencoder
"""
featureDim = 15
InterDim = 7
Numcol = 14

class VAE(nn.Module):
    def __init__(self, inChannels=1, OutChannels=1, InterDim=7, featureDim=15, zDim=1):
        super(VAE, self).__init__()
        
        self.pzFC0 = weight_norm(nn.Linear(zDim, InterDim*Numcol), name = 'weight')

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encFC0 = nn.Linear(featureDim, featureDim)
        self.encFC1 = nn.Linear(featureDim, InterDim)
        self.encFC2 = nn.Linear(InterDim, InterDim)
        self.encFC3 = nn.Linear(InterDim, InterDim)
        self.encFC4 = nn.Linear(InterDim, InterDim)
        self.encFC5 = nn.Linear(InterDim, zDim)
        self.encFC5_1 = nn.Linear(Numcol, zDim)
        self.encFC6 = nn.Linear(InterDim, zDim)
        self.encFC6_1 = nn.Linear(Numcol, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC0 = nn.Linear(InterDim, InterDim)
        self.decFC1 = nn.Linear(InterDim, featureDim)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = x.view(-1, featureDim)
        x = torch.tanh(self.encFC0(x)).to(device)
        x = torch.tanh(self.encFC1(x)).to(device)
        mu = self.encFC2(x).to(device)
        logVar = self.encFC3(x).to(device)
        mu_y = self.encFC5(x).to(device)
        mu_y = mu_y.view(1, 1, -1).to(device)
        mu_y = self.encFC5_1(mu_y).to(device)
        var_log_y = self.encFC6(x).to(device)
        var_log_y = var_log_y.view(1, 1, -1).to(device)
        var_log_y = self.encFC6_1(var_log_y).to(device)
        
        return mu, logVar, mu_y, var_log_y

    def reparameterize1(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std1 = torch.exp(logVar/2).to(device)
        eps1 = torch.randn_like(std1).to(device)
        return mu + std1 * eps1
    
    def reparameterize2(self, mu_y, var_log_y):
        
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std2 = torch.exp(var_log_y/2).to(device)
        eps2 = torch.randn_like(std2).to(device)
        return mu_y + std2 * eps2
        

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decFC0(z).to(device)
        x = self.decFC1(x).to(device)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar, mu_y, var_log_y = self.encoder(x)
        z = self.reparameterize1(mu, logVar)
        y_f = self.reparameterize2(mu_y, var_log_y)
        pz_mean = self.pzFC0(y_f).to(device)
        pz_mean = pz_mean.view(-1, InterDim).to(device)
        out = self.decoder(z)
        mu_y = mu_y.view(-1, 1).to(device)
        var_log_y = var_log_y.view(-1, 1).to(device)
        return out, mu, logVar, mu_y, var_log_y, pz_mean
    
"""
Initialize Hyperparameters
"""
learning_rate = 1e-4
num_epochs = 100


# In[3]:


"""
Standardize other exchange rate same as KR-Dollar exchange rate
""" 
seq_length = 15

def MinMaxScale(X, y):
    MIN0 = np.min(y[:, 0])
    MIN1 = np.min(y[:, 1])
    MAX0 = np.max(y[:, 0])
    MAX1 = np.max(y[:, 1])
    
    X[:, 0] = (X[:, 0] - MIN0)/(MAX0 - MIN0)
    y[:, 0] = (y[:, 0] - MIN0)/(MAX0 - MIN0)
    X[:, 1] = (X[:, 1] - MIN1)/(MAX1 - MIN1)
    y[:, 1] = (y[:, 1] - MIN1)/(MAX1 - MIN1)
    
    return MIN0, MAX0, MIN1, MAX1, X, y
    
X_Korea = np.load("X_10y_Korea_"+str(seq_length)+".npy")
y_Korea = np.load("Y_10y_Korea_"+str(seq_length)+".npy")
_, _, _, _, X_Korea, y_Korea = MinMaxScale(X_Korea, y_Korea)
X_USA = np.load("X_10y_USA_"+str(seq_length)+".npy")
y_USA = np.load("Y_10y_USA_"+str(seq_length)+".npy")
_, _, _, _, X_USA, y_USA = MinMaxScale(X_USA, y_USA)
X_Japan = np.load("X_10y_Japan_"+str(seq_length)+".npy")
y_Japan = np.load("Y_10y_Japan_"+str(seq_length)+".npy")
_, _, _, _, X_Japan, y_Japan = MinMaxScale(X_Japan, y_Japan)
X_Dutch = np.load("X_10y_Dutch_"+str(seq_length)+".npy")
y_Dutch = np.load("Y_10y_Dutch_"+str(seq_length)+".npy")
_, _, _, _, X_Dutch, y_Dutch = MinMaxScale(X_Dutch, y_Dutch)
X_England = np.load("X_10y_England_"+str(seq_length)+".npy")
y_England = np.load("Y_10y_England_"+str(seq_length)+".npy")
_, _, _, _, X_England, y_England = MinMaxScale(X_England, y_England)
X_France = np.load("X_10y_France_"+str(seq_length)+".npy")
y_France = np.load("Y_10y_France_"+str(seq_length)+".npy")
_, _, _, _, X_France, y_France = MinMaxScale(X_France, y_France)
X_China = np.load("X_10y_China_"+str(seq_length)+".npy")
y_China = np.load("Y_10y_China_"+str(seq_length)+".npy")
_, _, _, _, X_China, y_China = MinMaxScale(X_China, y_China)


# In[4]:


X = np.concatenate((X_Korea, X_USA, X_Japan, X_Dutch, X_England, X_France, X_China), axis = 1)
y = np.concatenate((y_Korea, y_USA, y_Japan, y_Dutch, y_England, y_France, y_China), axis = 1)


# In[5]:


"""
Create dataloaders to feed data into the neural network
Default MNIST dataset is used and standard train/test split is performed
"""    
# Train Data 
X_train = X[:int(0.8*len(X))] 
X_test = X[int(0.8*len(X)):]

# Test Data 
y_train = y[:int(0.8*len(X))] 
y_test = y[int(0.8*len(X)):] 

dt = np.load("date_rate.npy",allow_pickle=True)
date_train = np.array([date.fromisoformat(x) for x in dt[:int(0.8*len(X))]])
date_test = np.array([date.fromisoformat(x) for x in dt[int(0.8*len(X)):]])

# Tensor 형태로 변환
def make_Tensor(array):
    return torch.from_numpy(array).float()

X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)


# In[6]:


cwd = "./10000_after"
flist = os.listdir(cwd)
flist = [int(x.split('_')[1].split('e')[-1]) for x in flist]
flist.sort()
flist = [str(x) for x in flist]


# In[7]:


loss = {}
error = {}
for i in range(len(flist)):
    print(i)
    fnum = flist[i]
    loss[fnum] = 0
    """
    Initialize the network and the Adam optimizer
    """
    checkpoint = torch.load(os.path.join(cwd,'all_rate'+fnum+'_7.tar'), map_location=device)
    net = VAE().to(device)
    net.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    net.eval()

    """
    Training the network for a given number of epochs
    The loss after every epoch is printed
    """
    for idx, data in enumerate(X_test, 0):
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
        loss[fnum] += torch.mean(F.mse_loss(out, x) + label_loss - kl_divergence)
    loss[fnum] = (loss[fnum]/len(X_test)).detach().numpy()      
loss = pd.Series(loss)


# In[8]:


plt.plot(loss)


# In[9]:


loss.to_excel("Loss_rate_15000_test.xlsx")


# In[ ]:




