#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


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


# In[4]:


checkpoint = torch.load('./10000_after/all_rate15000_7.tar', map_location=device)
model = VAE().to(device)
model.load_state_dict(checkpoint['model'])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(checkpoint['optimizer'])
model.eval()


# In[5]:


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


# In[6]:


X = np.concatenate((X_Korea, X_USA, X_Japan, X_Dutch, X_England, X_France, X_China), axis = 1)
y = np.concatenate((y_Korea, y_USA, y_Japan, y_Dutch, y_England, y_France, y_China), axis = 1)


# In[7]:


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


# In[8]:


"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import TSNE
import plotly.express as px
import kaleido

# Test set save
mu_z_test = []
outy_test = []
with torch.no_grad():
    for idx, data in enumerate(X_test):
        x = data
        x = x.to(device)
        y = y_test[idx][:1,:]
        y = y.view(-1, 1)
        y = y.to(device)        
        out, mu, logVar, mu_y, var_log_y, pz_mean = model(x)
        mu_z_test.append(mu.detach().cpu().numpy())
        outy_test.append(y.detach().cpu().numpy()[0][0])
    
mu_z_test = np.array(mu_z_test)    
outy_test= np.array(outy_test)

# Train set save
mu_z_train = []
outy_train = []
with torch.no_grad():
    for idx, data in enumerate(X_train):
        x = data
        x = x.to(device)
        y = y_train[idx][:1,:]
        y = y.view(-1, 1)
        y = y.to(device)        
        out, mu, logVar, mu_y, var_log_y, pz_mean = model(x)
        mu_z_train.append(mu.detach().cpu().numpy())
        outy_train.append(y.detach().cpu().numpy()[0][0])
    
mu_z_train = np.array(mu_z_train)    
outy_train = np.array(outy_train)


# In[14]:


import plotly.offline as pyo
pyo.init_notebook_mode()

for i in range(mu_z_test.shape[1]):
    #extract last features only
    features = mu_z_test[:, i, :]

    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(features, )
    
    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=outy_test.reshape(-1)
        )
    fig.update_traces(marker_size=1)
    fig.write_image("./tsne_3d/result_tsne_rate15000_" + str(i+1) + "th.png")
    fig.show()


# In[12]:


features = mu_z_test[:, 0, :]
tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )
projections = pd.DataFrame(projections)
outy_test = pd.Series(outy_test)
all_proj = pd.concat([outy_test, projections], axis = 1)
all_proj.columns = ["Result", "X", "Y", "Z"]
all_proj.to_excel("Rate_TSNE_15000.xlsx")


# In[12]:


mu.shape


# In[ ]:




