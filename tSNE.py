import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random
from utils import MinMaxScale
from utils import make_Tensor
from VAE import VAE
from sklearn.manifold import MDS
from datetime import date
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.offline as pyo


os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Determine to use CUDA or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# VAE parameters
featureDim = 15
InterDim = 7
Numcol = 14

# Initialize Hyperparameters
learning_rate = 1e-4
num_epochs = 100

model = VAE(Numcol=Numcol, InterDim=InterDim, featureDim=featureDim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
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

"""
The following part takes a random image from test loader to feed into the VAE.
Both the original image and generated image from the distribution are shown.
"""

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

features = mu_z_test[:, 0, :]
tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )
projections = pd.DataFrame(projections)
outy_test = pd.Series(outy_test)
all_proj = pd.concat([outy_test, projections], axis = 1)
all_proj.columns = ["Result", "X", "Y", "Z"]
all_proj.to_excel("Rate_TSNE_15000.xlsx")

mu.shape