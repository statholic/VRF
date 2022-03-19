import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class VAE(nn.Module):
    def __init__(self, inChannels=1, OutChannels=1, InterDim=10, featureDim=15, zDim=1, Numcol=14):
        super(VAE, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.featureDim = featureDim
        self.InterDim = InterDim
        self.zDIm = zDim
        self.Numcol = Numcol

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
        x = x.view(-1, self.featureDim)
        x = torch.tanh(self.encFC0(x)).to(self.device)
        x = torch.tanh(self.encFC1(x)).to(self.device)
        mu = self.encFC2(x).to(self.device)
        logVar = self.encFC3(x).to(self.device)
        mu_y = self.encFC5(x).to(self.device)
        mu_y = mu_y.view(1, 1, -1).to(self.device)
        mu_y = self.encFC5_1(mu_y).to(self.device)
        var_log_y = self.encFC6(x).to(self.device)
        var_log_y = var_log_y.view(1, 1, -1).to(self.device)
        var_log_y = self.encFC6_1(var_log_y).to(self.device)
        
        return mu, logVar, mu_y, var_log_y

    def reparameterize1(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std1 = torch.exp(logVar/2).to(self.device)
        eps1 = torch.randn_like(std1).to(self.device)
        return mu + std1 * eps1
    
    def reparameterize2(self, mu_y, var_log_y):
        
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std2 = torch.exp(var_log_y/2).to(self.device)
        eps2 = torch.randn_like(std2).to(self.device)
        return mu_y + std2 * eps2
        

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = self.decFC0(z).to(self.device)
        x = self.decFC1(x).to(self.device)
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar, mu_y, var_log_y = self.encoder(x)
        z = self.reparameterize1(mu, logVar)
        y_f = self.reparameterize2(mu_y, var_log_y)
        pz_mean = self.pzFC0(y_f).to(self.device)
        pz_mean = pz_mean.view(-1, self.InterDim).to(self.device)
        out = self.decoder(z)
        mu_y = mu_y.view(-1, 1).to(self.device)
        var_log_y = var_log_y.view(-1, 1).to(self.device)
        return out, mu, logVar, mu_y, var_log_y, pz_mean
