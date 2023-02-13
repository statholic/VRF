import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class VRF(nn.Module):
    def __init__(self, inChannels=1, OutChannels=1, InterDim=10, featureDim=15, zDim=1, Numcol=14):
        super(VRF, self).__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.featureDim = featureDim
        self.InterDim = InterDim
        self.zDim = zDim
        self.Numcol = Numcol

        self.pzFC0 = weight_norm(nn.Linear(zDim, InterDim*Numcol), name = 'weight')

        self.encFC0 = nn.Linear(featureDim, featureDim)
        self.encFC1 = nn.Linear(featureDim, InterDim)
        self.encFC2 = nn.Linear(InterDim, InterDim)
        self.encFC3 = nn.Linear(InterDim, InterDim)
        self.encFC4 = nn.Linear(InterDim, InterDim)
        self.encFC5 = nn.Linear(InterDim, zDim)
        self.encFC5_1 = nn.Linear(Numcol, zDim)
        self.encFC6 = nn.Linear(InterDim, zDim)
        self.encFC6_1 = nn.Linear(Numcol, zDim)

        self.decFC0 = nn.Linear(InterDim, InterDim)
        self.decFC1 = nn.Linear(InterDim, featureDim)

    def encoder(self, x):

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

        std1 = torch.exp(logVar/2).to(self.device)
        eps1 = torch.randn_like(std1).to(self.device)
        return mu + std1 * eps1
    
    def reparameterize2(self, mu_y, var_log_y):
        
        std2 = torch.exp(var_log_y/2).to(self.device)
        eps2 = torch.randn_like(std2).to(self.device)
        return mu_y + std2 * eps2
        

    def decoder(self, z):

        x = self.decFC0(z).to(self.device)
        x = self.decFC1(x).to(self.device)
        return x

    def forward(self, x):

        mu, logVar, mu_y, var_log_y = self.encoder(x)
        z = self.reparameterize1(mu, logVar)
        y_f = self.reparameterize2(mu_y, var_log_y)
        pz_mean = self.pzFC0(y_f).to(self.device)
        pz_mean = pz_mean.view(-1, self.InterDim).to(self.device)
        out = self.decoder(z)
        mu_y = mu_y.view(-1, 1).to(self.device)
        var_log_y = var_log_y.view(-1, 1).to(self.device)
        return out, mu, logVar, mu_y, var_log_y, pz_mean
