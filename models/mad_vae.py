import torch
from torch import nn
import torch.nn.functional as F

class ConvMADVAE(nn.Module):
    def __init__(self, z_dim=256):
        super(ConvMADVAE, self).__init__()

        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) 
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        
        
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)

       
        self.dec_fc = nn.Linear(z_dim, 1024)
        
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) 
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) 
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)    

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1) 
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 256, 2, 2)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        return torch.tanh(self.dec_conv4(x)) 

    def forward(self, x):
        mu, logvar = self.encode(x)
        
        
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z_repara = mu + (std * epsilon)
        
        x_recon = self.decode(z_repara)
        return x_recon, mu, logvar

