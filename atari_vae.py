import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class VAE(nn.Module):
    def __init__(self, channel_in=3, ch=32, z=64, h_dim=512):
        super(VAE, self).__init__()
        self.channel_in = channel_in
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            #Flatten()

        )

        self.conv_mu = nn.Conv2d(ch*8, z, 2, 2)
        self.conv_log_var = nn.Conv2d(ch*8, z, 2, 2)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
        )

        
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.conv_mu(h), self.conv_log_var(h)
        #if self.channel_in != 4:
        #    mu = torch.flatten(mu, start_dim=1)
        #log_var = torch.flatten(log_var, start_dim=1)
        #encoding = self.sample(mu, log_var)
        #return self.decoder(encoding), mu, log_var
        
        return mu
        #return encoding


class LSTMVAE(nn.Module):
    def __init__(self, channel_in=3, ch=32, z=64, h_dim=512):
        super(LSTMVAE, self).__init__()
        self.channel_in = channel_in
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ELU(),
            #Flatten()

        )

        self.conv_mu = nn.Conv2d(ch*8, z, 2, 2)
        self.conv_log_var = nn.Conv2d(ch*8, z, 2, 2)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=6, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=8, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ELU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
        )

        
    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

        
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.conv_mu(h), self.conv_log_var(h)
        mu = torch.flatten(mu, start_dim=1)
        
        return mu