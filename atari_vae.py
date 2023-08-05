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


class VAEBEV(nn.Module):
    def __init__(self, channel_in=3, ch=32, h_dim=512, z=32):
        super(VAEBEV, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channel_in, ch, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z)
        self.fc2 = nn.Linear(h_dim, z)
        self.fc3 = nn.Linear(z, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, ch*8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(ch*2, channel_in, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_().cuda()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def recon(self, z):
        z = self.fc3(z)
        return self.decoder(z)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.recon(z), mu, logvar
        
class VAE(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),

        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)
        self.conv_log_var = nn.Conv2d(ch*32, z, 1, 1)
        
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
        mu = torch.flatten(mu, start_dim=1)
        log_var = torch.flatten(log_var, start_dim=1)
        encoding = self.sample(mu, log_var)
        encoding = mu
        return self.decoder(encoding), mu, log_var
        #return mu




class TEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(TEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_mu(x)
        #x = torch.flatten(x, start_dim=1)
        return x




class Encoder(TEncoder):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(Encoder, self).__init__(channel_in=channel_in, ch=ch, z=z, h_dim=h_dim)
        self.adapter = nn.Conv2d(z, z, 1, 1)


    def forward(self, x):
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = self.adapter(x)
        #x = torch.flatten(x, start_dim=1)
        return x
