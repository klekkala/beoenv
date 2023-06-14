import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResDown, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)
        
    def forward(self, x):
        skip = self.conv3(self.AvePool(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """
    def __init__(self, channel_in, channel_out, scale=2):
        super(ResUp, self).__init__()
        
        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)
        
        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        
        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")
        
    def forward(self, x):
        skip = self.conv3(self.UpNN(x))
        
        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))
        
        x = F.rrelu(x + skip)
        return x
    

class Encoder(nn.Module):
    """
    Encoder block
    Built for a 3x64x64 image and will result in a latent vector of size z x 1 x 1
    As the network is fully convolutional it will work for images LARGER than 64
    For images sized 64 * n where n is a power of 2, (1, 2, 4, 8 etc) the latent feature map size will be z x n x n

    When in .eval() the Encoder will not sample from the distribution and will instead output mu as the encoding vector
    and log_var will be None
    """
    def __init__(self, channels, ch=64, z=512):
        super(Encoder, self).__init__()
        self.conv1 = ResDown(channels, ch)  # 64
        self.conv2 = ResDown(ch, 2*ch)  # 32
        self.conv3 = ResDown(2*ch, 4*ch)  # 16
        self.conv4 = ResDown(4*ch, 8*ch)  # 8
        self.conv5 = ResDown(8*ch, 8*ch)  # 4
        self.conv_mu = nn.Conv2d(8*ch, z, 2, 2)  # 2
        self.conv_log_var = nn.Conv2d(8*ch, z, 2, 2)  # 2

    def sample(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        mu = self.conv_mu(x)
        #if self.training:
        #    mu = self.conv_mu(x)
        #    log_var = self.conv_log_var(x)
        #    x = self.sample(mu, log_var)
        #else:
        #    mu = self.conv_mu(x)
        #    x = mu
        #    log_var = None

        return mu


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch=64, z=512):
        super(Decoder, self).__init__()
        self.conv1 = ResUp(z, ch*8)
        self.conv2 = ResUp(ch*8, ch*8)
        self.conv3 = ResUp(ch*8, ch*4)
        self.conv4 = ResUp(ch*4, ch*2)
        self.conv5 = ResUp(ch*2, ch)
        self.conv6 = ResUp(ch, ch//2)
        self.conv7 = nn.Conv2d(ch//2, channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        return x 


class LargeVAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """
    def __init__(self, channel_in, ch=64, z=512):
        super(LargeVAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """
        
        self.encoder = Encoder(channel_in, ch=ch, z=z)
        self.decoder = Decoder(channel_in, ch=ch, z=z)

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, log_var



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

ch = 16
from ray.rllib.models.torch.misc import same_padding


class SmallVAE(nn.Module):
    def __init__(self, channel_in=3, z=64, h_dim=512):
        super(SmallVAE, self).__init__()
        from IPython import embed
        pad1, out1 = same_padding([84, 84], 8, 4)
        pad2, out2 = same_padding(out1, 8, 4)
        pad3, out3 = same_padding(out2, 8, 4)
        #embed()
        self.encoder = nn.Sequential(

            nn.Conv2d(channel_in, 16, kernel_size=8, stride=4, padding=pad1[:2]),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=pad2[:2]),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=11, stride=1, padding=pad3[:2]),
            nn.ReLU(),
            #nn.Conv2d(ch*4, ch*8, kernel_size=4, stride=2),
            #nn.ReLU(),
            #nn.Flatten(),
            #nn.Linear(18432, 512)

        )

        #self.conv_mu = nn.Conv2d(256, z, 2, 2)
        #self.conv_log_var = nn.Conv2d(256, z, 2, 2)
        
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
        #x = x.type(torch.float32).permute(0, 3, 1, 2)
        h = self.encoder(x)
        
        #mu, log_var = self.conv_mu(h), self.conv_log_var(h)
        #mu = torch.flatten(mu, start_dim=1)
        #log_var = torch.flatten(log_var, start_dim=1)
        #encoding = self.sample(mu, log_var)
        #return self.decoder(encoding), mu, log_var
        return h