from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 3*256*306
d = 20
s = 300
class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size= (7,3), stride= 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size= 3, stride= 3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size= 3, stride= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 5, stride= 3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size= 3, stride= 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2,True)
        )
        self.l1=nn.Sequential(
            nn.Linear(1024*48, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, s * 2)
        )
        self.l2=nn.Sequential(
            nn.Linear(s, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, 1024*48)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(64, 3, kernel_size=(7,3), stride=3),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        enc = self.encoder(x)
        mu_logvar = self.l1(enc.view(len(x),-1)).view(-1, 2, s)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        dec = self.decoder(self.l2(z).view(len(z),1024, 6, 8))
        return dec, mu, logvar
