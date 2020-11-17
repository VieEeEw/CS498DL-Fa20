import torch
from .spectral_normalization import SpectralNorm as sn
import torch.nn.functional as F
from torch import nn


class Discriminator(torch.nn.Module):
    def __init__(self, in_ch=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            sn(nn.Conv2d(in_ch, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            sn(nn.Conv2d(512, 1024, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, out_ch=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x
