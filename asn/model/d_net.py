import torch
import torch.nn as nn
from asn.utils.log import log


class Discriminator(nn.Module):
    """ Discriminator """

    def __init__(self, D_in, H, z_dim, d_out):
        super().__init__()
        self.z_dim = z_dim
        log.info('Discriminator domain net in_channels: {} out: {} hidden {}, z dim {}'.format(D_in, d_out, H, z_dim))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.Dropout2d(0.25),
            nn.ReLU(),
        )
        self.l_mu = nn.Linear(H, z_dim)
        self.l_var = nn.Linear(H, z_dim)
        # to output class layer
        self.out_layer = nn.ModuleList()
        for out_n in d_out:
            out = nn.Sequential(
                nn.Linear(z_dim, z_dim),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Linear(z_dim, out_n),
            )
            self.out_layer.append(out)

    def forward(self, x):
        enc = self.encoder(x)
        mu, logvar = self.l_mu(enc), self.l_var(enc)
        z = self.reparameterize(mu, logvar)
        return self.kl_loss(mu, logvar), [l(z) for l in self.out_layer]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu, logvar):
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD
