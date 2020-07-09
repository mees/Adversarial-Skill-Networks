# based on:
# https://github.com/kekeblom/tcn
import datetime
import os
import shutil

import numpy as np

import torch
from torch import nn
from torch.autograd import Function, Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from asn.utils.log import log
from torchvision import models

class KlDiscriminator(nn.Module):
    ''' dis with n dist '''

    def __init__(self, D_in, H, z_dim, D_out, grad_rev):
        super().__init__()
        self._rev_grad=grad_rev
        self.z_dim=z_dim
        log.info('KlDiscriminator domain net in_channels: {} out: {} hidden {}, z dim {}'.format(D_in,D_out,H,z_dim))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            nn.Dropout2d(0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            nn.Dropout2d(0.25),
            torch.nn.ReLU(),
        )
        self.l_mu=nn.Linear(H, z_dim)
        self.l_var=nn.Linear(H, z_dim)
        # to output class layer
        self.out_layer = nn.ModuleList()
        for out_n in D_out:
            out = torch.nn.Sequential(
                torch.nn.Linear(z_dim, z_dim),
                nn.Dropout2d(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(z_dim, out_n),
            )
            self.out_layer.append(out)

    def forward(self, x):
        if self._rev_grad:
            x=grad_reverse(x)
        enc = self.encoder(x)
        mu,logvar=self.l_mu(enc),self.l_var(enc)
        z = self.reparameterize(mu, logvar)
        self.z=z
        return self.kl_loss(mu,logvar),[l(z) for l in self.out_layer]


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def kl_loss(self,mu, logvar):

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return KLD


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        x = self.conv2d(x)
        return F.relu(x, inplace=True)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class EmbeddingNet(nn.Module):
    def normalize(self, x):
        # Normalize output such that output lives on unit sphere.
        # buffer = torch.pow(x, 2)
        # normp = torch.sum(buffer, 1).add_(1e-10)
        # normalization_constant = torch.sqrt(normp)
        # output = torch.div(x, normalization_constant.view(-1, 1).expand_as(x))
        # return output
        # L2n
        return F.normalize(x, p=2, dim=1)


class SpatialSoftmax(nn.Module):
    '''
        This is a form of spatial attention over the activations.
        See more here: http://arxiv.org/abs/1509.06113
        based on:
        https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834

        feature.shape height, width,
    '''

    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super().__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., self.height),
                                   np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(
            self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
                # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(
                2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) *
                               softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) *
                               softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints



class MiniTCNModel(nn.Module):
    '''
        input is img 128x128
    based on https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py'''
    def __init__(self, nc=3, ndf=128,fc_hidden_sizes=1028, # 1024 todo
                 embedding_size=32,
                 dp_ratio_pretrained_act=0.2,dp_ratio_fc=0.3):
        super().__init__()
                # encoder
        self.ndf=ndf
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.drop_h4 = nn.Dropout(dp_ratio_pretrained_act)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)
        self.drop_h5 = nn.Dropout(dp_ratio_pretrained_act)
        self.fc1 =Dense(ndf*8*4*4, fc_hidden_sizes,activation=self.leakyrelu)
        self.fc1_drop=nn.Dropout(p=dp_ratio_fc)
        self.fc2 = nn.Linear(fc_hidden_sizes, embedding_size)

    def forward(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.drop_h4(self.leakyrelu(self.bn4(self.e4(h3))))
        h5 = self.drop_h5(self.leakyrelu(self.bn5(self.e5(h4))))
        h5 = h5.view(-1, self.ndf*8*4*4)
        f1= self.fc1_drop(self.fc1(h5))
        return self.fc2(f1)

class TCNModel(EmbeddingNet):
    """TCN Embedder V1.

    InceptionV3 (mixed_5d) -> conv layers -> spatial softmax ->
        fully connected -> optional l2 normalize -> embedding.
    """

    def __init__(self, inception,
                 additional_conv_sizes=[512, 512],
                 fc_hidden_sizes=[2048],
                 embedding_size=32,
                 dp_ratio_pretrained_act=0.2,
                 dp_ratio_conv=1.,
                 dp_ratio_fc=0.2,
                 l2_normalize_output=False,
                 finetune_inception=False):
        super().__init__()
        self.embedding_size=embedding_size
        log.info('finetune_inception: {}'.format(finetune_inception))
        if not finetune_inception:
            # disable trainingn for inception v3
            for child in inception.children():
                for param in child.parameters():
                    param.requires_grad = False

        # see:
        # https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        self.inception_end_point_mixed_5d = nn.ModuleList(
            [inception.Conv2d_1a_3x3,
             inception.Conv2d_2a_3x3,
             inception.Conv2d_2b_3x3,
             nn.MaxPool2d(kernel_size=3, stride=2),
             inception.Conv2d_3b_1x1,
             inception.Conv2d_4a_3x3,
             nn.MaxPool2d(kernel_size=3, stride=2),
             inception.Mixed_5b,
             inception.Mixed_5c,
             inception.Mixed_5d])

        # TCN Conv Layer: Optionally add more conv layers.<
        in_channels = 288
        self.Conv2d_6n_3x3 = nn.ModuleList()
        if dp_ratio_pretrained_act < 1.:
            self.Conv2d_6n_3x3.append(nn.Dropout(p=dp_ratio_pretrained_act))
        # padding=1 so like in the tf =SAME
        for i, out_channels in enumerate(additional_conv_sizes):
            self.Conv2d_6n_3x3.append(Conv2d(in_channels, out_channels,
                                                      padding=1, kernel_size=3, stride=1))
            if dp_ratio_conv < 1.:
                self.Conv2d_6n_3x3.append(nn.Dropout(p=dp_ratio_conv))
            in_channels = out_channels

        # Take the spatial soft arg-max of the last convolutional layer.
        self.SpatialSoftmax = SpatialSoftmax(
            channel=512, height=35, width=35)  # nn.Softmax2d()
        self.FullyConnected7n = nn.ModuleList([Flatten()])
        in_channels = 1024  # out of SpatialSoftmax

        self.num_freatures=int(in_channels)
        for i, num_hidden in enumerate(fc_hidden_sizes):
            self.FullyConnected7n.append(Dense(in_channels, num_hidden,activation=F.relu))
            if dp_ratio_fc > 0.:
                self.FullyConnected7n.append(nn.Dropout(p=dp_ratio_fc))
            in_channels = num_hidden

        self.FullyConnected7n.append(Dense(in_channels, embedding_size))

        self._all_sequential_feature = nn.Sequential(*self.inception_end_point_mixed_5d,
                                                     *self.Conv2d_6n_3x3,
                                                     self.SpatialSoftmax)

        self._all_sequential_emb = nn.Sequential(*self.FullyConnected7n)
        self.l2_normalize_output = l2_normalize_output
        # use l2 norm with triplet loss
        if l2_normalize_output:
            log.info("model with l2 norm out")

    def forward(self, x):

        feature = self._all_sequential_feature(x)
        x = self._all_sequential_emb(feature)
        if self.l2_normalize_output:
            return self.normalize(x)
        else:
            return x



def define_model(pretrained=True, **kwargs):
    return MiniTCNModel()# DEDBUG
    # return TCNModel(models.inception_v3(pretrained=pretrained), **kwargs)


def create_model(use_cuda, load_model_file=None, **kwargs):
    asn = define_model(use_cuda, **kwargs)
    start_epoch, start_step = 0, 0
    optimizer_state_dict = None
    training_args = None
    if load_model_file:
        load_model_file = os.path.expanduser(load_model_file)
        assert os.path.isfile(
            load_model_file), "file not found {}".format(load_model_file)
        checkpoint = torch.load(load_model_file)
        start_epoch = checkpoint.get('epoch', 0)
        start_step = checkpoint.get('step', 0)
        training_args = checkpoint.get('training_args', None)
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        asn.load_state_dict(checkpoint['model_state_dict'])
        log.info("Restoring Model from: {}, epoch {}, step {}, datetime {}".format(
            load_model_file, start_epoch, start_step, checkpoint.get('datetime')))


    if use_cuda:
        asn = asn.cuda()
    return asn, start_epoch, start_step, optimizer_state_dict, training_args


def save_model(model, optimizer, training_args, is_best, model_folder, epoch, step):
    ''' '''

    state = {
        'datetime': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'epoch': epoch,
        'step': step,
        'training_args': training_args,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    model_folder = os.path.expanduser(model_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    filename = os.path.join(model_folder, 'model.pth.tar')
    torch.save(state, filename)

    checkpoint = torch.load(filename)

    log.info("Saved Model from: {}, epoch {}, step {}".format(
        filename, epoch, step))
    if is_best:
        filename_copy = os.path.join(model_folder, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_copy)
        log.info("copyed to model_best!")
