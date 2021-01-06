# based on:
# https://github.com/kekeblom/tcn
import datetime
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import models

from asn.utils.log import log


class BNConv2d(nn.Module):
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


class SpatialSoftmax(nn.Module):
    """
    This is a form of spatial attention over the activations.
    See more here: http://arxiv.org/abs/1509.06113
    based on:
    https://gist.github.com/jeasinema/1cba9b40451236ba2cfb507687e08834
    feature.shape height, width,
    """

    def __init__(self, height, width, channel, temperature=None):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


class EncoderModel(nn.Module):
    """Embedder V1. based on TCN
    InceptionV3 (mixed_5d) -> conv layers -> spatial softmax ->
    fully connected -> optional l2 normalize -> embedding.
    """

    def __init__(
        self,
        inception,
        additional_conv_sizes=[512, 512],
        fc_hidden_sizes=[2048],
        embedding_size=32,
        dp_ratio_pretrained_act=0.2,
        dp_ratio_conv=1.0,
        dp_ratio_fc=0.2,
        rnn_type=None,
        mode_gaussian_dist=False,
        latent_z_dim=512,
        rnn_forward_seqarade=False,
        l2_normalize_output=False,
        finetune_inception=False,
    ):
        super().__init__()
        self.gaussian_mode = mode_gaussian_dist
        self.embedding_size = embedding_size
        log.info("finetune_inception: {}".format(finetune_inception))
        if not finetune_inception:
            # disable training for inception v3
            for child in inception.children():
                for param in child.parameters():
                    param.requires_grad = False

        # see:
        # https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
        self.inception_end_point_mixed_5d = nn.ModuleList(
            [
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2),
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
            ]
        )

        in_channels = 288
        self.Conv2d_6n_3x3 = nn.ModuleList()
        if dp_ratio_pretrained_act < 1.0:
            self.Conv2d_6n_3x3.append(nn.Dropout(p=dp_ratio_pretrained_act))
        # padding=1 so like in the tf =SAME
        for i, out_channels in enumerate(additional_conv_sizes):
            self.Conv2d_6n_3x3.append(BNConv2d(in_channels, out_channels, padding=1, kernel_size=3, stride=1))
            if dp_ratio_conv < 1.0:
                self.Conv2d_6n_3x3.append(nn.Dropout(p=dp_ratio_conv))
            in_channels = out_channels

        # Take the spatial soft arg-max of the last convolutional layer.
        self.SpatialSoftmax = SpatialSoftmax(channel=512, height=35, width=35)  # nn.Softmax2d()
        self.FullyConnected7n = nn.ModuleList([Flatten()])
        in_channels = 1024  # out of SpatialSoftmax

        self.num_freatures = int(in_channels)
        for i, num_hidden in enumerate(fc_hidden_sizes):
            self.FullyConnected7n.append(Dense(in_channels, num_hidden, activation=F.relu))
            if dp_ratio_fc > 0.0:
                self.FullyConnected7n.append(nn.Dropout(p=dp_ratio_fc))
            in_channels = num_hidden

        if self.gaussian_mode:
            self.FullyConnected7n.append(Dense(in_channels, 512, activation=F.relu))
            self.l_mu = Dense(512, latent_z_dim)
            self.l_var = Dense(512, latent_z_dim)
            # out layer for sampeld lat var
            self.lat_sampled_out_emb = nn.ModuleList(
                [
                    Dense(latent_z_dim, 512, activation=F.relu),
                    nn.Dropout(p=0.2),
                    Dense(512, 512, activation=F.relu),
                    nn.Dropout(p=0.2),
                    Dense(512, embedding_size),
                ]
            )
            self._sequential_z_out = nn.Sequential(*self.lat_sampled_out_emb)
        else:
            self.FullyConnected7n.append(Dense(in_channels, embedding_size))

        self._all_sequential_feature = nn.Sequential(
            *self.inception_end_point_mixed_5d, *self.Conv2d_6n_3x3, self.SpatialSoftmax
        )

        self._all_sequential_emb = nn.Sequential(*self.FullyConnected7n)
        self.l2_normalize_output = l2_normalize_output
        # use l2 norm with triplet loss
        if l2_normalize_output:
            log.info("TCN with l2 norm out")

    def forward(self, x):
        feature = self._all_sequential_feature(x)
        y = self._all_sequential_emb(feature)
        if self.l2_normalize_output:
            return self.normalize(y)
        else:
            return y

    def normalize(self, x):
        return F.normalize(x, p=2, dim=1)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_loss_func(mu, logvar):
    # KL divergence losses summed over all elements and batch

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD


def define_model(pretrained=True, **kwargs):
    return EncoderModel(models.inception_v3(pretrained=pretrained), **kwargs)


def create_model(use_cuda, load_model_file=None, **kwargs):
    asn = define_model(use_cuda, **kwargs)
    start_step = 0
    optimizer_state_dict = None
    training_args = None
    if load_model_file:
        load_model_file = os.path.expanduser(load_model_file)
        assert os.path.isfile(load_model_file), "file not found {}".format(load_model_file)
        checkpoint = torch.load(load_model_file)
        start_step = checkpoint.get("step", 0)
        training_args = checkpoint.get("training_args", None)
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        asn.load_state_dict(checkpoint["model_state_dict"], strict=False)
        log.info(
            "Restoring Model from: {}, step {}, datetime {}".format(
                load_model_file, start_step, checkpoint.get("datetime")
            )
        )

    if use_cuda:
        asn = asn.cuda()
    return asn, start_step, optimizer_state_dict, training_args


def save_model(model, optimizer, training_args, is_best, model_folder, step):
    state = {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "step": step,
        "training_args": training_args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    model_folder = os.path.expanduser(model_folder)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    filename = os.path.join(model_folder, "model.pth.tar")
    torch.save(state, filename)

    log.info("Saved Model from: {}, step {}".format(filename, step))
    if is_best:
        filename_copy = os.path.join(model_folder, "model_best.pth.tar")
        shutil.copyfile(filename, filename_copy)
        log.info("copyed to model_best!")
