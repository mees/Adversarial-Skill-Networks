import argparse
import functools

import os
import time
import itertools

import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
from sklearn import preprocessing
from tensorboardX import SummaryWriter
from torch import multiprocessing, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from asn.loss.metric_learning import (LiftedStruct,LiftedCombined)
from asn.utils.comm import create_dir_if_not_exists, data_loader_cycle, create_label_func
from asn.utils.train_utils import get_metric_info_multi_example, log_train, multi_vid_batch_loss
from asn.model.asn import create_model, save_model, KlDiscriminator

from asn.utils.log import log
from asn.utils.train_utils import init_log_tb,get_skill_dataloader,get_dataloader_val,vid_name_to_task,get_dataloader_train,val_fit_task_lable
from asn.val.alignment import view_pair_alignment_loss
from asn.val.classification_accuracy import accuracy, accuracy_batch
from asn.val.embedding_visualization import visualize_embeddings
from torchvision.utils import save_image
from collections import OrderedDict
from torch.backends import cudnn
import pytorch_lightning as pl
# For fast training
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1500000)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--save-folder', type=str,
                        default='/tmp/asn_out')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--task', type=str, default="cstack",help='dataset, load tasks for real block data (cstack)')
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--val-dir-metric',
                        type=str, default='~/asn_data/val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str,help="metric loss lifted or liftedcombi", default="liftedcombi")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--plot-tsne', action="store_true", default=False)
    parser.add_argument('--num-domain-frames', type=int, default=2)
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15)
    parser.add_argument('--train-filter-tasks',help="task names to filter for training data, format, taskx,taskb. videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--num-example-batch',help="num example per batch each vid, only lifted loss support", type=int, default=4)
    return parser.parse_args()

# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets
#
#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)
#
#     def __len__(self):
#         return min(len(d) for d in self.datasets)

class ASN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.criterion = {"lifted": LiftedStruct(), "liftedcombi": LiftedCombined()}[self.hparams.loss]
        self.asn_model, self.start_epoch, self.global_step, _, _ = create_model(
             self.hparams.load_model)
        self.in_channels = 32  # out embedding size of asn
        # Discriminator network with iputs outputs depending on the args settings
        self.net_input = self.in_channels * self.hparams.num_domain_frames

        self.img_size = 299
        # load function which maps video file name to task for different datasets
        # not used for training only for evaluation
        self.vid_name_to_task_func = vid_name_to_task(self.hparams.task)
        self.train_filter_func = None
        self.filter_func_domain = None
        if self.hparams.train_filter_tasks is not None:
            # filter out tasks by names for the training set
            self.train_filter_tasks = self.hparams.train_filter_tasks.split(',')
            log.info('train_filter_tasks: {}'.format(self.train_filter_tasks))

            def train_filter_func(name, n_frames):
                return all(task not in name for task in self.train_filter_tasks)  # ABD->C

            def filter_func_domain(name, frames_cnt):
                ' return no fake exmaples for filtered tasks'
                return "fake" not in name and all(task not in name for task in self.train_filter_tasks)
        else:
            def filter_func_domain(name, frames_cnt):
                ' return no fake exmaples for filtered tasks'
                return "fake" not in name
        self.dataloader_train = get_dataloader_train(self.hparams.train_dir, self.hparams.num_views, self.hparams.batch_size,
                                                True,
                                                img_size=self.img_size,
                                                filter_func=self.train_filter_func,
                                                examples_per_seq=self.hparams.num_example_batch)
        all_view_pair_names = self.dataloader_train.dataset.get_all_comm_view_pair_names()
        # for every task one lable
        self.transform_comm_name, self.num_domain_task_classes, self.task_names = val_fit_task_lable(
            self.vid_name_to_task_func,
            all_view_pair_names)
        log.info('task mode task_names: {}'.format(self.task_names))

        self.lable_funcs = {'domain task lable': self.transform_comm_name}  # add domain lables to the sample
        self.d_net = KlDiscriminator(self.net_input, H=500, z_dim=64, D_out=[self.num_domain_task_classes], grad_rev=False)
        self.key_views = ["frames views {}".format(i) for i in range(self.hparams.num_views)]
        self.dataloader_train_domain = get_skill_dataloader(self.hparams.train_dir,
                                                       self.hparams.num_views,
                                                       self.hparams.batch_size,
                                                       True,
                                                       img_size=self.img_size,
                                                       filter_func=self.filter_func_domain,
                                                       lable_funcs=self.lable_funcs,
                                                       num_domain_frames=self.hparams.num_domain_frames,
                                                       stride=self.hparams.multi_domain_frames_stride)
        self.iter_domain = iter(data_loader_cycle(self.dataloader_train_domain))



    def forward(self, frame_batch):
        emb = self.asn_model.forward(frame_batch)
        return emb


    def training_step(self, batch, batch_idx, optimizer_idx):
        # one = torch.tensor(1.0, dtype=torch.float).cuda()
        # mone = torch.tensor(-1.0, dtype=torch.float).cuda()

        # set input and targets
        sample_batched_domain = next(self.iter_domain)

        img_domain = torch.cat([sample_batched_domain[self.key_views[0]],
                                sample_batched_domain[self.key_views[1]]])
        img_domain = img_domain.type_as(batch[self.key_views[0]])
        # emb_tcn = model_forward_domain(Variable(img_domain))
        emb_tcn = self(Variable(img_domain))
        if self.hparams.num_domain_frames != 1:
            # multiple frames as skills
            bl = emb_tcn.size(0)
            emb_size = emb_tcn.size(1)
            emb_tcn = emb_tcn.view(bl // self.hparams.num_domain_frames, self.hparams.num_domain_frames * emb_size)
        kl_loss, d_out_gen = self.d_net(emb_tcn)
        d_out_gen = d_out_gen[0]

        # train generator
        if optimizer_idx == 0:
            # metric loss
            img = torch.cat([batch[self.key_views[0]],
                             batch[self.key_views[1]]])
            embeddings = self(Variable(img))
            n = batch[self.key_views[0]].size(0)
            # labels metric loss from video seq.
            label_positive_pair = np.arange(n)
            labels = Variable(torch.Tensor(np.concatenate([label_positive_pair, label_positive_pair])))

            # METRIC loss
            if self.hparams.num_example_batch == 1:
                loss_metric = self.criterion(embeddings, labels)
            elif self.trainer.global_step == 1:
                # loss and same debug input images
                loss_metric = multi_vid_batch_loss(self.criterion, embeddings, labels, self.hparams.num_views,
                                                   num_vid_example=self.hparams.num_example_batch, img_debug=img,
                                                   save_folder=self.hparams.save_folder)
            else:
                loss_metric = multi_vid_batch_loss(self.criterion, embeddings, labels, self.hparams.num_views,
                                                   num_vid_example=self.hparams.num_example_batch)


            # max entro
            entropy1_fake = self.entropy(d_out_gen)
            entropy_margin = self.marginalized_entropy(d_out_gen)
            # #ensure equal usage of fake samples
            loss_g = loss_metric * 0.1 + entropy1_fake + -1.0*entropy_margin

            tqdm_dict = {'g_loss': loss_g}
            output = OrderedDict({
                'loss': loss_g,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
                     # maximize marginalized entropy over real samples to ensure equal usage
            entropy_margin = self.marginalized_entropy(d_out_gen)
            # minimize entropy to make certain prediction of real sample
            entropy1_real = self.entropy(d_out_gen)
            d_loss = -1.0*entropy_margin + entropy1_real + kl_loss

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def train_dataloader(self):
        return self.dataloader_train

    # def val_dataloader(self):
    #     dataloader_val = get_dataloader_val(self.hparams.val_dir_metric,
    #                                         self.hparams.num_views, self.hparams.batch_size, True, None)
    #     return dataloader_val

    def configure_optimizers(self):
        optimizer_g = torch.optim.Adam(self.asn_model.parameters(), lr=self.hparams.lr)
        optimizer_d = torch.optim.Adam(self.d_net.parameters(), lr=self.hparams.lr)
        return [optimizer_g, optimizer_d], []

    # marginalized entropy
    def marginalized_entropy(self, y):
        y = F.softmax(y, dim=1)
        y1 = y.mean(0)
        y2 = -torch.sum(y1 * torch.log(y1 + 1e-6))
        return y2

    def entropy(self, y):
        y1 = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
        y2 = 1.0 / y.size(0) * y1.sum()
        return y2

if __name__ == '__main__':
    args = get_args()
    log.info("args: {}".format(args))

    asn = ASN(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(asn)
