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

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers import TensorBoardLogger

import asn
from asn.loss.metric_learning import (LiftedStruct,LiftedCombined)
from asn.utils.comm import get_git_commit_hash, data_loader_cycle
from asn.utils.train_utils import multi_vid_batch_loss
from asn.model.asn import create_model, save_model, KlDiscriminator

from asn.utils.train_utils import get_skill_dataloader,get_dataloader_val,vid_name_to_task,get_dataloader_train,val_fit_task_lable
from collections import OrderedDict
from torch.backends import cudnn
import pytorch_lightning as pl
# For fast training
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--name", type=str, default='tb_logs', help="experiment name")
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--task', type=str, default="cstack",help='dataset, load task from muliti view dataset (cstack or uulm action dataset)')
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str,help="metric loss lifted or liftedcombi", default="liftedcombi")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-domain-frames', type=int, default=2)
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15)
    parser.add_argument('--train-filter-tasks',help="task names to filter for training data, format, taskx,taskb. videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--num-example-batch',help="num example per batch each vid, only lifted loss support", type=int, default=4)
    return parser.parse_args()

class ASNConcatDataLoader(torch.utils.data.DataLoader):

    def __init__(self, hparams):
        # super().__init__(num_workers=hparams.num_workers)
        self.num_workers=hparams.num_workers
        self.train_filter_func = None
        self.hparams=hparams
        self.filter_func_domain = None
        self.img_size = 299
        assert hparams.batch_size>hparams.num_example_batch

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

        self.dataloader_train = get_dataloader_train(self.hparams.train_dir,
                                                self.hparams.num_views,
                                                self.hparams.batch_size,
                                                True,
                                                img_size=self.img_size,
                                                filter_func=self.train_filter_func,
                                                examples_per_seq=self.hparams.num_example_batch,
                                                num_workers =hparams.num_workers)

        all_view_pair_names = self.dataloader_train.dataset.get_all_comm_view_pair_names()
        self.key_views = ["frames views {}".format(i) for i in range(self.hparams.num_views)]
        # for every task one lable
        # used for validation
        self.vid_name_to_task_func = vid_name_to_task(self.hparams.task)
        self.transform_comm_name, self.num_domain_task_classes, self.task_names = val_fit_task_lable(
            self.vid_name_to_task_func,
            all_view_pair_names)
        log.info('train task names: {}'.format(self.task_names))
        self.lable_funcs = {'domain task lable': self.transform_comm_name}  # add domain lables to the sample
        self.dataloader_train_d = get_skill_dataloader(self.hparams.train_dir,
                                                       self.hparams.num_views,
                                                       self.hparams.batch_size,
                                                       True,
                                                       img_size=self.img_size,
                                                       filter_func=self.filter_func_domain,
                                                       lable_funcs=self.lable_funcs,
                                                       num_domain_frames=self.hparams.num_domain_frames,
                                                       stride=self.hparams.multi_domain_frames_stride,
                                                       num_workers =hparams.num_workers)
        self.iter_d = iter(data_loader_cycle(self.dataloader_train_d))
        self.iter_m = iter(data_loader_cycle(self.dataloader_train))

    def __iter__(self):
        return self

    def __next__(self):
        return [next(self.iter_m), next(self.iter_d)]

    def __len__(self):
        return len(self.dataloader_train)

class ASN(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        log.info(get_git_commit_hash(asn.__file__))
        self.hparams = hparams
        self.criterion = {"lifted": LiftedStruct(), "liftedcombi": LiftedCombined()}[self.hparams.loss]
        self.asn_model = create_model(
             self.hparams.load_model)
        self.in_channels = 32  # out embedding size
        # Discriminator network with iputs outputs depending on the args settings
        self.net_input = self.in_channels * self.hparams.num_domain_frames

        self.dataset = ASNConcatDataLoader(self.hparams)
        # load function which maps video file name to task for different datasets
        # not used for training only for evaluation
        self.d_net = KlDiscriminator(self.net_input, H=500, z_dim=64, D_out=[self.dataset.num_domain_task_classes])
        # self.dataloader_val = get_dataloader_val(hparams.val_dir,
                                            # hparams.num_views,
                                            # hparams.batch_size,
                                            # True)

    def forward(self, frame_batch):
        emb = self.asn_model.forward(frame_batch)
        return emb

    def _stack_multiview_vid(self, batch):
        img = torch.cat([batch[k] for k in self.dataset.key_views])
        n = batch[self.dataset.key_views[0]].size(0)
        # labels metric loss from video seq.
        label_positive_pair = np.arange(n)
        labels = Variable(torch.Tensor(np.concatenate([label_positive_pair for _ in self.dataset.key_views])))
        return img, labels

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        # do a custom way of backward
        if optimizer_idx==0:
            return loss.backward(retain_graph=True)
        else:
            return loss.backward()

    def training_step(self, batch, batch_idx, optimizer_idx):
        # set input and targets
        # d_out_gen = batch_d, batch_metric =batch
        batch_metric, batch_d = batch

        self.kl_loss, d_out = self._d_forward(batch_d, batch_metric)
        # maximize marginalized entropy over real samples to ensure equal usage
        entropy_margin = self.marginalized_entropy(d_out)
        # for next optimizer update
        self.d_out=d_out
        self.entropy_margin=entropy_margin
        # train generator
        if optimizer_idx == 0:
            # metric loss
            img, labels = self._stack_multiview_vid(batch_metric)
            embeddings = self(Variable(img))

            # METRIC loss
            if self.hparams.num_example_batch == 1:
                loss_metric = self.criterion(embeddings, labels)
                # loss and same debug input images
            else:
                loss_metric = multi_vid_batch_loss(self.criterion, embeddings, labels,
                                                   num_vid_example=self.hparams.num_example_batch)

            # max entro
            entropy1_fake = self.entropy(d_out)
            # ensure equal usage of fake samples
            loss_g = loss_metric * 0.1 + entropy1_fake + -1.0*entropy_margin

            tqdm_dict = {'g_loss': loss_g}
            output = OrderedDict({
                'loss': loss_g,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            # minimize entropy to make certain prediction of real sample
            entropy_real = self.entropy(self.d_out)
            d_loss = -1.0*self.entropy_margin + entropy_real + self.kl_loss

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def _d_forward(self, sample_batched_domain, batch_metric):

        img_domain, _ = self._stack_multiview_vid(sample_batched_domain)
        img_domain = img_domain.type_as(batch_metric[self.dataset.key_views[0]])
        emb_tcn = self(Variable(img_domain))
        if self.hparams.num_domain_frames != 1:
            # multiple frames as skills
            bl = emb_tcn.size(0)
            emb_size = emb_tcn.size(1)
            emb_tcn = emb_tcn.view(bl // self.hparams.num_domain_frames, self.hparams.num_domain_frames * emb_size)
        kl_loss, d_out_gen = self.d_net(emb_tcn)
        return kl_loss, d_out_gen[0]


    def train_dataloader(self):
        # return self.dataset.dataloader_train_d
        return self.dataset

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

    def training_epoch_end(self, outputs):
        save_model(self.asn_model,self.hparams.default_root_dir,self.current_epoch)
#         self.asn_model.eval()
        # def model_forward(b):
            # emb = asn.forward(b.cuda())
            # return emb
        # with torch.no_grad():
            # TODO fi
            # loss_val, *_ = view_pair_alignment_loss(model_forward,
                                                    # self.hparams.num_views,
                                                    # self.dataloader_val)
            # print('epoch {} loss_val: {}'.format(self.current_epoch,loss_val))
            # # lable function: task name to labe
            # visualize_embeddings(model_forward, dataloader_val,
                                 # save_dir=self.hparams.default_root_dir,
                                 # lable_func=self.vid_name_to_task_func)

        # self.asn_model.train()
        # self.logger.experiment.add_scalar("loss_val/aligment", loss_val, self.current_epoch)
        # self.logger.experiment.add_scalar("loss_val/aligment", loss_val, self.current_epoch)

        return {}

if __name__ == '__main__':
    args = get_args()
    if args.default_root_dir is None:
        args.default_root_dir='/tmp/asn'
    args.logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        name=args.name
    )
    print('args.default_root_dir: {}'.format(args.default_root_dir))

    asn = ASN(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(asn)
