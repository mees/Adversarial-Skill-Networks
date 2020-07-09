"""
useage:
    python eval_asn.py --val-dir /data_vid/val --load_model mdl.pth.tar

"""
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
from asn.model.asn import create_model

from asn.utils.log import log
from asn.utils.train_utils import get_dataloader_val, vid_name_to_task, val_fit_task_lable
from asn.val.alignment import view_pair_alignment_loss
from asn.val.classification_accuracy import accuracy
from asn.val.embedding_visualization import visualize_embeddings

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-folder', type=str,
                        default='/tmp/asn_val_out')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--val-dir-metric',
                        type=str, default='~/asn_data/val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--task', type=str, default="cstack",help='dataset, load tasks for real block data (cstack)')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    log.info("args: {}".format(args))
    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}'.format(use_cuda))

    asn, start_epoch, global_step, _, _ = create_model(
        use_cuda, args.load_model)
    log.info('asn: {}'.format(asn.__class__.__name__))

    img_size=299

    vid_name_to_task_func=vid_name_to_task(args.task)

    dataloader_val = get_dataloader_val(args.val_dir_metric,
                                        args.num_views,
                                        args.batch_size,
                                        use_cuda)

    if use_cuda:
        asn.cuda()

    def model_forward(frame_batch):
        if use_cuda:
            frame_batch = frame_batch.cuda()
        emb = asn.forward(frame_batch)
        return emb # .data.cpu().numpy()

    asn.eval()
    # lable function: task name to labe
    visualize_embeddings(model_forward, dataloader_val,
                         save_dir=args.save_folder,
                         lable_func=vid_name_to_task_func)

    loss_val, nn_dist, dist_view_pais, _ = view_pair_alignment_loss(model_forward,
                                                                    args.num_views,
                                                                    dataloader_val)
