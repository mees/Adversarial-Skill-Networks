
import functools
import logging
import os
import time

import numpy as np

import argparse
import torch
import torch.nn as nn
import asn
from tensorboardX import SummaryWriter
from torch import multiprocessing, optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from asn.loss.metric_learning import LiftedStruct, TripletLoss
from asn.loss.metric_learning import NpairLoss as NPairLoss
from asn.loss.metric_learning import LiftedCombined
from asn.utils.comm import distance, get_git_commit_hash, create_dir_if_not_exists, sliding_window
from asn.utils.dataset import (DoubleViewPairDataset,
                                    MultiViewVideoDataset, ViewPairDataset)
from asn.utils.log import log, set_log_file
from asn.utils.sampler import ViewPairSequenceSampler
from torchvision import transforms
from torchvision.utils import save_image
from torch.backends import cudnn
import sklearn
from sklearn import preprocessing
from scipy.spatial import distance

IMAGE_SIZE = (299, 299) #inception_v3


# For fast training
cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--save-folder', type=str,
                        default='~/tcn_traind/asn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--val-dir',
                        type=str, default='~/tcn_data/val')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-filter-tasks',
            help="task names to filter for trainig data, fromat, taskx,taskb. videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--lr-start', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--num-example-batch',help="num exmaple per batch each video pair, only lifted loss support", type=int, default=1)
    parser.add_argument('--loss', type=str, default="lifted", help="metric loss triplet,lifted,npair,liftedcombi")

    return parser.parse_args()

class GradReverse(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -.6)


def grad_reverse(x):
    return GradReverse()(x)

def combi_push_stack_color_to_task(vid_file_comm,frame_idx=None,vid_len=None,csv_file=None,state_lable=None):
    ''' return task label for  push stadck color '''
    head, tail = os.path.split(vid_file_comm)
    # check for fake first
    for task in ["cstack","cpush","2blockstack","2block_stack","sort","sep"]:
        if task in tail:
            # return task +"_fake" if "fake"in tail else task
            return task
    raise ValueError("task not fount {}".format(tail))

def _fit_task_label(vid_name_to_task, all_view_pair_names):
    ''' returns func to encode a single file name to labes, for the task'''
    all_view_pair_names = [vid_name_to_task(f) for f in all_view_pair_names]
    comm_name_to_lable = preprocessing.LabelEncoder()
    comm_name_to_lable.fit(all_view_pair_names)
    num_classes=len(comm_name_to_lable.classes_)
    name_classes=comm_name_to_lable.classes_
    log.info("number of vid domains task: {}".format(
        len(comm_name_to_lable.classes_)))
    log.info("vid domains in train set: {}".format(name_classes))
    def transform_comm_name(vid_file_comm,*args,**kwargs):
        return comm_name_to_lable.transform([vid_name_to_task(vid_file_comm)])[0]
    return transform_comm_name, num_classes,name_classes


def get_train_transformer(img_size=IMAGE_SIZE[0]):
    transformer_train = transforms.Compose([
        transforms.ToPILImage(),
         # transforms.Resize(img_size),  #  original tcn with center crop (pouring), user for pybullet data
	transforms.CenterCrop(img_size),  # TODO original tcn with center crop(pouring)
        # transforms.RandomResizedCrop(IMAGE_SIZE[0], scale=(0.9, 1.0)), # for real data
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
#                               hue=0.03,# use for real block data
                               saturation=0.3),
        transforms.ToTensor(),
    ])
    return transformer_train


def get_dataloader_train(dir_vids, num_views, batch_size, criterion, use_cuda, img_size=IMAGE_SIZE[0], filter_func=None, lable_funcs=None,examples_per_seq=None):
    transformer_train = get_train_transformer(img_size)
    sampler = None
    shuffle = True
    test_normalize_transformer()
    if examples_per_seq is None:
        # sample one vid per batch used for lifted loss
        # used for default tcn for lifted and npair loss
        examples_per_batch=batch_size
    else:

        examples_per_batch=batch_size//examples_per_seq
    log.info('train data loader example per sequence: {}'.format(examples_per_seq))
    if isinstance(criterion, (NPairLoss, LiftedStruct,LiftedCombined, TripletLoss)):
        shuffle = False
        # only one view pair in batch
        # sim_frames = 5
        transformed_dataset_train = DoubleViewPairDataset(vid_dir=dir_vids,
                                                          number_views=num_views,
                                                          filter_func=filter_func,
                                                          lable_funcs=lable_funcs,
                                                          # random_view_index=True,
                                                          # std_similar_frame_margin_distribution=sim_frames,
                                                          transform_frames=transformer_train)
        # sample so that only one view pairs is in a batch
        sampler = ViewPairSequenceSampler(dataset=transformed_dataset_train,
                                          examples_per_sequence=examples_per_batch,
                                          # similar_frame_margin=3,# TODO
                                          batch_size=batch_size)
    else:
        transformed_dataset_train = MultiViewVideoDataset(vid_dir=dir_vids,
                                                          number_views=num_views,
                                                          # similar_frame_margin=sim_frames,
                                                          transform_frames=transformer_train)
    dataloader_train = DataLoader(transformed_dataset_train, drop_last=True,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  sampler=sampler,
                                  num_workers=4,
                                  pin_memory=use_cuda)

    return dataloader_train

def test_normalize_transformer():
    ''' test if boths val and test transformer contain norm or not '''
    contrain_norm= lambda x: any((isinstance(t,transforms.Normalize) for t in x.transforms))
    assert contrain_norm(get_val_transformer())==contrain_norm(get_train_transformer()), "transforms not same for training and val"

def get_val_transformer(img_size=IMAGE_SIZE[0]):
    transformer_val = transforms.Compose([
        transforms.ToPILImage(),  # expects rgb, moves channel to front
	 # transforms.Resize(img_size),
       transforms.CenterCrop(img_size),  # TODO original tcn with center crop
        transforms.ToTensor(),  # imgae 0-255 to 0. - 1.0
        # normalize https://pytorch.org/docs/master/torchvision/models.html
#        transforms.Normalize([0.485, 0.456, 0.406],
#                             [0.229, 0.224, 0.225])
    ])
    return transformer_val

def init_log_tb(save_folder):
    log.setLevel(logging.INFO)
    set_log_file(os.path.join(save_folder, "train.log"))

    log.info("torchtcn commit hash: {}".format(
        get_git_commit_hash(torchtcn.__file__)))
    tb_log_dir = os.path.join(
        os.path.expanduser(save_folder), "tensorboard_log")
    writer = SummaryWriter(tb_log_dir)
    # start_tb_task(tb_log_dir)
    return writer

def get_dataloader_val(dir_vids, num_views, batch_size, use_cuda, filter_func=None, img_size=IMAGE_SIZE[0]):
    transformer_val = get_val_transformer(img_size)
    dataset_val = ViewPairDataset(vid_dir=dir_vids,
                                  number_views=num_views,
                                  filter_func=filter_func,
                                  transform_frames=transformer_val)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=batch_size,  # * 2,
                                shuffle=False,
                                num_workers=4,
                                pin_memory=use_cuda)

    return dataloader_val


def multi_vid_batch_loss(criterion_metric, batch, targets, n_views ,num_vid_example, img_debug=None, save_folder=None):
    ''' mutlitple vid in bathc, metric loss for mutil example for frame ,only 2 view support'''
    batch_size=batch.size(0)
    emb_view0,emb_view1=batch[:batch_size//2],batch[batch_size//2:]
    t_view0,t_view1=targets[:batch_size//2],targets[batch_size//2:]
    batch_example_vid=emb_view0.size(0)//num_vid_example
    slid_vid = lambda x :sliding_window(x,winSize=batch_example_vid, step=batch_example_vid)
    loss = torch.zeros(1).cuda()
    loss = torch.autograd.Variable(loss)
    # compute loss for each video
    for emb_view0_vid, emb_view1_vid,t0,t1 in zip(slid_vid(emb_view0),slid_vid(emb_view1),slid_vid(t_view0),slid_vid(t_view1)):
        loss+=criterion_metric(torch.cat((emb_view0_vid, emb_view1_vid)),torch.cat((t0, t1)))
    if img_debug is not None:
        v0,v1=img_debug[:batch_size//2],img_debug[batch_size//2:]
        create_dir_if_not_exists(os.path.join(save_folder,"images/"))
        join_log_path = lambda x: os.path.join(save_folder,"images",x)
        for view_i, (v,b) in enumerate(zip(slid_vid(v0),slid_vid(v1))):
            f_name=join_log_path("multi_exampe_input_{}.png".format(view_i))
            save_image(torch.cat((v,b)),f_name)
    return loss

def save_image_input(img,sample_batched,key_views,save_folder):
    # same modle input images
    create_dir_if_not_exists(os.path.join(save_folder,"images/"))
    join_log_path = lambda x: os.path.join(save_folder,"images",x)
    for view_i, k in enumerate(key_views):
        f_name=join_log_path("tcn_view{}.png".format(view_i))
        save_image(sample_batched[k],f_name)
    save_image(img, join_log_path("batch_input.png"))

def get_metric_info(anchor_emb,positive_emb):
    info_metric={}
    # distance to positiv example
    info_metric['dist pos'] = np.linalg.norm(
        anchor_emb - positive_emb, axis=1).mean()
    dots, cosin_neg_dist_pos = [],[]
    for e1, e2 in zip(anchor_emb, positive_emb):
        dots.append(np.dot(e1-e1.mean(), e2-e2.mean()))
        cosin_neg_dist_pos.append(-distance.cosine(e1,e2)) # Histogram and Binomial Deviance losses,
    info_metric['dist pos dot'] = np.mean(dots) # mean free dot pro
    info_metric['dist pos cos'] = np.mean(cosin_neg_dist_pos)

    n = anchor_emb.shape[0]
    emb_pos = anchor_emb
    emb_neg = positive_emb
    cnt, d_neg, dots_neg,cosin_neg_dist_neg = 0., 0., 0.,0.
    # TODO rm this loop
    for i in range(n):
        for j in range(n):
            if j != i:
                d_negative = np.linalg.norm(
                    emb_pos[i] - emb_neg[j])
                d_neg += d_negative
                dots_neg += np.dot(emb_pos[i]-emb_pos[i].mean(),
                                   emb_neg[j]-emb_neg[j].mean())
                cosin_neg_dist_neg+=-distance.cosine(emb_pos[i], emb_neg[j])

                cnt += 1
    info_metric['dist neg'] = d_neg / cnt
    info_metric['dist neg dot']  = dots_neg / cnt
    info_metric['dist neg cos'] = cosin_neg_dist_neg/cnt
    return info_metric

def get_metric_info_multi_example(anchor_emb,positive_emb,num_vid_example):
    ''' get seperate info for each vid in batch '''
    batch_example_vid=anchor_emb.shape[0]//num_vid_example
    slid_vid = lambda x :sliding_window(x,winSize=batch_example_vid, step=batch_example_vid)
    info = None
    for a,p in zip(slid_vid(anchor_emb),slid_vid(positive_emb)):
        mi =get_metric_info(a,p)
        if info is None:
            info={k:[v] for k,v in mi.items()}
        else:
            for k,v in mi.items():
                info[k].append(v)

    return {k:np.mean(v) for k,v in info.items()}

def log_train(writer,mi,loss_metric,criterion_metric,global_step,fps,examples_per_seq):
        ''' log to tb and pritn log msg '''

        msg = "steps {}, dist: pos {:.2},neg {:.2},neg cos dist: pos {:.2},cos_neg {:.2}, loss metric:{:.3}, fps {:.4}".format(
            global_step, mi['dist pos'], mi["dist neg"], mi['dist pos cos'], mi['dist neg cos'],loss_metric, fps)
        log.info(msg)
        # log.info("dot pos: {:.4} dot neg: {:.4}".format(product_pos, product_neg))
        writer.add_scalar('train/loss' + criterion_metric.__class__.__name__,
                          loss_metric, global_step)
        writer.add_scalar('train/fps', fps, global_step)
        writer.add_scalars('train/distane',
                           {'positive':  mi['dist pos'],
                            'negative': mi['dist neg']}, global_step)
        writer.add_scalars('train/product',
                           {'positive':  mi['dist pos dot'],
                            'negative': mi['dist pos dot']}, global_step)

        writer.add_scalars('train/negativ_cosin_dist',
                           {'positive':  mi['dist pos cos'],
                            'negative': mi['dist neg cos']}, global_step)


