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
from torch.backends import cudnn
# For fast training
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--val-dir-domain',
                        type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str,help="metric loss lifted or liftedcombi", default="liftedcombi")
    parser.add_argument('--lr-start', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--plot-tsne', action="store_true", default=True)
    parser.add_argument('--num-domain-frames', type=int, default=2)
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15)
    parser.add_argument('--train-filter-tasks',
            help="task names to filter for training data, format, taskx,taskb. videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--num-example-batch',help="num example per batch each vid, only lifted loss support", type=int, default=4)
    return parser.parse_args()


#marginalized entropy
def marginalized_entropy(y):
    y=F.softmax(y, dim=1)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+1e-6))
    return y2

def entropy(y):
    y1 = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
    y2 = 1.0/y.size(0)*y1.sum()
    return y2


if __name__ == '__main__':

    args = get_args()
    log.info("args: {}".format(args))
    writer = init_log_tb(args.save_folder)
    use_cuda = torch.cuda.is_available()
    criterion = {"lifted":LiftedStruct(),"liftedcombi":LiftedCombined()}[args.loss]
    log.info("criterion: for {} ".format(
        criterion.__class__.__name__))
    if use_cuda:
        torch.cuda.seed()
        criterion.cuda()

    asn, start_epoch, global_step, _, _ = create_model(
        use_cuda, args.load_model)
    log.info('asn: {}'.format(asn.__class__.__name__))

    img_size=128

    # var for train info
    loss_val_min = None
    loss_val_min_step = 0

    # load function which maps video file name to task for different datasets
    # not used for training only for evaluation
    vid_name_to_task_func=vid_name_to_task(args.task)

    filter_func=None
    dataloader_val = get_dataloader_val(args.val_dir_metric,
                                        args.num_views, args.batch_size,use_cuda,filter_func)

    train_filter_func=None
    if args.train_filter_tasks is not None:
        # filter out tasks by namnes for the training set
        train_filter_tasks= args.train_filter_tasks.split(',')
        log.info('train_filter_tasks: {}'.format(train_filter_tasks))
        def train_filter_func(name, n_frames):
            return all(task not in name for task in train_filter_tasks)#ABD->C
    examples_per_seq=args.num_example_batch
    dataloader_train = get_dataloader_train(args.train_dir, args.num_views, args.batch_size,
                                                    use_cuda,
                                                    img_size=img_size,
                                                    filter_func=train_filter_func,
                                                    examples_per_seq=examples_per_seq)

    all_view_pair_names = dataloader_train.dataset.get_all_comm_view_pair_names()
    all_view_pair_frame_lengths = dataloader_train.dataset.frame_lengths

    # for every task one lable
    transform_comm_name, num_domain_task_classes, task_names=val_fit_task_lable(vid_name_to_task_func,all_view_pair_names)
    log.info('task mode task_names: {}'.format(task_names))

    lable_funcs={'domain task lable':transform_comm_name}# add domain lables to the sample

    num_domain_frames=args.num_domain_frames
    log.info('num_domain_frames: {}'.format(num_domain_frames))

    # embedding class
    log.info('num_domain_frames: {}'.format(num_domain_frames))
    #in_channels = 1024*num_domain_frames  # out of SpatialSoftmax tcn

    in_channels = 32 # out embedding size of asn

    # Discriminator network with iputs outputs depending on the args settings
    net_input=in_channels*num_domain_frames
    d_net = KlDiscriminator(net_input, H=500,z_dim=64,D_out= [num_domain_task_classes], grad_rev=False)

    # DATA domain
    # filter out fake examples and tasks for D net
    stride=args.multi_domain_frames_stride
    if args.train_filter_tasks is not None:
        def filter_func_domain(name,frames_cnt):
            ' return no fake exmaples for filtered tasks'
            return "fake" not in name and all(task not in name for task in train_filter_tasks)
    else:
        def filter_func_domain(name,frames_cnt):
            ' return no fake exmaples for filtered tasks'
            return "fake" not in name

    dataloader_train_domain = get_skill_dataloader(args.train_dir,
                                                          args.num_views,
                                                          args.batch_size,
                                                          use_cuda,
                                                          img_size=img_size,
                                                          filter_func=filter_func_domain,
                                                          lable_funcs=lable_funcs,
                                                          num_domain_frames=num_domain_frames,
                                                          stride=stride)


    asn.train()


    if use_cuda:
        asn.cuda()
        d_net.cuda()

    def model_forward(frame_batch, to_numpy=True):
        if use_cuda:
            frame_batch = frame_batch.cuda()
        emb = asn.forward(frame_batch)
        if to_numpy:
            return emb.data.cpu().numpy()
        else:
            return emb


    model_forward_domain=functools.partial(model_forward,to_numpy=False)

    params = filter(lambda p: p.requires_grad, asn.parameters())

    optimizer_g = optim.Adam(params, lr=args.lr_start)
    optimizer_d = optim.Adam(d_net.parameters(), lr=args.lr_start)

    key_views = ["frames views {}".format(i) for i in range(args.num_views)]
    iter_metric=iter(data_loader_cycle(dataloader_train))
    iter_domain=iter(data_loader_cycle(dataloader_train_domain))

    mone = torch.FloatTensor([-1]).cuda()
    one = torch.FloatTensor([1]).cuda()


    for epoch in range(start_epoch, args.steps):

        global_step += 1
        sample_batched=next(iter_metric)
        # metric loss
        img = torch.cat([sample_batched[key_views[0]],
                         sample_batched[key_views[1]]])
        embeddings = model_forward(Variable(img), False)
        n = sample_batched[key_views[0]].size(0)

        # labels metric loss from video seq.
        label_positive_pair = np.arange(n)
        labels = Variable(torch.Tensor(np.concatenate([label_positive_pair, label_positive_pair]))).cuda()

        # METRIC loss
        if examples_per_seq ==1:
            loss_metric = criterion(embeddings, labels)
        elif global_step ==1:
            # loss and same debug input images
            loss_metric = multi_vid_batch_loss(criterion,embeddings, labels,args.num_views,
                                            num_vid_example=examples_per_seq,img_debug=img,
                                            save_folder=args.save_folder)
        else:
            loss_metric = multi_vid_batch_loss(criterion,embeddings, labels,args.num_views,
                                            num_vid_example=examples_per_seq)

        # set input and targets
        sample_batched_domain=next(iter_domain)

        img_domain = torch.cat([sample_batched_domain[key_views[0]],
                                 sample_batched_domain[key_views[1]]])
        emb_tcn =model_forward_domain(Variable(img_domain))

        if num_domain_frames !=1:
            # multiple frames as skills
            bl=emb_tcn.size(0)
            emb_size=emb_tcn.size(1)
            emb_tcn=emb_tcn.view(bl//num_domain_frames,num_domain_frames*emb_size)
            # mask out lable for cat view
            mask = torch.ByteTensor([i%num_domain_frames==0 for i in range(bl)]).cuda()

        kl_loss, d_out_gen = d_net(emb_tcn)
        d_out_gen= d_out_gen[0]

        # min the entro for diffent classes
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()


        loss_g=loss_metric*0.1

        # max entro
        entropy1_fake = entropy(d_out_gen)
        entropy1_fake.backward(one, retain_graph=True)
        entorpy_margin = marginalized_entropy(d_out_gen)
        # #ensure equal usage of fake samples
        entorpy_margin.backward(mone, retain_graph=True)
        loss_g=loss_metric
        loss_g.backward(retain_graph=True)
        optimizer_g.step()

        # update the Discriminator
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # maximize marginalized entropy over real samples to ensure equal usage
        entorpy_margin = marginalized_entropy(d_out_gen)
        entorpy_margin.backward(mone,retain_graph=True)
        # minimize entropy to make certain prediction of real sample
        entropy1_real = entropy(d_out_gen)

        entropy1_real.backward(mone,retain_graph=True)

        kl_loss.backward()

        optimizer_d.step()

        # var to monitor the training
        if global_step ==1:
            create_dir_if_not_exists(os.path.join(args.save_folder,"images/"))
            save_image(sample_batched_domain[key_views[0]], os.path.join(args.save_folder,"images/tcn_view0_domain.png"))
            save_image(sample_batched_domain[key_views[1]], os.path.join(args.save_folder,"images/tcn_view1_domain.png"))
            save_image(sample_batched[key_views[0]], os.path.join(args.save_folder,"images/tcn_view0.png"))
            save_image(sample_batched[key_views[1]], os.path.join(args.save_folder,"images/tcn_veiw1.png"))
            save_image(img, os.path.join(args.save_folder,"images/all.png"))

        if global_step % 100 == 0 or global_step == 1:
            # log train dist
            loss_metric = np.asscalar(loss_g.data.cpu().numpy())

            anchor_emb, positive_emb = embeddings[:n], embeddings[n:]
            mi=get_metric_info_multi_example(anchor_emb.data.cpu().numpy(),positive_emb.data.cpu().numpy())
            # log training info
            log_train(writer,mi,loss_metric,criterion,global_step)

        if global_step % 1000 == 0:
            # valGidation
            log.info("==============================")
            asn.eval()
            d_net.eval()


            if  args.plot_tsne and global_step % 20000 == 0:
                visualize_embeddings(model_forward, dataloader_val, summary_writer=None,
                                     global_step=global_step, save_dir=args.save_folder, lable_func=vid_name_to_task_func)
            loss_val, nn_dist, dist_view_pais, _ = view_pair_alignment_loss(model_forward,
                                                                                              args.num_views,
                                                                                              dataloader_val)
            asn.train()
            d_net.train()

            writer.add_scalar('validation/alignment_loss',
                              loss_val, global_step)
            writer.add_scalar('validation/nn_distance',
                              nn_dist, global_step)
            writer.add_scalar(
                'validation/distance_view_pairs_same_frame', dist_view_pais, global_step)

            if loss_val_min is None or loss_val < loss_val_min:
                loss_val_min = loss_val
                loss_val_min_step = global_step
                is_best = True
            else:
                is_best = False
            msg= "Validation alignment epoch {} loss: {}, nn mean dist {:.3}, lowest loss {:.4} at {} steps".format(
                epoch, loss_val, nn_dist, loss_val_min, loss_val_min_step)
            log.info(msg)
            print('msg: {}'.format(msg))
            save_model(asn, optimizer_g, args, is_best,
                       args.save_folder, epoch, global_step)

    writer.close()

