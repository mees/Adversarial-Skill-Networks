import argparse
import functools

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from asn.loss.metric_learning import (LiftedStruct,LiftedCombined)
from asn.loss.entro import (marginalized_entropy, entropy)
from asn.utils.comm import create_dir_if_not_exists, data_loader_cycle, create_label_func
from asn.utils.train_utils import get_metric_info_multi_example, log_train, multi_vid_batch_loss
from asn.model.asn import create_model, save_model
from asn.model.d_net import Discriminator
from asn.utils.train_utils import get_dataloader_val, get_dataloader_train, get_domain_dataloader_train
from asn.utils.log import log
from asn.utils.train_utils import init_log_tb
from asn.utils.train_utils import combi_push_stack_color_to_task, _fit_task_label,uulmMAD_file_name_to_task
from asn.val.alignment import view_pair_alignment_loss
from asn.val.embedding_visualization import visualize_embeddings
from torch.backends import cudnn

# For fast training
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1500000)
    parser.add_argument('--save-folder', type=str, default='/tmp/asn')
    parser.add_argument('--load-model', type=str, required=False, help='path to model_best.pth.tar file')
    parser.add_argument('--val-step', type=int, default=1000, help='Validation every n step')
    parser.add_argument('--task', type=str, default="cstack", help='mulitview dataset, for real block data with cstack" or persion action dataset with "uulmMAD")')
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--val-dir-metric', type=str, default='~/asn_data/val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str,help="metric loss lifted or liftedcombi", default="liftedcombi")
    parser.add_argument('--lr-d', type=float, default=0.0001)
    parser.add_argument('--lr-g', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2,help='number for campera views in the dataset')
    parser.add_argument('--plot-tsne', action="store_true", default=False)
    parser.add_argument('--num-domain-frames', type=int, default=2,help='num of embedded frames defining a skill')
    parser.add_argument('--emb-dim', type=int, default=32, help='embedding dimention')
    parser.add_argument('--d-net-hidden-dim', type=int, default=500, help='hidden dimention for the Discriminator network')
    parser.add_argument('--d-net-z-dim', type=int, default=64, help='z dimention for the Discriminator network')
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15, help='num of frames between each sampeled frame defining a skil')
    parser.add_argument('--train-filter-tasks',
            help="task names to filter for training data, format: 'taskx,taskb' videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--num-example-batch',help="num of frames from different video pairs sampeld for a batch", type=int, default=4)
    return parser.parse_args()

def main():
    args = get_args()
    log.info("args: {}".format(args))
    writer = init_log_tb(args.save_folder)
    use_cuda = torch.cuda.is_available()
    criterion = {"lifted":LiftedStruct(),"liftedcombi":LiftedCombined()}[args.loss]
    log.info("criterion: for {} ".format(
        criterion.__class__.__name__))

    asn, global_step_start, _, _ = create_model(
        use_cuda, args.load_model, embedding_size = args.emb_dim)
    log.info('asn: {}'.format(asn.__class__.__name__))

    loss_val_min = None
    loss_val_min_step = 0

    # load function which maps video file name to task for different datasets
    vid_name_to_task={"cstack":combi_push_stack_color_to_task,
                      "uulmMAD":uulmMAD_file_name_to_task}[args.task]
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
                                                    criterion,use_cuda,
                                                    filter_func=train_filter_func,
                                                    examples_per_seq=examples_per_seq)
    all_view_pair_names = dataloader_train.dataset.get_all_comm_view_pair_names()
    all_view_pair_frame_lengths = dataloader_train.dataset.frame_lengths

    # for every task one lable based on video name
    # not used to train the models
    transform_comm_name, num_domain_task_classes,task_names=_fit_task_label(vid_name_to_task,all_view_pair_names)
    log.info('task names: {}'.format(task_names))

    # func to transform video name to a task labled
    lable_funcs={'domain task lable':transform_comm_name}
    num_domain_frames=args.num_domain_frames

    # embedding class
    log.info('num_domain_frames: {}'.format(num_domain_frames))

    # Discriminator network with iputs outputs depending on the args settings
    net_input = args.emb_dim*num_domain_frames
    d_net = Discriminator(net_input, H=args.d_net_hidden_dim, z_dim=args.d_net_z_dim, d_out = [num_domain_task_classes])

    # DATA domain
    # filter out fake examples and tasks for D net
    stride=args.multi_domain_frames_stride
    if args.train_filter_tasks is not None:
        def filter_func_domain(name,frames_cnt):
            ' return no fake exmaples for filtered tasks'
            return "fake" not in name and all(task not in name for task in train_filter_tasks)
    else:
        filter_func_domain = None
        def filter_func_domain(name,frames_cnt):
            ' return no fake exmaples for filtered tasks'
            return "fake" not in name

    dataloader_train_domain = get_domain_dataloader_train(args.train_dir,
                                                          args.num_views,
                                                          args.batch_size,
                                                          criterion,use_cuda,
                                                          filter_func=filter_func_domain,
                                                          lable_funcs=lable_funcs,
                                                          num_domain_frames=num_domain_frames,
                                                          stride=stride)
    if use_cuda:
        torch.cuda.seed()
        criterion.cuda()
        asn.cuda()
        d_net.cuda()

    def model_forward(frame_batch, to_numpy=True):
        if use_cuda:
            frame_batch = frame_batch.cuda()
        feat, emb, kl_loss = asn.forward(frame_batch)
        if to_numpy:
            return emb.data.cpu().numpy()
        else:
            return emb, kl_loss

    model_forward_domain=functools.partial(model_forward,to_numpy=False)

    def domain_val_forward(x, out_idx=0):
        emb,_=model_forward_domain(x)
        if num_domain_frames !=1:
            bl=emb.size(0)
            emb_size=emb.size(1)
            emb=emb.view(bl//num_domain_frames,num_domain_frames*emb_size)
        out = d_net(emb)
        kl_loss, out = out
        return out[out_idx] if isinstance(out, (tuple, list)) else out

    def domain_val_forward_kl_sample(x, out_idx=0):
        # return sampled z to plt distributuion kl
        emb,_=model_forward_domain(x)
        if num_domain_frames !=1:
            bl=emb.size(0)
            emb_size=emb.size(1)
            emb=emb.view(bl//num_domain_frames,num_domain_frames*emb_size)
        _ = d_net(emb)
        return d_net.z
    # define optimizer for encoder (g) and Discriminator (d)
    params_asn = filter(lambda p: p.requires_grad, asn.parameters())
    optimizer_g = optim.Adam(params_asn, lr=args.lr_d)
    optimizer_d = optim.Adam(d_net.parameters(), lr=args.lr_g)

    assert isinstance(criterion, (LiftedStruct, LiftedCombined))
    key_views = ["frames views {}".format(i) for i in range(args.num_views)]
    iter_metric = iter(data_loader_cycle(dataloader_train))
    iter_domain = iter(data_loader_cycle(dataloader_train_domain))
    asn.train()

    for global_step in range(global_step_start, args.steps):

        # =======================================================
        # update the encoder network
        sample_batched = next(iter_metric)
        # metric loss
        img = torch.cat([sample_batched[key_views[0]],
                         sample_batched[key_views[1]]])
        embeddings, _ = model_forward(Variable(img), False)
        n = sample_batched[key_views[0]].size(0)
        anchor_emb, positive_emb = embeddings[:n], embeddings[n:]
        label_positive_pair = np.arange(n)
        labels = Variable(torch.Tensor(np.concatenate([label_positive_pair, label_positive_pair]))).cuda()
        if examples_per_seq ==1:
            loss_metric = criterion(embeddings, labels)
        else:
            loss_metric = multi_vid_batch_loss(criterion,embeddings, labels,
                                               num_vid_example=examples_per_seq)

        # set input and targets
        sample_batched_domain = next(iter_domain)

        img_domain = torch.cat([sample_batched_domain[key_views[0]],
                                 sample_batched_domain[key_views[1]]])
        emb_tcn, kl_loss_tcn_domain = model_forward_domain(Variable(img_domain))

        if num_domain_frames !=1:
            # multiple frames as input
            bl=emb_tcn.size(0)
            emb_size=emb_tcn.size(1)
            emb_tcn=emb_tcn.view(bl//num_domain_frames,num_domain_frames*emb_size)
            # mask out lable for cat view

        kl_loss, d_out_gen = d_net(emb_tcn)
        d_out_gen= d_out_gen[0]

        # min the entro for diffent classes
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        #ensure equal usage of fake samples
        loss_g=loss_metric*0.5

        mone = torch.FloatTensor([-1]).cuda()
        one = torch.FloatTensor([1]).cuda()
        # max entro
        entropy_fake = entropy(d_out_gen)
        entropy_fake.backward(retain_graph=True)
        entorpy_margin =-1. * marginalized_entropy(d_out_gen)
        # ensure equal usage of fake samples
        entorpy_margin.backward(retain_graph=True)

        # update the encoder network
        loss_g.backward(retain_graph=True)
        optimizer_g.step()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # =======================================================
        # update the Discriminator

        # maximize marginalized entropy over real samples to ensure equal usage
        entorpy_margin = -1. * marginalized_entropy(d_out_gen)
        entorpy_margin.backward(retain_graph=True)
        # minimize entropy to make certain prediction of real sample
        entropy_real =  -1. * entropy(d_out_gen)
        entropy_real.backward(retain_graph=True)
        kl_loss.backward()
        optimizer_d.step()


        if global_step % 50 == 0 or global_step == 1:
            # log training
            loss_metric = loss_g.data.cpu().numpy().item()
            mi=get_metric_info_multi_example(anchor_emb.data.cpu().numpy(),positive_emb.data.cpu().numpy(),1)
            log_train(writer, mi, loss_metric, criterion, entropy_fake, global_step)

        # =======================================================
        # Validation
        if global_step % args.val_step == 0:
            log.info("==============================")
            asn.eval()

            if  args.plot_tsne and global_step % 20000 == 0:
                def emb_only(d):
                    return model_forward(d,to_numpy=False)[0]
                # save a tsne plot
                visualize_embeddings(emb_only, dataloader_val, summary_writer=None,
                                     global_step=global_step, save_dir=args.save_folder,
                                     lable_func=vid_name_to_task)
            loss_val, nn_dist, dist_view_pais, frame_distribution_err_cnt = view_pair_alignment_loss(model_forward,
                                                                                              args.num_views,
                                                                                              dataloader_val)
            asn.train()

            writer.add_histogram("val/frame_error_count",
                                 np.array(frame_distribution_err_cnt), global_step)
            writer.add_scalar('val/alignment_loss',
                              loss_val, global_step)
            writer.add_scalar('val/nn_distance',
                              nn_dist, global_step)
            writer.add_scalar(
                'val/distance_view_pairs_same_frame', dist_view_pais, global_step)

            is_best = False
            if loss_val_min is None or loss_val < loss_val_min:
                loss_val_min = loss_val
                loss_val_min_step = global_step
                is_best = True

            msg = "Validation alignment loss: {}, nn mean dist {:.3}, lowest loss {:.4} at {} steps".format(
                loss_val, nn_dist, loss_val_min, loss_val_min_step)
            log.info(msg)
            save_model(asn, optimizer_g, args, is_best,
                       args.save_folder, global_step)

    writer.close()

if __name__ == '__main__':
    main()
