import argparse
import functools

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn

from asn.loss.entro import (marginalized_entropy, entropy)
from asn.loss.metric_learning import (LiftedStruct, LiftedCombined)
from asn.model.asn import create_model, save_model
from asn.model.d_net import Discriminator
from asn.utils.comm import data_loader_cycle
from asn.utils.log import log
from asn.utils.train_utils import get_dataloader_val, get_dataloader_train, get_skill_dataloader, \
    transform_vid_name_to_task
from asn.utils.train_utils import get_metric_info_multi_example, log_train, multi_vid_batch_loss
from asn.utils.train_utils import init_log_tb
from asn.utils.train_utils import val_fit_task_label
from asn.val.alignment import view_pair_alignment_loss
from asn.val.embedding_visualization import visualize_embeddings

# For fast training
cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1500000)
    parser.add_argument('--save-folder', type=str, default='/tmp/asn')
    parser.add_argument('--load-model', type=str, required=False, help='path to model_best.pth.tar file')
    parser.add_argument('--val-step', type=int, default=1000, help='Validation every n step')
    parser.add_argument('--task', type=str, default="cstack",
                        help='multiview dataset, for real block data with cstack" or persion action dataset with "uulmMAD")')
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--val-dir-metric', type=str, default='~/asn_data/val')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str, help="metric loss lifted or liftedcombi", default="liftedcombi")
    parser.add_argument('--lr-d', type=float, default=0.0001)
    parser.add_argument('--lr-g', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2, help='number for camera views in the dataset')
    parser.add_argument('--plot-tsne', action="store_true", default=False)
    parser.add_argument('--num-domain-frames', type=int, default=2, help='num of embedded frames defining a skill')
    parser.add_argument('--emb-dim', type=int, default=32, help='embedding dimension')
    parser.add_argument('--d-net-hidden-dim', type=int, default=500,
                        help='hidden dimension for the Discriminator network')
    parser.add_argument('--d-net-z-dim', type=int, default=64, help='z dimension for the Discriminator network')
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15,
                        help='num of frames between each sampled frame defining a skill')
    parser.add_argument('--train-filter-tasks',
                        help="task names to filter for training data, format: 'taskx,taskb' videos with the file name will be filtered",
                        type=str, default=None)
    parser.add_argument('--num-example-batch', help="num of frames from different video pairs sampled for a batch",
                        type=int, default=4)
    return parser.parse_args()


def model_forward(frame_batch, mdl, use_cuda, to_numpy=True):
    if use_cuda:
        frame_batch = frame_batch.cuda()
    emb = mdl.forward(frame_batch)
    if to_numpy:
        return emb.data.cpu().numpy()
    else:
        return emb


def main():
    args = get_args()
    log.info("args: {}".format(args))
    writer = init_log_tb(args.save_folder)
    use_cuda = torch.cuda.is_available()
    print('use_cuda: {}'.format(use_cuda))
    criterion = {"lifted": LiftedStruct(), "liftedcombi": LiftedCombined()}[args.loss]
    log.info("criterion: for {} ".format(
        criterion.__class__.__name__))

    asn, global_step_start, _, _ = create_model(
        use_cuda, args.load_model, embedding_size=args.emb_dim)
    log.info('asn: {}'.format(asn.__class__.__name__))
    asn.train()

    # load function which maps video file name to task for different datasets
    vid_name_to_task = transform_vid_name_to_task(args.task)
    dataloader_val = get_dataloader_val(args.val_dir_metric,
                                        args.num_views, args.batch_size, use_cuda)

    train_filter_func = None
    if args.train_filter_tasks is not None:
        # filter out tasks by names for the training set
        train_filter_tasks = args.train_filter_tasks.split(',')
        log.info('train_filter_tasks: {}'.format(train_filter_tasks))

        def train_filter_func(name, n_frames):
            return all(task not in name for task in train_filter_tasks)  # ABD->C
    examples_per_seq = args.num_example_batch
    dataloader_train = get_dataloader_train(args.train_dir, args.num_views, args.batch_size,
                                            use_cuda,
                                            img_size=299,
                                            filter_func=train_filter_func,
                                            examples_per_seq=examples_per_seq)

    all_view_pair_names = dataloader_train.dataset.get_all_comm_view_pair_names()
    all_view_pair_frame_lengths = dataloader_train.dataset.frame_lengths

    # for every task one label based on video name
    # not used to train the models
    transform_comm_name, num_domain_task_classes, task_names = val_fit_task_label(vid_name_to_task, all_view_pair_names)
    log.info('task names: {}'.format(task_names))

    # func to transform video name to a task label
    label_funcs = {'domain task label': transform_comm_name}
    num_domain_frames = args.num_domain_frames

    # embedding class
    log.info('num_domain_frames: {}'.format(num_domain_frames))

    # Discriminator network with iputs outputs depending on the args settings
    net_input = args.emb_dim * num_domain_frames
    d_net = Discriminator(net_input, H=args.d_net_hidden_dim, z_dim=args.d_net_z_dim, d_out=[num_domain_task_classes])

    # DATA domain
    # filter out fake examples and tasks for D net
    stride = args.multi_domain_frames_stride
    if args.train_filter_tasks is not None:
        def filter_func_domain(name, frames_cnt):
            """ return no fake examples for filtered tasks"""
            return "fake" not in name and all(task not in name for task in train_filter_tasks)
    else:
        filter_func_domain = None

        def filter_func_domain(name, frames_cnt):
            """ return no fake exmaples for filtered tasks"""
            return "fake" not in name

    dataloader_train_domain = get_skill_dataloader(args.train_dir,
                                                   args.num_views,
                                                   args.batch_size,
                                                   use_cuda,
                                                   img_size=299,
                                                   filter_func=filter_func_domain,
                                                   label_funcs=label_funcs,
                                                   num_domain_frames=num_domain_frames,
                                                   stride=stride)
    if use_cuda:
        torch.cuda.seed()
        criterion.cuda()
        asn.cuda()
        d_net.cuda()

    model_forward_cuda = functools.partial(model_forward, mdl=asn, use_cuda=use_cuda, to_numpy=False)
    model_forward_np = functools.partial(model_forward, mdl=asn, use_cuda=use_cuda, to_numpy=True)

    # define optimizer for encoder (g) and Discriminator (d)
    params_asn = filter(lambda p: p.requires_grad, asn.parameters())
    optimizer_g = optim.Adam(params_asn, lr=args.lr_d)
    optimizer_d = optim.Adam(d_net.parameters(), lr=args.lr_g)

    assert isinstance(criterion, (LiftedStruct, LiftedCombined))
    key_views = ["frames views {}".format(i) for i in range(args.num_views)]
    iter_metric = iter(data_loader_cycle(dataloader_train))
    iter_domain = iter(data_loader_cycle(dataloader_train_domain))
    loss_val_min = None
    loss_val_min_step = 0

    for global_step in range(global_step_start, args.steps):

        # =======================================================
        # update the encoder network
        sample_batched = next(iter_metric)
        # metric loss
        img = torch.cat([sample_batched[key_views[0]],
                         sample_batched[key_views[1]]])
        embeddings = model_forward_cuda(Variable(img))
        n = sample_batched[key_views[0]].size(0)
        anchor_emb, positive_emb = embeddings[:n], embeddings[n:]
        label_positive_pair = np.arange(n)
        labels = Variable(torch.Tensor(np.concatenate([label_positive_pair, label_positive_pair]))).cuda()

        # METRIC loss
        if examples_per_seq == 1:
            loss_metric = criterion(embeddings, labels)
        else:
            loss_metric = multi_vid_batch_loss(criterion, embeddings, labels,
                                               num_vid_example=examples_per_seq)

        # set input and targets
        sample_batched_domain = next(iter_domain)

        img_domain = torch.cat([sample_batched_domain[key_views[0]],
                                sample_batched_domain[key_views[1]]])
        emb_asn = model_forward_cuda(Variable(img_domain))

        if num_domain_frames != 1:
            # multiple frames as skills
            bl = emb_asn.size(0)
            emb_size = emb_asn.size(1)
            emb_asn = emb_asn.view(bl // num_domain_frames, num_domain_frames * emb_size)
            # mask out label for cat view

        kl_loss, d_out_gen = d_net(emb_asn)
        d_out_gen = d_out_gen[0]

        # min the entropy for different classes
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # ensure equal usage of fake samples
        loss_g = loss_metric * 0.1

        # maximize the entropy
        entropy_fake = entropy(d_out_gen)
        entropy_fake.backward(retain_graph=True)
        entropy_margin = -1. * marginalized_entropy(d_out_gen)
        # ensure equal usage of fake samples
        entropy_margin.backward(retain_graph=True)

        # update the encoder network
        loss_g.backward(retain_graph=True)
        optimizer_g.step()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # =======================================================
        # update the Discriminator

        # maximize marginalized entropy over real samples to ensure equal usage
        entropy_margin = -1. * marginalized_entropy(d_out_gen)
        entropy_margin.backward(retain_graph=True)
        # minimize entropy to make certain prediction of real sample
        entropy_real = -1. * entropy(d_out_gen)
        entropy_real.backward(retain_graph=True)
        kl_loss.backward()
        optimizer_d.step()

        if global_step % 100 == 0 or global_step == 1:
            # log training
            loss_metric = loss_g.data.cpu().numpy().item()
            mi = get_metric_info_multi_example(anchor_emb.data.cpu().numpy(), positive_emb.data.cpu().numpy())
            log_train(writer, mi, loss_metric, criterion, entropy_fake, global_step)

        # =======================================================
        # Validation
        if global_step % args.val_step == 0 and global_step > global_step_start:
            log.info("==============================")
            asn.eval()

            if args.plot_tsne and global_step % 20000 == 0:
                # save a tsne plot
                visualize_embeddings(model_forward_cuda, dataloader_val, summary_writer=None,
                                     global_step=global_step, save_dir=args.save_folder,
                                     label_func=vid_name_to_task)
            loss_val, nn_dist, dist_view_pais, frame_distribution_err_cnt = view_pair_alignment_loss(model_forward_np,
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
