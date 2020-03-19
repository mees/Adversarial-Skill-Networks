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
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader 
from asn.loss.metric_learning import NpairLoss as NPairLoss 
from asn.loss.metric_learning import (LiftedStruct,LiftedCombined) 
from asn.utils.comm import create_dir_if_not_exists, data_loader_cycle, create_label_func 
from asn.utils.train_utils import get_metric_info_multi_example, log_train, multi_vid_batch_loss 
from asn.model.asn import create_model, save_model 
from asn.utils.train_utils import get_dataloader_val, get_dataloader_train 
from asn.utils.train_utils import get_train_transformer  


from asn.utils.dataset import DoubleViewPairDataset 
from asn.utils.log import log 
from asn.utils.train_utils import init_log_tb
from asn.utils.train_utils import grad_reverse, combi_push_stack_color_to_task, _fit_task_label
from asn.val.alignment import view_pair_alignment_loss 
from asn.val.classification_accuracy import accuracy, accuracy_batch 
from asn.val.embedding_visualization import visualize_embeddings 
from torchvision.utils import save_image
from torch.backends import cudnn
from asn.utils.sampler import RNNViewPairSequenceSampler
# For fast training
cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1500000)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--save-folder', type=str,
                        default='~/trained/asn/')
    parser.add_argument('--load-model', type=str, required=False)
    parser.add_argument('--task', type=str, default="cstack",help='dataset, load tasks for real block data (cstack) or ulm dataset:  cstack or uulm')
    parser.add_argument('--train-dir', type=str, default='~/data/train/')
    parser.add_argument('--val-dir-metric',
                        type=str, default='~/tcn_data/val')
    parser.add_argument('--val-dir-domain',
                        type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str,help="metric loss lifted or liftedcombi", default="lifted")
    parser.add_argument('--lr-start', type=float, default=0.0001)
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--no-tsne', action='store_true')
    parser.add_argument('--num-domain-frames', type=int, default=2)
    parser.add_argument('--multi-domain-frames-stride', type=int, default=15)
    parser.add_argument('--label-mode', default='task', type=str,help="use task id or task sector lable output for D.")
    parser.add_argument('--mode', type=str, default="cat-gan",help='cat-gan')
    parser.add_argument('--mode-dist', type=str, default="kl",help='DistDiscriminator mode, fc or kl, fc-sector, kl-sector')
    parser.add_argument('--mode-input', type=str, default="emb",help='input for the discriminator dist, emb, emb-combi, emb-combi-bot or freatures')
    parser.add_argument('--train-filter-tasks',
            help="task names to filter for trainig data, fromat, taskx,taskb. videos with the file name will be filtered", type=str, default=None)
    parser.add_argument('--num-example-batch',help="num example per batch each vid, only lifted loss support", type=int, default=4)
    return parser.parse_args()

class Discriminator(nn.Module):
    def __init__(self, D_in, H, D_out, grad_rev):
        super().__init__()
        self._rev_grad=grad_rev
        log.info('Discriminator domain net in_channels: {} out: {} hidden {}'.format(D_in,D_out,H))
        #self.gradient_reversal = GradientReversal()
        self._model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            # nn.Dropout2d(0.25),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            # nn.Dropout2d(0.25),
            torch.nn.ReLU(),
        )

        self.out_layer = nn.ModuleList()
        for out_n in D_out:
            self.out_layer.append(torch.nn.Linear(H, out_n))
        # NOTE nn.CrossEntropyLoss applies F.log_softmax and bot not  nn.NLLLos

    def forward(self, x):
        if self._rev_grad:
            x=grad_reverse(x)
        out = self._model(x)
        return [l(out) for l in self.out_layer]

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

class HLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b/x.size(0)
#marginalized entropy

def marginalized_entropy(y):
    # dtype = torch.FloatTensor
    # y1 = Variable(torch.randn(y.size(1)).type(dtype), requires_grad=True)
    # y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    y=F.softmax(y, dim=1)
    y1 = y.mean(0)
    y2 = -torch.sum(y1*torch.log(y1+1e-6))
    return y2

# entropy
def entropy(y):
    # dtype = torch.FloatTensor
    # y1 = Variable(torch.randn(y.size()).type(dtype), requires_grad=True)
    # y2 = Variable(torch.randn(1).type(dtype), requires_grad=True)
    # y1 = -F.softmax(y, dim=1)*torch.log_softmax(y+1e-6)
    y1 = F.softmax(y, dim=1) * F.log_softmax(y, dim=1)
    y2 = 1.0/y.size(0)*y1.sum()
    return y2


def get_domain_dataloader_train(dir_vids, num_views, batch_size, criterion,use_cuda,filter_func,lable_funcs,num_domain_frames=1,stride=1):
    # sampler with all rand frames from alle task
    transformer_train =get_train_transformer()
   # sample differt view
    transformed_dataset_train_domain = DoubleViewPairDataset(vid_dir=dir_vids,
                                                                 number_views=num_views,
                                                                 # std_similar_frame_margin_distribution=sim_frames,
                                                                 transform_frames=transformer_train,
                                                                 lable_funcs=lable_funcs,
                                                                 filter_func=filter_func)

    sampler=None
    drop_last=True
    log.info('transformed_dataset_train_domain len: {}'.format(len(transformed_dataset_train_domain)))
    if num_domain_frames >1:
        assert batch_size % num_domain_frames == 0,'wrong batch size for multi frames '
        sampler = RNNViewPairSequenceSampler(dataset=transformed_dataset_train_domain,
                                             stride=stride,
                                             allow_same_frames_in_seq=True,
                                             sequence_length=num_domain_frames,
                                             sequences_per_vid_in_batch=1,
                                             batch_size=batch_size)
        log.info('use multi frame dir {} len sampler: {}'.format(dir_vids,len(sampler)))
        drop_last=len(sampler)>=batch_size

     # random smaple vid
    dataloader_train_domain = DataLoader(transformed_dataset_train_domain,
                                         drop_last=drop_last,
                                         batch_size=batch_size,
                                         shuffle=True if sampler is None else False,
                                         num_workers=4,
                                         sampler=sampler,
                                         pin_memory=use_cuda)

    if sampler is not None and len(sampler)<=batch_size:
        log.warn("dataset sampler batch size")
    return dataloader_train_domain

if __name__ == '__main__':

    args = get_args()
    log.info("args: {}".format(args))
    writer = init_log_tb(args.save_folder)
    use_cuda = torch.cuda.is_available()
    criterion = {"lifted":LiftedStruct(),"liftedcombi":LiftedCombined()}[args.loss]
    log.info("criterion: for {} ".format(
        criterion.__class__.__name__))

    grad_rev= {'rev-grad':True}.get(args.mode,False)
    log.info('mode: {}'.format(args.mode))
    log.info('input mode: {}'.format(args.mode_input))
    log.info('grad_rev: {}'.format(grad_rev))
    if use_cuda:
        torch.cuda.seed()
        criterion.cuda()

    asn, start_epoch, start_step, _, _ = create_model(
        use_cuda, args.load_model)
    log.info('asn: {}'.format(asn.__class__.__name__))

    # var for train info
    loss_val_min = None
    loss_val_min_step = 0

    # load function which maps video file name to task for different datasets
    vid_name_to_task={"cstack":combi_push_stack_color_to_task}[args.task]
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
    num_secotrs=5
    if args.label_mode=='task':
        # for every task one lable
        transform_comm_name,num_domain_task_classes,task_names=_fit_task_label(vid_name_to_task,all_view_pair_names)
        print('task mode task_names: {}'.format(task_names))
        num_secotrs=None

    lable_funcs={'domain task lable':transform_comm_name}# add domain lables to the sample

    num_domain_frames=args.num_domain_frames
    log.info('num_domain_frames: {}'.format(num_domain_frames))

    # embedding class
    log.info('num_domain_frames: {}'.format(num_domain_frames))
    #in_channels = 1024*num_domain_frames  # out of SpatialSoftmax tcn
    if args.mode_input=='emb':
        in_channels = 32 # out of tcn
   

    # Discriminator network with iputs outputs depending on the args settings
    net_input=in_channels*num_domain_frames if args.mode_input !='dist' else in_channels
    if args.mode_dist=='kl':
        d_net = KlDiscriminator(net_input, H=500,z_dim=64,D_out= [num_domain_task_classes], grad_rev=grad_rev)

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

    dataloader_train_domain = get_domain_dataloader_train(args.train_dir,
                                                          args.num_views,
                                                          args.batch_size,
                                                          criterion,use_cuda,
                                                          filter_func=filter_func_domain,
                                                          lable_funcs=lable_funcs,
                                                          num_domain_frames=num_domain_frames,
                                                          stride=stride)
    if  args.val_dir_domain is not None:
        dataloader_val_domain = get_domain_dataloader_train(args.val_dir_domain,
                                                              args.num_views,
                                                              args.batch_size,
                                                              criterion,use_cuda,
                                                              filter_func=filter_func_domain,
                                                              lable_funcs=lable_funcs,
                                                              num_domain_frames=num_domain_frames,
                                                              stride=stride)

    global_step = start_step
    asn.train()
    asn.cuda()

    criterion_entro = HLoss()
    criterion_domain = nn.CrossEntropyLoss()
    log.info("criterion_domain: {} ".format(
        criterion_domain.__class__.__name__))
    if use_cuda:
        d_net.cuda()
        criterion_domain.cuda()
        criterion_entro.cuda()

    def model_forward(frame_batch, to_numpy=True):
        if use_cuda:
            frame_batch = frame_batch.cuda()
        feat, emb, kl_loss = asn.forward(frame_batch)
        if to_numpy:
            return emb.data.cpu().numpy()
        else:
            return emb, kl_loss

    def model_forward_feature_ex(frame_batch,to_numpy=False):
        if use_cuda:
            frame_batch = frame_batch.cuda()
        feature_ex, emb, kl_loss = asn.forward(frame_batch, only_feat=True)
        return feature_ex, kl_loss

    if args.mode_input in ['emb', 'emb-combi', 'emb-combi-dot','dist']:
        model_forward_domain=functools.partial(model_forward,to_numpy=False)
#    else:
#        model_forward_domain=model_forward_feature_ex

    def domain_val_forward(x, out_idx=0):
        emb_tcn,_=model_forward_domain(x)
        if num_domain_frames !=1:
            bl=emb_tcn.size(0)
            emb_size=emb_tcn.size(1)
            emb_tcn=emb_tcn.view(bl//num_domain_frames,num_domain_frames*emb_size)
        out = d_net(emb_tcn)
        if args.mode_dist in ['kl','kl-sector']:
            kl_loss, out=out
        return out[out_idx] if isinstance(out, (tuple, list)) else out

    def domain_val_forward_kl_sample(x, out_idx=0):
        # return sampled z to plt distributuion kl
        emb_tcn,_=model_forward_domain(x)
        if num_domain_frames !=1:
            bl=emb_tcn.size(0)
            emb_size=emb_tcn.size(1)
            emb_tcn=emb_tcn.view(bl//num_domain_frames,num_domain_frames*emb_size)
        _ = d_net(emb_tcn)
        return d_net.z

    params_tcn = filter(lambda p: p.requires_grad, asn.parameters())

    optimizer_g = optim.Adam(params_tcn, lr=args.lr_start)
    optimizer_d = optim.Adam(d_net.parameters(), lr=args.lr_start)

    assert isinstance(criterion, (LiftedStruct, LiftedCombined))
    epoch_start = time.time()
    # pouring task
    frame_distribution = {}  # no dist
    key_views = ["frames views {}".format(i) for i in range(args.num_views)]
    cnt_data_fps = 0
    iter_metric=iter(data_loader_cycle(dataloader_train))
    iter_domain=iter(data_loader_cycle(dataloader_train_domain))

    for epoch in range(start_epoch, args.steps):

        sample_indexes_anchor, sample_indexes_pos, sample_indexes_neg = [], [], []
        # =======================================================
        # update the tcn network: metrinc loss and max entro in d
        global_step += 1
        sample_batched=next(iter_metric)
        # metric loss
        img = torch.cat([sample_batched[key_views[0]],
                         sample_batched[key_views[1]]])
        embeddings, kl_loss_tcn_metric = model_forward(Variable(img), False)
        n = sample_batched[key_views[0]].size(0)
        anchor_emb, positive_emb = embeddings[:n], embeddings[n:]
        cnt_data_fps += embeddings.size(0)
        label_positive_pair = np.arange(n)
        labels = Variable(torch.Tensor(np.concatenate([label_positive_pair, label_positive_pair]))).cuda()
        if examples_per_seq ==1:
            loss_metric = criterion(embeddings, labels)
        elif global_step ==1:
            # loss and same debug input images
            loss_metric = multi_vid_batch_loss(criterion,embeddings, labels,args.num_views,
                                            num_vid_example=examples_per_seq,img_debug=img,save_folder=args.save_folder)
        else:
            loss_metric = multi_vid_batch_loss(criterion,embeddings, labels,args.num_views,
                                            num_vid_example=examples_per_seq)

        # set input and targets
        sample_batched_domain=next(iter_domain)
        lable_domain = sample_batched_domain['domain task lable']
        lable_domain = Variable(torch.cat([lable_domain, lable_domain])).cuda()
        
        img_domain = torch.cat([sample_batched_domain[key_views[0]],
                                 sample_batched_domain[key_views[1]]])
        emb_tcn, kl_loss_tcn_domain =model_forward_domain(Variable(img_domain))

        if num_domain_frames !=1:
            # multiple frames as input
            bl=emb_tcn.size(0)
            emb_size=emb_tcn.size(1)
            emb_tcn=emb_tcn.view(bl//num_domain_frames,num_domain_frames*emb_size)
            # mask out lable for cat view
            mask = torch.ByteTensor([i%num_domain_frames==0 for i in range(bl)]).cuda()
            
            lable_domain=lable_domain.masked_select(mask)
              
        if args.mode_dist=='kl':
            kl_loss, d_out_gen = d_net(emb_tcn)
            d_out_gen= d_out_gen[0]
            
        # min the entro for diffent classes
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        loss_entro = criterion_entro(d_out_gen)
        retain_graph=True
        
        if  args.mode=='cat-gan':
            #ensure equal usage of fake samples
            loss_g=loss_metric*0.1

            mone = torch.FloatTensor([-1]).cuda()
            one = torch.FloatTensor([1]).cuda()
            # max entro
            entropy1_fake = entropy(d_out_gen)
            entropy1_fake.backward(one, retain_graph=True)
            entorpy_margin = marginalized_entropy(d_out_gen)
            # #ensure equal usage of fake samples
            entorpy_margin.backward(mone, retain_graph=True)
            # retain_graph=False
        elif  args.mode=='normal-tcn':
            #ensure equal usage of fake samples
            loss_g=loss_metric
        if kl_loss_tcn_domain is not None:
            # kl loss for tcn model
            loss_g+=(kl_loss_tcn_domain+kl_loss_tcn_metric)*0.01
        # update the tcn gen network
        loss_g.backward(retain_graph=retain_graph)
        # torch.nn.utils.clip_grad_norm_(params_tcn, 5)
        optimizer_g.step()

        # update the Discriminator
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d=None
        if  args.mode=='cat-gan':
            one = torch.FloatTensor([1]).cuda()
            mone = torch.FloatTensor([-1]).cuda()
            # maximize marginalized entropy over real samples to ensure equal usage
            entorpy_margin = marginalized_entropy(d_out_gen)
            entorpy_margin.backward(mone,retain_graph=True)
            # minimize entropy to make certain prediction of real sample
            entropy1_real = entropy(d_out_gen)

            if args.mode_dist in ['kl','fc-sector','kl-sector']:
                entropy1_real.backward(mone,retain_graph=True)
            else:
                entropy1_real.backward(mone)

        if args.mode_dist in ['kl','kl-sector'] and  loss_d is not None:
            loss_d+=kl_loss
        elif args.mode_dist in ['kl','kl-sector']:
            kl_loss.backward()
        if loss_d is not None:
            loss_d.backward()

        # torch.nn.utils.clip_grad_norm_(d_net.parameters(), 5)
        optimizer_d.step()

        # var to monitor the training
        if global_step ==1:
            create_dir_if_not_exists(os.path.join(args.save_folder,"images/"))
            save_image(sample_batched_domain[key_views[0]], os.path.join(args.save_folder,"images/tcn_view0_domain.png"))
            save_image(sample_batched_domain[key_views[1]], os.path.join(args.save_folder,"images/tcn_view1_domain.png"))
            save_image(sample_batched[key_views[0]], os.path.join(args.save_folder,"images/tcn_view0.png"))
            save_image(sample_batched[key_views[1]], os.path.join(args.save_folder,"images/tcn_veiw1.png"))
            save_image(img, os.path.join(args.save_folder,"images/all.png"))

        if global_step % 500 == 0 or global_step == 1:
            # log train dist
            loss_metric = np.asscalar(loss_g.data.cpu().numpy())
            mi=get_metric_info_multi_example(anchor_emb.data.cpu().numpy(),positive_emb.data.cpu().numpy(),1)
            # log training info

            fps = cnt_data_fps / (time.time() - epoch_start)
            cnt_data_fps = 0


            # log training info
            fps = cnt_data_fps / (time.time() - epoch_start)
            acc=accuracy_batch(d_out_gen ,lable_domain)[0]
            mean_pred=F.softmax(d_out_gen, dim=1).mean(0).data.cpu().numpy()
            d_pred=F.softmax(d_out_gen, dim=1).data.cpu().numpy()

            log_train(writer,mi,loss_metric,criterion,global_step,fps,1)
            msg = "steps {}, domain acc {:.2}, mean out d {}".format( global_step , acc,mean_pred)
            log.info(msg)
            print('msg: {}'.format(msg))
            writer.add_scalar('train/loss_entro',
                              loss_entro, global_step)
            writer.add_scalar('train/accuracy_batch',
                              acc,  global_step)
            epoch_start = time.time()

        if global_step % 5000== 0:
            # valGidation
            log.info("==============================")
            asn.eval()
            d_net.eval()

            if args.val_dir_domain is not None:
                acc,loss_val_domain=accuracy(dataloader_val_domain,domain_val_forward,criterion_domain,
                                             key_views,
                                             ['domain task lable']*2,# key for inputs and targets
                                             writer=writer,step_writer=global_step,
                                             task_names=task_names,
                                             plot_name='domain_acc')
                writer.add_scalar('validation/accuracy_domain', acc, global_step)
                writer.add_scalar('validation/loss_val_domain', loss_val_domain, global_step)
                log.info('val domain acc: {}'.format(acc))

            if not args.no_tsne and global_step % 20000 == 0:
                def emb_only(d):
                    return model_forward(d,to_numpy=False)[0]
                visualize_embeddings(emb_only, dataloader_val, summary_writer=None,
                                     global_step=global_step, save_dir=args.save_folder, lable_func=vid_name_to_task)
                if args.mode_dist in ['kl','kl-sector']:
                    # tsn plt sample dnet gaussian
                    visualize_embeddings(domain_val_forward_kl_sample, dataloader_val, writer,
                                         global_step=global_step, save_dir=args.save_folder,
                                         seq_len=num_domain_frames, stride=stride,emb_size=d_net.z_dim)
            loss_val, nn_dist, dist_view_pais, frame_distribution_err_cnt = view_pair_alignment_loss(model_forward,
                                                                                              args.num_views,
                                                                                              dataloader_val)
            asn.train()
            d_net.train()

            writer.add_histogram("validation/frame_error_count",
                                 np.array(frame_distribution_err_cnt), global_step)
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

