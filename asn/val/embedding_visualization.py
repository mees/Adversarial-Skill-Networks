"""
t sne plot with sklearn and tensorboardX
"""

import argparse
import functools
import itertools
import os
from collections import defaultdict
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import offsetbox
import time
import torch  # before cv2
import cv2
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from torch import multiprocessing
from torch.autograd import Variable
from asn.utils.img import flip_imgs
from asn.utils.comm import get_other_view_files, sliding_window, create_dir_if_not_exists
from asn.utils.data import AligmentViewPairSampler, get_data_loader
from asn.utils.dataset import ViewPairDataset,get_video_csv_file
from asn.utils.frame_sequence import show_sequence
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from MulticoreTSNE import MulticoreTSNE as TSNE_multi
from collections import OrderedDict
from asn.utils.vid_montage import get_fourcc
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid-dir', type=str)
    parser.add_argument('--model', type=str,
                        default=None)
    parser.add_argument('--num-views', type=int, required=False, default=2)
    parser.add_argument('--use-cuda', type=int, required=False, default=1)
    parser.add_argument('--model-mode', type=str, default='asn',help="tcn asn or lstm")
    parser.add_argument('--batch-size', type=int, required=False, default=16)
    parser.add_argument('--seq-len', type=int, default=None,
                        help="len of the sequence, must betbatch-size \% rnn-seq-len ==0 ")
    parser.add_argument('--stride', type=int, default=None,
                        help="frame between sampeled seq. frame")
    return parser.parse_args()


def visualize_embeddings(func_model_forward, data_loader, summary_writer=None,
                         global_step=0, seq_len=None, stride=None, lable_func=None, save_dir=None, tag="",emb_size=32):
    """ visualize embeddings with tensorboardX

    Args:
        summary_writer(tensorbsoardX.SummaryWriter):
        data_loader(ViewPairDataset): with shuffle false
        lable_func: function to labe a frame: input is (vid_file_comm,frame_idx=None,vid_len=None,csv_file=None,state_lable=None)
    Returns:
        None

    """
    assert isinstance(data_loader.dataset,
                      ViewPairDataset), "dataset must be form type ViewPairDataset"
    data_len = len(data_loader.dataset)
    vid_dir = data_loader.dataset.vid_dir

    if seq_len:
        assert stride is not None
        # cut off first frames
        data_len -= seq_len*stride * len(data_loader.dataset.video_paths)
    embeddings = torch.empty((data_len, emb_size))
    img_size = 50  # image size to plot
    frames = torch.empty((data_len, 3, img_size, img_size))
    # trans form the image to plot it later
    trans = transforms.Compose([
        transforms.ToPILImage(),  # expects rgb, moves channel to front
        transforms.Resize(img_size),
        transforms.ToTensor(),  # imgae 0-255 to 0. - 1.0
    ])
    n_views = data_loader.dataset.n_views
    cnt_data = 0
    labels = []
    view_pair_name_labels = []
    labels_frame_idx = []
    vid_len_frame_idx = []
    view_pair_emb_seq = {}
    with tqdm(total=len(data_loader), desc='computing emgeddings for {} frames'.format(len(data_loader))) as pbar:
        for i, data in enumerate(data_loader):
            # compute the emb for a batch
            frames_batch = data['frame']
            if seq_len is None:
                emb = func_model_forward(frames_batch)
                # add emb to dict and to quue if all frames
                # for e, name, view, last in zip(emb, data["common name"], data["view"].numpy(), data['is last frame'].numpy()):
                # transform all frames to a smaler image to plt laster
                for e, frame in zip(emb, frames_batch):
                    embeddings[cnt_data] = e.cpu().detach()
                    # transform only for on img possilbe
                    frames[cnt_data] = trans(frame).cpu()
                    cnt_data += 1
                    if data_len == cnt_data:
                        break
                state_lable = data.get('state lable', None)
                comm_name = data['common name']
                frame_idx = data['frame index']
                vid_len = data['video len']
                labels_frame_idx.extend(frame_idx.numpy())
                vid_len_frame_idx.extend(vid_len.numpy())
                if lable_func is not None:
                    state_lable = len(
                        comm_name)*[None] if state_lable is None else state_lable
                    state_lable = [lable_func(c, i, v_len,get_video_csv_file(vid_dir,c),la)
                                   for c, la, i, v_len in zip(comm_name, state_lable, frame_idx, vid_len)]
                else:
                    state_lable=comm_name
                labels.extend(state_lable)
                view_pair_name_labels.extend(comm_name)
                if data_len == cnt_data:
                    break
            else:
                for i, (frame, name, view, last) in enumerate(zip(frames_batch, data["common name"], data["view"].numpy(), data['is last frame'].numpy())):
                    if name not in view_pair_emb_seq:
                        # empty lists for each view
                        # note: with [[]] * n_views the reference is same
                        view_pair_emb_seq[name] = {"frame": [[] for _ in range(n_views)],
                                                   "common name": [],
                                                   "frame index": [],
                                                   "video len": [],
                                                   "state lable": [],
                                                   "done": [False] * n_views}
                    view_pair_emb_seq[name]["frame"][view].append(
                        frame.view(1, *frame.size()))
                    view_pair_emb_seq[name]["common name"].append(name)
                    view_pair_emb_seq[name]["frame index"].append(data['frame index'][i].numpy())
                    view_pair_emb_seq[name]["video len"].append(data['video len'][i].numpy())

                    state_lable = data.get('state lable', None)
                    if state_lable is not None:
                        state_lable = state_lable[i]
                    view_pair_emb_seq[name]["state lable"].append(state_lable)
                    view_pair_emb_seq[name]["done"][view] = last

                    # compute embeds if all frames
                    if last:

                        # loop over all seq as batch
                        n = len(view_pair_emb_seq[name]["frame"][view])
                        frame_batch = torch.cat(
                            view_pair_emb_seq[name]["frame"][view])
                        for i, batch_seq in enumerate(sliding_window(frame_batch, seq_len, step=1, stride=stride, drop_last=True)):
                            if len(batch_seq) == seq_len:
                                index_data = i+seq_len*stride
                                if index_data >= len(view_pair_emb_seq[name]["common name"]):
                                    # log.error('drop frame expexted longer vid: index {} size vid {}'.format(
                                        # index_data, frame_batch.size()))
                                    continue
                                emb = func_model_forward(batch_seq).cpu().detach()
                                assert len(emb) == 1
                                frame = batch_seq[-1]  # last frame from seq in plt
                                frames[cnt_data] = trans(frame).cpu()
                                embeddings[cnt_data] = emb.cpu().detach()
                                cnt_data += 1

                                # add labels
                                comm_name = view_pair_emb_seq[name]['common name'][index_data]
                                frame_idx = view_pair_emb_seq[name]['frame index'][index_data]
                                state_lable = view_pair_emb_seq[name]['state lable'][index_data]
                                vid_len = view_pair_emb_seq[name]['video len'][index_data]
                                labels_frame_idx.append(np.asscalar(frame_idx))
                                vid_len_frame_idx.append(np.asscalar(vid_len))
                                if lable_func is not None:
                                    state_lable = lable_func(comm_name, state_lable, frame_idx)
                                labels.append(state_lable)

                                view_pair_name_labels.append(comm_name)
                            else:
                                log.warn("seq len to small")
                    if all(view_pair_emb_seq[name]["done"]):
                        view_pair_emb_name = view_pair_emb_seq.pop(name, None)
                        del view_pair_emb_name

            pbar.update(1)

    log.info('number of found labels: {}'.format(len(labels)))
    if len(labels) != len(embeddings):
        # in case of rnn seq cut cuff an the end, in case of drop last
        log.warn('number of labels {} smaller than embeddings, chaning embeddings size'.format(len(labels)))
        embeddings = embeddings[:len(labels)]
        frames = frames[:len(labels)]
    if summary_writer is not None:
        summary_writer.add_embedding(embeddings,  # expext torch tensor
                                     label_img=frames,
                                     global_step=global_step,
                                     metadata=labels)
    if len(labels) == 0:
        labels = None
    else:
        log.info('start TSNE fit')
        labels = labels[:data_len]
        metadata = [[s]for s in labels]
        embeddings = embeddings.numpy()
        imgs = flip_imgs(frames.numpy(), rgb_to_front=False)
        s_time = time.time()
        rnn_tag = "_seq{}_stride{}".format(seq_len, stride)if seq_len is not None else ""
        X_tsne = TSNE_multi(n_jobs=4, perplexity=40).fit_transform(
	    embeddings)  # perplexity = 40, theta=0.5
        # create_time_vid(X_tsne,labels_frame_idx,vid_len_frame_idx)
        plot_embedding(X_tsne, labels, title=tag+"multi-t-sne_perplexity40_theta0.5_step" +
              str(global_step)+rnn_tag, imgs=imgs, save_dir=save_dir,
              frame_lable=labels_frame_idx, max_frame=vid_len_frame_idx,
              vid_lable=view_pair_name_labels)
        # log.info('mulit tsne s_time: {}'.format(time.time()-s_time))
    del embeddings, imgs, labels, X_tsne



def plt_lables_blow(ax, legend_handels):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(handles=legend_handels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)


def plt_labled_data(ax, X, labels_str, plt_cm=plt.cm.gist_ncar, lable_filter_legend=None, index_color_factor=None,hide_legend=False):
    assert X.shape[0] == len(labels_str), "plt X shape {}, string lable len {}".format(
        X.shape[0], len(labels_str))
    le = preprocessing.LabelEncoder()
    le.fit(labels_str)
    metadata_header = list(le.classes_)
    n_classes = float(len(metadata_header))
    y = le.transform(labels_str)
    if index_color_factor is None:
        colors = [plt_cm(y_i / float(n_classes))for y_i in y]
        # classes as color sector
    else:
        # color factor for index
        # norm for vid len
        colors = [plt_cm(y_i / float(c_n))
                  for y_i, c_n in zip(y, index_color_factor)]
        # factor
    legend_elements = ax.scatter(
        X[:, 0], X[:, 1], color=colors)
    # plt some points again to get lable hadles, added iteralbel labes to ax.scatter cant be filder
    #  and to handler
    # find ould legend elements if filtered
    filtered_leged = []

    def check_filter(l): return (
        lable_filter_legend is None or lable_filter_legend(l))
    for l in metadata_header:
        if check_filter(l) and l not in filtered_leged:
            filtered_leged.append(l)

    # get for each class one (if not filterd) legend handle
    legend_elements = {}
    if not hide_legend:
       for i in range(X.shape[0]):
           l = labels_str[i]
           if l not in legend_elements and check_filter(l):
               h = ax.scatter(X[i, 0], X[i, 1], color=plt_cm(
                   y[i] / float(n_classes)), label=l)
               legend_elements[l] = h
               if len(legend_elements) == len(filtered_leged):
               # all handels found
                   break
       # sort lables:
       legend_elements = OrderedDict(sorted(legend_elements.items()))
       # Shrink current axis's height by 10% on the bottom
       plt_lables_blow(ax, list(legend_elements.values()))

    return n_classes, y, colors, legend_elements


def plot_embedding(X, labels_str, title, imgs=None, save_dir=None, frame_lable=None, max_frame=None, vid_lable=None):

    # http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    if imgs is not None:
        fig = plt.figure(figsize=(20, 20))
        ax = plt.subplot(221)
    else:
        fig = plt.figure()
        ax = fig.gca()

    # lablels blow plt
    n_classes, y, colors, legend_elements = plt_labled_data(ax, X, labels_str)

    plt.title(title)
    if imgs is not None:
        # plt again but with image overlay
        ax = plt.subplot(222)
        ax.set_title("image overlay")
        ax.scatter(X[:, 0], X[:, 1], color=colors)
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 5e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(
                        imgs[i], cmap=plt.cm.gray_r, zoom=0.75),
                    X[i], pad=0.0)
                ax.add_artist(imagebox)

        # plt legend same as befor
        plt_lables_blow(ax, list(legend_elements.values()))

    if frame_lable is not None:
        # plt the frames classe
        # show color for ever 50 frame in legend
        ax = plt.subplot(223)
        plt_labled_data(ax, X, frame_lable,
                        lable_filter_legend=lambda l: l % 50 == 0,
                        plt_cm=plt.cm.Spectral, index_color_factor=max_frame)

        ax.set_title("frames as label (color range normalized for ervery vid)")
    if vid_lable is not None:
        # plt the view pair as classe
        ax = plt.subplot(224)
        plt_labled_data(ax, X, vid_lable,
                        lable_filter_legend=lambda x: False)

        ax.set_title("view pair as label")

    if save_dir is not None:
        create_dir_if_not_exists(save_dir)
        save_dir = os.path.expanduser(save_dir)
        title = os.path.join(save_dir, title)
    fig.savefig(title+".pdf", bbox_inches='tight')
    log.info('save TSNE plt to: {}'.format(title))
    plt.close('all')


def create_time_vid(X,frame_lable,max_frame, frame_num_step=100,plt_cm=plt.cm.Spectral):
        frame_steps=np.linspace(0, 1.0, num=frame_num_step)
	# add some frame of the full plot at the end of the vid
        # frame_steps=np.append(frame_steps,[1.]*20)
        frame_idx_norm = np.array([y_i / float(c_n)
                  for y_i, c_n in zip(frame_lable, max_frame)])
        print('frame_idx_norm: {}'.format(frame_idx_norm.shape))
        vid_name= "test.mp4"
        fps=10
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)
        xmin,xmax=np.min(X[:, 0]),np.max(X[:, 0])
        ymin,ymax=np.min(X[:, 1]),np.max(X[:, 1])
        vid_writer=None
        for f in frame_steps:
            fig=plt.figure(figsize=(5,5),facecolor=(0, 0, 0))
            # set tight margin
            left = 0.05
            bottom = 0.05
            width = 0.95
            height = 0.95
            ax = fig.add_axes([left, bottom, width, height])
            tmp_img="/tmp/plt_{}.png".format(f)
            # selcect vid frames with norm vid len
            dig_idx = frame_idx_norm<=f
            X_dig = X[dig_idx]
            frame_lable_dig = np.array(frame_lable)[dig_idx]
            max_frame_dig = np.array(max_frame)[dig_idx]
            plt_labled_data(ax, X_dig, frame_lable_dig,
                # lable_filter_legend=lambda l: l % 50 == 0,
                   plt_cm=plt_cm,
                   index_color_factor=max_frame_dig,hide_legend=True)
            # same axis for every frames
            plt.axis('off')
            off_s=0.02
            ax.set_xlim([xmin-off_s*(xmax-xmin),xmax+off_s*(xmax-xmin)])
            ax.set_ylim([ymin-off_s*(ymax-ymin),ymax+off_s*(ymax-ymin)])
            # plt.savefig(tmp_img,dpi = 200, facecolor=fig.get_facecolor())
            plt.savefig(tmp_img,dpi = 200)
            img=cv2.imread(tmp_img)
            if vid_writer is None:
                fourcc = get_fourcc(vid_name)
                # vid writer if shape isknows after rot
                vid_writer = cv2.VideoWriter(
                     vid_name, fourcc, fps, (img.shape[1],img.shape[0]))
            vid_writer.write(img)
            plt.close('all')
        for _ in range(20):
            vid_writer.write(img)
        vid_writer.release()

def main():
    args = get_args()
    from asn.train_utils import get_dataloader_val
    use_cuda = torch.cuda.is_available() and args.use_cuda
    if args.model_mode == "asn":
        from asn.model.asn import create_model
        asn_model, epoch, global_step, _, _ = create_model(
            use_cuda, args.model)

        def model_forward(frame_batch, to_numpy=False):
            if use_cuda:
                frame_batch = frame_batch.cuda()
            feat, emb, kl_loss = asn_model.forward(frame_batch)
            if to_numpy:
                return emb.data.cpu().numpy()
            else:
                return emb
    elif args.model_mode == "tcn":
        from asn.model.tcn import create_model
        tcn_model, epoch, global_step, _, _ = create_model(
            use_cuda, args.model)

        def model_forward(frame_batch):
            if use_cuda:
                frame_batch = frame_batch.cuda()
            return tcn_model.forward(frame_batch)  # normal tcn
    else:
        from asn.model.tcn import create_model, RNN
        rnn_type = RNN.TYPE_GRU

        tcn_model, epoch, global_step, _, _ = create_model(
            use_cuda, args.model, rnn_type=rnn_type, rnn_forward_seqarade=False)

        def model_forward(frame_batch):
            ''' input is one seq, return the hidden state at the end of the seq'''
            if use_cuda:
                frame_batch = frame_batch.cuda()
                # feed embeddins
            assert not tcn_model.rnn_forward_seqarade
            hidden = None
            if len(frame_batch) == args.seq_len:
                # only one seq as batch
                # use for eval
                for e in frame_batch:
                    _, hidden = tcn_model(e.view(1, *e.size()), hidden)
            else:
                raise ValueError()
            if rnn_type == RNN.TYPE_LSTM:
                ret = hidden[0]  # hidden state
            else:
                ret = hidden

            ret = ret.view(len(frame_batch) // args.seq_len,
                           ret.size(2))  # TODO
            assert len(ret) != 0
            # DEBUG:
            if len(frame_batch) == args.seq_len:
                # check to work with aligment loss
                assert len(ret) == 1
            assert len(ret) == len(frame_batch) // args.seq_len
            return ret


    tcn_model.eval()

    filter_func=None
    lable_func = None

    dataloader = get_dataloader_val(
        args.vid_dir, args.num_views, args.batch_size, use_cuda,
        filter_func=filter_func)

    visualize_embeddings(model_forward, dataloader, summary_writer=None,
                         global_step=global_step, tag="", lable_func=lable_func,
                         seq_len=args.seq_len, stride=args.stride)


if __name__ == "__main__":
    main()
