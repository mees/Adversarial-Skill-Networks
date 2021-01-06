"""
t sne plot with sklearn and tensorboardX
"""

import argparse
import os

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
from collections import OrderedDict

import cv2
from matplotlib import offsetbox
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE_multi
import numpy as np
from sklearn import preprocessing
import torch  # before cv2
from torchvision import transforms
from tqdm import tqdm

from asn.utils.comm import create_dir_if_not_exists
from asn.utils.dataset import get_video_csv_file, ViewPairDataset
from asn.utils.img import flip_imgs
from asn.utils.log import log
from asn.utils.vid_montage import get_fourcc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid-dir", type=str)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-views", type=int, required=False, default=2)
    parser.add_argument("--use-cuda", type=int, required=False, default=1)
    parser.add_argument("--model-mode", type=str, default="asn", help="tcn asn or lstm")
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--seq-len", type=int, default=None, help="seq len, must be batch-size % rnn-seq-len == 0")
    parser.add_argument("--stride", type=int, default=None, help="frame between sampled seq. frame")
    return parser.parse_args()


def visualize_embeddings(
    func_model_forward,
    data_loader,
    summary_writer=None,
    global_step=0,
    seq_len=None,
    stride=None,
    label_func=None,
    save_dir=None,
    tag="",
    emb_size=32,
):
    """visualize embeddings with tensorboardX

    Args:
        summary_writer(tensorboardX.SummaryWriter):
        data_loader(ViewPairDataset): with shuffle false
        label_func: function to label a frame: input is (vid_file_comm,frame_idx=None,vid_len=None,csv_file=None,state_label=None)
    Returns:
        None
        :param func_model_forward:
        :param global_step:
        :param seq_len:
        :param stride:
        :param save_dir:
        :param tag:
        :param emb_size:

    """
    assert isinstance(data_loader.dataset, ViewPairDataset), "dataset must be form type ViewPairDataset"
    data_len = len(data_loader.dataset)
    vid_dir = data_loader.dataset.vid_dir

    if seq_len:
        assert stride is not None
        # cut off first frames
        data_len -= seq_len * stride * len(data_loader.dataset.video_paths)
    embeddings = np.empty((data_len, emb_size))
    img_size = 50  # image size to plot
    frames = torch.empty((data_len, 3, img_size, img_size))
    # trans form the image to plot it later
    trans = transforms.Compose(
        [
            transforms.ToPILImage(),  # expects rgb, moves channel to front
            transforms.Resize(img_size),
            transforms.ToTensor(),  # image 0-255 to 0. - 1.0
        ]
    )
    cnt_data = 0
    labels = []
    view_pair_name_labels = []
    labels_frame_idx = []
    vid_len_frame_idx = []
    with tqdm(total=len(data_loader), desc="computing embeddings for {} frames".format(len(data_loader))) as pbar:
        for i, data in enumerate(data_loader):
            # compute the emb for a batch
            frames_batch = data["frame"]
            if seq_len is None:
                emb = func_model_forward(frames_batch)
                # add emb to dict and to quue if all frames
                # for e, name, view, last in zip(emb, data["common name"], data["view"].numpy(), data['is last frame'].numpy()):
                # transform all frames to a smaller image to plt later
                for e, frame in zip(emb, frames_batch):
                    embeddings[cnt_data] = e
                    # transform only for on img possible
                    frames[cnt_data] = trans(frame).cpu()
                    cnt_data += 1
                    if data_len == cnt_data:
                        break
                state_label = data.get("state lable", None)
                comm_name = data["common name"]
                frame_idx = data["frame index"]
                vid_len = data["video len"]
                labels_frame_idx.extend(frame_idx.numpy())
                vid_len_frame_idx.extend(vid_len.numpy())
                if label_func is not None:
                    state_label = len(comm_name) * [None] if state_label is None else state_label
                    state_label = [
                        label_func(c, i, v_len, get_video_csv_file(vid_dir, c), la)
                        for c, la, i, v_len in zip(comm_name, state_label, frame_idx, vid_len)
                    ]
                else:
                    state_label = comm_name
                labels.extend(state_label)
                view_pair_name_labels.extend(comm_name)
                if data_len == cnt_data:
                    break
            else:
                raise NotImplementedError()

            pbar.update(1)

    log.info("number of found labels: {}".format(len(labels)))
    if len(labels) != len(embeddings):
        # in case of rnn seq cut cuff an the end, in case of drop last
        log.warn("number of labels {} smaller than embeddings, changing embeddings size".format(len(labels)))
        embeddings = embeddings[: len(labels)]
        frames = frames[: len(labels)]
    if len(labels) == 0:
        labels = None
    else:
        log.info("start TSNE fit")
        labels = labels[:data_len]
        imgs = flip_imgs(frames.numpy(), rgb_to_front=False)
        rnn_tag = "_seq{}_stride{}".format(seq_len, stride) if seq_len is not None else ""
        X_tsne = TSNE_multi(n_jobs=4, perplexity=40).fit_transform(embeddings)  # perplexity = 40, theta=0.5
        create_time_vid(X_tsne, labels_frame_idx, vid_len_frame_idx)
        plot_embedding(
            X_tsne,
            labels,
            title=tag + "multi-t-sne_perplexity40_theta0.5_step" + str(global_step) + rnn_tag,
            imgs=imgs,
            save_dir=save_dir,
            frame_lable=labels_frame_idx,
            max_frame=vid_len_frame_idx,
            vid_lable=view_pair_name_labels,
        )

def plt_labels_blow(ax, legend_handels):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(
        handles=legend_handels, loc="upper center", bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5
    )


def plt_labeled_data(
    ax, X, labels_str, plt_cm=plt.cm.gist_ncar, label_filter_legend=None, index_color_factor=None, hide_legend=False
):
    assert X.shape[0] == len(labels_str), "plt X shape {}, string lable len {}".format(X.shape[0], len(labels_str))
    le = preprocessing.LabelEncoder()
    le.fit(labels_str)
    metadata_header = list(le.classes_)
    n_classes = float(len(metadata_header))
    y = le.transform(labels_str)
    if index_color_factor is None:
        colors = [plt_cm(y_i / float(n_classes)) for y_i in y]
        # classes as color sector
    else:
        # color factor for index
        # norm for vid len
        colors = [plt_cm(y_i / float(c_n)) for y_i, c_n in zip(y, index_color_factor)]
        # factor
    scatter = ax.scatter(X[:, 0], X[:, 1], color=colors)
    filtered_legend = []

    def check_filter(l):
        return label_filter_legend is None or label_filter_legend(l)

    for l in metadata_header:
        if check_filter(l) and l not in filtered_legend:
            filtered_legend.append(l)

    # get for each class one (if not filtered) legend handle
    legend_elements = {}
    if not hide_legend:
        for i in range(X.shape[0]):
            l = labels_str[i]
            if l not in legend_elements and check_filter(l):
                h = ax.scatter(X[i, 0], X[i, 1], color=plt_cm(y[i] / float(n_classes)), label=l)
                legend_elements[l] = h
                if len(legend_elements) == len(filtered_legend):
                    # all handles found
                    break
        # sort labels:
        legend_elements = OrderedDict(sorted(legend_elements.items()))
        # Shrink current axis's height by 10% on the bottom
        plt_labels_blow(ax, list(legend_elements.values()))

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

    # labels blow plt
    n_classes, y, colors, legend_elements = plt_labeled_data(ax, X, labels_str)

    plt.title(title)
    if imgs is not None:
        # plt again but with image overlay
        ax = plt.subplot(222)
        ax.set_title("image overlay")
        ax.scatter(X[:, 0], X[:, 1], color=colors)
        if hasattr(offsetbox, "AnnotationBbox"):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1.0, 1.0]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 5e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(imgs[i], cmap=plt.cm.gray_r, zoom=0.75), X[i], pad=0.0
                )
                ax.add_artist(imagebox)

        # plt legend same as befor
        plt_labels_blow(ax, list(legend_elements.values()))

    if frame_lable is not None:
        # plt the frames classe
        # show color for ever 50 frame in legend
        ax = plt.subplot(223)
        plt_labeled_data(
            ax,
            X,
            frame_lable,
            label_filter_legend=lambda l: l % 50 == 0,
            plt_cm=plt.cm.Spectral,
            index_color_factor=max_frame,
        )

        ax.set_title("frames as label (color range normalized for every vid)")
    if vid_lable is not None:
        # plt the view pair as classe
        ax = plt.subplot(224)
        plt_labeled_data(ax, X, vid_lable, label_filter_legend=lambda x: False)

        ax.set_title("view pair as label")

    if save_dir is not None:
        create_dir_if_not_exists(save_dir)
        save_dir = os.path.expanduser(save_dir)
        title = os.path.join(save_dir, title)
    fig.savefig(title + ".pdf", bbox_inches="tight")
    log.info("save TSNE plt to: {}".format(title))
    plt.close("all")


def create_time_vid(X, frame_lable, max_frame, frame_num_step=100, plt_cm=plt.cm.Spectral):
    frame_steps = np.linspace(0, 1.0, num=frame_num_step)
    # add some frame of the full plot at the end of the vid
    # frame_steps=np.append(frame_steps,[1.]*20)
    frame_idx_norm = np.array([y_i / float(c_n) for y_i, c_n in zip(frame_lable, max_frame)])
    print("frame_idx_norm: {}".format(frame_idx_norm.shape))
    vid_name = "test.mp4"
    fps = 10
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    vid_writer = None
    for f in frame_steps:
        fig = plt.figure(figsize=(5, 5), facecolor=(0, 0, 0))
        # set tight margin
        left = 0.05
        bottom = 0.05
        width = 0.95
        height = 0.95
        ax = fig.add_axes([left, bottom, width, height])
        tmp_img = "/tmp/plt_{}.png".format(f)
        # selcect vid frames with norm vid len
        dig_idx = frame_idx_norm <= f
        X_dig = X[dig_idx]
        frame_lable_dig = np.array(frame_lable)[dig_idx]
        max_frame_dig = np.array(max_frame)[dig_idx]
        plt_labeled_data(
            ax,
            X_dig,
            frame_lable_dig,
            # lable_filter_legend=lambda l: l % 50 == 0,
            plt_cm=plt_cm,
            index_color_factor=max_frame_dig,
            hide_legend=True,
        )
        # same axis for every frames
        plt.axis("off")
        off_s = 0.02
        ax.set_xlim([xmin - off_s * (xmax - xmin), xmax + off_s * (xmax - xmin)])
        ax.set_ylim([ymin - off_s * (ymax - ymin), ymax + off_s * (ymax - ymin)])
        # plt.savefig(tmp_img,dpi = 200, facecolor=fig.get_facecolor())
        plt.savefig(tmp_img, dpi=200)
        img = cv2.imread(tmp_img)
        if vid_writer is None:
            fourcc = get_fourcc(vid_name)
            # vid writer if shape isknows after rot
            vid_writer = cv2.VideoWriter(vid_name, fourcc, fps, (img.shape[1], img.shape[0]))
        vid_writer.write(img)
        plt.close("all")
    for _ in range(20):
        vid_writer.write(img)
    vid_writer.release()


def main():
    args = get_args()
    from asn.train_utils import get_dataloader_val

    use_cuda = torch.cuda.is_available() and args.use_cuda
    if args.model_mode == "asn":
        from asn.model.asn import create_model

        asn_model, epoch, global_step, _ = create_model(use_cuda, args.model)

        def model_forward(frame_batch, to_numpy=False):
            if use_cuda:
                frame_batch = frame_batch.cuda()
            feat, emb, kl_loss = asn_model.forward(frame_batch)
            if to_numpy:
                return emb.data.cpu().numpy()
            else:
                return emb

    filter_func = None
    label_func = None

    dataloader = get_dataloader_val(args.vid_dir, args.num_views, args.batch_size, use_cuda, filter_func=filter_func)

    visualize_embeddings(
        model_forward,
        dataloader,
        summary_writer=None,
        global_step=global_step,
        tag="",
        label_func=label_func,
        seq_len=args.seq_len,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
