import argparse
import os

import cv2
import numpy as np
import sklearn.utils

from asn.utils.comm import get_view_pair_vid_files
from asn.utils.img import convert_to_uint8, montage
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-imgs', type=str,
                        help="directory of file separated with \",\" like vid1.mov,vid2.mov",
                        default="~/data/multiview-pouring/videos/train-small")
    parser.add_argument('--output-img-name', type=str,
                        default="frame_sequence_out.png")
    parser.add_argument(
        '--img-height', help="vid output height", type=int, default=100)
    parser.add_argument(
        '--num-views', help="number of views", type=int, default=2)
    parser.add_argument(
        '--img-width', help="vid output width", type=int, default=100)
    parser.add_argument(
        '--n-frame', help="erver n frame", type=int, default=4)
    parser.add_argument(
        '--max-vid-num', help="maximum of view pair videos to display,-1 to show all", type=int, default=4)

    return parser.parse_args()


def show_sequence(imgs, delay=0, n_frame=1, save_name=None, to_rgb=True):
    """ shows a 2d imgs array like [[img1,img2], with frame counter as titles """
    if n_frame != 1:
        imgs = [i_v[::n_frame] for i_v in imgs]
    titles = [["frame {}".format(n * n_frame)
               for n in range(len(i))] for i in imgs]
    # take one the longest title row to show on top
    titles = [sorted(titles, key=len, reverse=True)[0]]
    montage_image = montage(imgs, titles=titles,
                            margin_separate_vertical=0,
                            margin_separate_horizontal=5)
    montage_image = convert_to_uint8(montage_image)
    if to_rgb:
        montage_image = cv2.cvtColor(montage_image, cv2.COLOR_RGB2BGR)
    if save_name is not None:
        cv2.imwrite(save_name, montage_image)
    cv2.imshow('sequence', montage_image)
    log.info('click image and then a key to continue')
    cv2.waitKey(delay)  # == 27:  # ESC
    cv2.destroyAllWindows()


def main():
    args = get_args()
    input_imgs_file = args.input_imgs
    dt = np.float32
    n_views = args.num_views
    image_size = (args.img_height, args.img_width)
    imgs, titles = [], []
    total_frames, n_frame = 0, args.n_frame
    if os.path.isdir(input_imgs_file):
        input_imgs_file = get_view_pair_vid_files(n_views, input_imgs_file)
        input_imgs_file_shuffle = sklearn.utils.shuffle(input_imgs_file)
        input_imgs_file = []
        for f in input_imgs_file_shuffle:
            input_imgs_file.extend(f)
    else:
        input_imgs_file = input_imgs_file.split(",")
    # only show less the max img
    if args.max_vid_num != -1 and args.max_vid_num < len(input_imgs_file):
        input_imgs_file = input_imgs_file[:args.max_vid_num]
    assert len(input_imgs_file), "no vids found"
    # read vids
    for f in input_imgs_file:
        imgs_vid = VideoFrameSampler(
            f, dtype=dt, resize_shape=image_size, to_rgb=False).get_all()
        total_frames = len(imgs_vid)
        log.info("file {} with frames: {}".format(f, total_frames))
        imgs.append(imgs_vid)

    show_sequence(imgs, delay=0, n_frame=n_frame,
                  save_name=args.output_img_name, to_rgb=False)


if __name__ == '__main__':
    main()
