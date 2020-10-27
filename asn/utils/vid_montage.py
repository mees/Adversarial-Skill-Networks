import argparse
import os
from multiprocessing import Process

import cv2
import numpy as np
import sklearn.utils
from torch import multiprocessing
from tqdm import tqdm

from asn.utils.comm import get_view_pair_vid_files
from asn.utils.img import (convert_to_uint8, montage, np_shape_to_cv)
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler

'''
create a grid of vids

usage:
python utils/vid_montage.py  \
 --input-imgs ~/data/multiview-pouring/torch-data/train/ \
 --img-height 100 --img-width 100

'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-imgs', type=str,
                        help="directory of file separated with \",\" like vid1.mov,vid2.mov",
                        default="~/tcn_data/can_stacking_3_V6/videos/train")
    parser.add_argument('--output-vid-name', type=str,
                        default="out1.mov")
    parser.add_argument(
        '--mun-col', help="vids in x", type=int, default=4)
    parser.add_argument(
        '--mun-row', help="vids in y", type=int, default=5)
    parser.add_argument(
        '--view-idx', help="view to show", type=int, default=0)
    parser.add_argument(
        '--num-views', help="number of views", type=int, default=2)
    parser.add_argument(
        '--img-height', help="vid output height", type=int, default=150)
    parser.add_argument(
        '--img-width', help="vid output width", type=int, default=150)
    parser.add_argument(
        '--num-frames', help="num of frames", type=int, default=500)
    parser.add_argument(
        '--fps', help="out vid fps", type=int, default=24)
    parser.add_argument(
        '--max-vid-num', help="maximum of view pair videos to display,-1 to show all", type=int, default=4)

    return parser.parse_args()


def get_frames(input_vid_file, n_process, num_frames, image_size):
    """ yield n frames from differt viedeos by n processes,
        runs untill all video frames are sampeled, other frames are black
    """
    print("number of vid files:", len(input_vid_file))
    assert len(input_vid_file) >= n_process
    process_frames = []
    input_frames_q = multiprocessing.Queue()
    for f in input_vid_file:
        input_frames_q.put(f)
    result_frames_qs = []

    def get_frame_worker(p_ranke, num_frames, image_size, input_frames_q, result_frames_q):
        vid = None
        sample_fames_vid = 0
        dt = np.float32
        for frame_i in range(num_frames):
            if vid is None or sample_fames_vid >= len(vid):
                if input_frames_q.empty():
                    vid = None
                    print("# WARNING: no more vids, ranke", p_ranke)
                else:
                    f = input_frames_q.get()
                    vid = VideoFrameSampler(f, dtype=dt, resize_shape=image_size, to_rgb=False)
                sample_fames_vid = 0
            if vid is None:
                # return empty if no more fames to get
                frame = np.zeros(image_size + (3,), dtype=dt)
            else:
                frame = vid.get_frame(sample_fames_vid)
            sample_fames_vid += 1
            result_frames_q.put(frame)

    for p_ranke in range(n_process):
        result_frames_q = multiprocessing.Queue(1)
        result_frames_qs.append(result_frames_q)
        args_worker = (p_ranke, num_frames,
                       image_size, input_frames_q, result_frames_q)
        p = Process(target=get_frame_worker, args=args_worker)
        p.daemon = True
        process_frames.append(p)
        p.start()

    for frame_i in range(num_frames):
        # get frames for process
        fames = [q_res.get() for q_res in result_frames_qs]
        yield fames

    for p in process_frames:
        p.join()


def get_fourcc(file_name):
    """ input file_name or extension """
    assert "." in file_name
    if file_name.strip()[0] == ".":
        file_extension_out = file_name
    else:
        _, file_extension_out = os.path.splitext(file_name)
    ex_to_fourcc = {'.mov': 'jpeg', '.mp4': 'MP4V', '.avi': "MJPG"}
    file_extension_out = file_extension_out.strip().lower()
    fourcc = ex_to_fourcc.get(file_extension_out, None)
    if fourcc is not None:
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
    else:
        raise ValueError("file type not supported, use: {}"
                         .format(", ".join(ex_to_fourcc.keys())))
    return fourcc


def main():
    args = get_args()
    input_imgs_file = os.path.expanduser(args.input_imgs)
    dt = np.float32
    n_views = args.num_views
    image_size = (args.img_height, args.img_width)
    assert os.path.isdir(input_imgs_file), "input not a dir"
    input_imgs_file = get_view_pair_vid_files(n_views, input_imgs_file)
    assert len(input_imgs_file), "no vids found"
    view_pair_idx = args.view_idx
    input_imgs_file = [view_piar_vid[view_pair_idx] for view_piar_vid in input_imgs_file]
    input_imgs_file = sklearn.utils.shuffle(input_imgs_file)
    ex_to_fourcc = {'.mov': 'jpeg', '.mp4': 'MP4V', '.avi': "MJPG"}
    fourcc = get_fourcc(args.output_vid_name)
    log.info("output vid: {}".format(args.output_vid_name))
    fps = args.fps
    vid_writer = None
    for frames in tqdm(get_frames(input_imgs_file, args.mun_col * args.mun_row, args.num_frames, image_size),
                       desc="frame", total=args.num_frames):

        imgs = [[frames[y] for y in range(x, x + args.mun_row)]
                for x in range(0, len(frames), args.mun_row)]

        margin = 2
        montage_image = montage(imgs,
                                margin_color_bgr=[0, 0, 0],
                                margin_top=margin, margin_bottom=margin,
                                margin_left=margin, margin_right=margin,
                                margin_separate_vertical=margin,
                                margin_separate_horizontal=margin)
        montage_image = convert_to_uint8(montage_image)
        if vid_writer is None:
            # vid writer if shape isknows after rot
            out_shape_cv = np_shape_to_cv(montage_image.shape[:2])
            vid_writer = cv2.VideoWriter(
                args.output_vid_name, fourcc, fps, out_shape_cv)
        vid_writer.write(montage_image)
    vid_writer.set(cv2.CAP_PROP_FRAME_COUNT, args.num_frames)
    vid_writer.set(cv2.CAP_PROP_FPS, fps)
    vid_writer.release()


if __name__ == '__main__':
    main()
