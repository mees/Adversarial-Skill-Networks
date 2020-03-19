import warnings
import json
import os
import os.path as op
import time

import numpy as np

import cv2
import pandas
import torch
from scipy.stats import norm
from torch import multiprocessing
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from asn.utils.comm import (get_files, get_other_view_files,
                                 get_view_pair_vid_files, split_view_file_name)
from asn.utils.img import convert_to_uint8, flip_img, flip_imgs
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler
from torchvision import transforms
from sklearn.utils import shuffle

def get_negative_frame_index(anchor_index, video_length, margin=0):
    ''' return frame range option excluding the postive frame  '''
    if anchor_index >= video_length:
        anchor_index = video_length
    lower_bound = 0
    upper_bound = max(0, anchor_index - margin)  # -1 -> not in range
    range1 = np.arange(lower_bound, upper_bound)
    lower_bound = min(anchor_index + margin + 1, video_length)
    upper_bound = video_length
    range2 = np.arange(lower_bound, upper_bound)
    range_option = np.concatenate([range1, range2])
    if margin == 0:
        assert len(range_option) - 1 == video_length
    return range_option


def get_positive_frame_index(anchor_index, video_length, margin=0):
    ''' return frame range option around the postive frame   '''
    anchor_index = int(anchor_index)
    if anchor_index >= video_length:
        # if the two view have a different frame length, take the last frame
        anchor_index = video_length
    lower_bound = max(0, anchor_index - margin)
    range1 = np.arange(lower_bound, min(anchor_index + 1, video_length))
    upper_bound = min(video_length - 1, anchor_index +
                      margin + 1)
    # anchor_index+1 because in range1
    range2 = np.arange(max(0, anchor_index + 1), upper_bound)
    range_option = np.concatenate([range1, range2])
    if margin == 0:
        assert len(range_option) == 1
    return range_option

def get_video_csv_file(video_dir,common_video_dir):
    ''' return a csv file for the video  '''
    csv_file = os.path.join(video_dir, common_video_dir+".csv")
    return csv_file if os.path.exists(csv_file) else None



def get_state_labled_frame(csv_file, frame, key='state'):
    ''' get state for a frame (row index is the frame index)'''
    # read with skipwro 0 frame to keep header
    assert frame>=0
    df = pandas.read_csv(csv_file,header=0,skiprows=lambda x: x not in [0,frame+1],nrows=2)
    assert len(df)
    assert df["frames"].iloc[-1]==frame
    return df[key].iloc[-1]


def get_camera_info(csv_file, frame, num_views):
    '''get camera infos TODO key'''
    info = {}
    df = pandas.read_csv(csv_file,header=0,skiprows=lambda x: x not in [0,frame+1],nrows=2)
    # mask = df['frames'] == frame
    # df = df.loc[mask]
    for view_i in range(num_views):
        info_keys = ["cam_pitch_view_{}".format(view_i),
                     "cam_yaw_view_{}".format(view_i),
                     "cam_distance_view_{}".format(view_i)]
        for ik in info_keys:
            info[ik] = df[ik].iloc[-1]
            assert df["frames"].iloc[-1]==frame
            # convert string to num lit
    # for k in df.keys():
    #     if isinstance(df[k], str) and "[" in df[k]:
    #         df[k] = json.loads(df[k])
    return info


def are_csv_files_in_dir(dir_f):
    return len(get_files(dir_f, file_types=".csv"))


def _get_all_comm_view_pair_names(video_paths):
    all_file_comm = []
    for views_vid_file in video_paths:
        view0_vid_file = views_vid_file[0]
        _, tail = os.path.split(view0_vid_file)
        vid_file_comm, _, _, _ = split_view_file_name(tail)
        assert vid_file_comm not in all_file_comm, "dulicate comman name found {}".format(
            vid_file_comm)
        all_file_comm.append(vid_file_comm)
    return all_file_comm


def _filter_view_pairs(video_paths_pair, frames_length_pair, filter_func):
    filtered_paths = []
    filtered_vid_len = []
    all_comm_names = _get_all_comm_view_pair_names(video_paths_pair)
    for vp, comm_name, len_views in zip(video_paths_pair, all_comm_names, frames_length_pair):
        if filter_func(str(comm_name), list(len_views)):
            assert len(vp) <2 or vp[0]!=vp[1], "view pair with same video: {}".format(vp)
            filtered_paths.append(vp)
            filtered_vid_len.append(len_views)
    if len(video_paths_pair) != len(filtered_paths):
        log.warn('dataset filterd videos form {} to {}'.format(
            len(video_paths_pair), len(filtered_paths)))
    else:
        log.warn("no videos filtered, but filter function is not not None")
    assert len(filtered_paths) > 0
    return filtered_paths, filtered_vid_len


def get_frame(vid_file, frame, use_image_if_exists):
    ''' load frames form vid file or load from images if exists (set in .csv) '''
    _, tail = os.path.split(vid_file)
    vid_file_comm, view_num_i, _, _ = split_view_file_name(tail)
    csv_file_dir=os.path.dirname(vid_file)
    csv_file = get_video_csv_file(csv_file_dir, vid_file_comm)
    if csv_file is not None:
        # read image file form csv
        try:
            key_image="image_file_view_{}".format(view_num_i)
            image_file=get_state_labled_frame(csv_file,frame, key=key_image)
            image_file=os.path.join(csv_file_dir,image_file)
            image_file=os.path.abspath(image_file)
            rgb_unit8=cv2.imread(image_file)[...,::-1]
            if rgb_unit8 is None:
                raise ValueError("file not found form csv {}".format(image_file))
            #load image file
            return vid_file_comm,rgb_unit8
        except KeyError:
            log.warn("no image key in csv for {}".format(vid_file))
    vid = VideoFrameSampler(vid_file)
    # warnings.warn("video file as input")
    return vid_file_comm, vid.get_frame(frame)


class DoubleViewPairDataset(Dataset):
    """multi view pair video dataset.
        dataset with mulitple views pairs video synced in time
        frames are provided was a view pair
    """

    def __init__(self, vid_dir, number_views, n_frame=1,
                 transform_frames=transforms.ToTensor(),
                 std_similar_frame_margin_distribution=None,
                 random_view_index=False,
                 use_img_if_exists=True,
                 filter_func=None, get_edges=False, add_camera_info=False,
                 lable_funcs={}):
        """
        Args:
            vid_dir (string): Directory with all the muliti view videos.
            number_views (int): number of views
            n_frame(int): number of sample to skip in sequence
            use_img_if_exists(bool): use image for the frames if set in csv file
            transform (callable, optional): Optional transform to be applied
                on a sample. if None no transformation
            std_similar_frame_margin_distribution: positive frame variable:
                postive frame ist sampled around the anchor frame with a normal distribution
            filter_func: function with two inputs: common name and list with frames number,
                        returns true if videp is ok
            dict lable_funcs: dict with tag in key and function to add a user labe,
                               inputs common video file name, frame, vid len, csv file (None in not exists)as input


        """
        self.transform_frames = transform_frames
        self.n_views = number_views
        self.lable_funcs = lable_funcs
        self.vid_dir = vid_dir
        self.n_frame = n_frame
        self.get_edges = get_edges
        self.get_camera_info = add_camera_info
        self._sample_counter = 0
        self.random_view_index=random_view_index
        self.std_similar_frame_margin_distribution = std_similar_frame_margin_distribution
        self.video_paths = get_view_pair_vid_files(
            self.n_views, self.vid_dir, join_path=True)

        self._count_frames()
        if filter_func is not None:
            # filter video based on input function
            self.video_paths, self.frame_lengths = _filter_view_pairs(
                self.video_paths, self.frame_lengths, filter_func)
        self._create_look_up()
        self.use_img_if_exists=use_img_if_exists
        self.print_frame_len_info()

    def __len__(self):
        return len(self.item_idx_lookup)

    def __getitem__(self, idx):
        self._sample_counter += 1
        view_pair_index, frame_index_anchor = self.item_idx_lookup[idx]
        samples = {}

        #flip rand view index
        view_index_shuffled =list(range(self.n_views))
        if self.random_view_index:
            view_index_shuffled = shuffle(view_index_shuffled)
        view_min_frame_len=min(self.frame_lengths[view_pair_index])
        for view_index in range(self.n_views):
            sample_frame_index = frame_index_anchor
            if self.std_similar_frame_margin_distribution:
                frames_length = self.frame_lengths[view_pair_index][view_index]
                # variable frame around frame anchor
                mean = frame_index_anchor
                std = self.std_similar_frame_margin_distribution
                # pdf as int
                similar_frame_index = norm.ppf(np.random.random(
                    1), loc=mean, scale=std).astype(int)[0]
                sample_frame_index = min(similar_frame_index, frames_length-1)
                sample_frame_index = max(0, sample_frame_index)
                sample_frame_index = sample_frame_index

            # get frame form vid
            vid_file = self.video_paths[view_pair_index][view_index]
            vid_file_comm, frame = get_frame(vid_file,sample_frame_index,self.use_img_if_exists)
            if view_index == 0:
                # add frame infos
                samples["common name"] = vid_file_comm
                samples["frame index"] = frame_index_anchor
                samples["vid len"] = view_min_frame_len
            if self.transform_frames:
                frame = self.transform_frames(frame)

            # is_last_frame_view.append(len(vid) - self.n_frame <= frame_index)
            view_index=view_index_shuffled[view_index]
            samples["frames views " + str(view_index)] = frame
            if self.get_edges:
                samples["frames edge views " + str(view_index)] = self._get_edges(frame)
        csv_file=get_video_csv_file(self.vid_dir, vid_file_comm)
        if self.get_camera_info and csv_file is not None:
            samples.update(get_camera_info(csv_file, view_index, self.n_views))
        self._update_user_lables(samples, vid_file_comm, frame_index_anchor, view_min_frame_len, csv_file)
        # "is last frame views": is_last_frame_view
        return samples

    def _update_user_lables(self,samples, vid_file_comm,frame,vid_len,csv_file):
        if self.lable_funcs is None:
            return
        for sample_key, get_lable_func in self.lable_funcs.items():
            assert not sample_key in samples
            samples[sample_key]=get_lable_func(vid_file_comm,frame,vid_len,csv_file)

    def _get_edges(self, torch_img):
        # TODO
        i = flip_img(torch_img.numpy(), rgb_to_front=False)
        img = np.zeros_like(i, dtype=np.uint8)
        img[:] = i*255.

        frame_edges = cv2.Canny(img, 100, 200).astype(dtype=np.float32)
        frame_edges = frame_edges.reshape(torch_img.size(1), torch_img.size(2), 1)
        frame_edges = np.concatenate(
            [frame_edges, frame_edges, frame_edges], axis=2)
        frame_edges = flip_img(frame_edges)
        return frame_edges

    def get_number_view_pairs(self):
        return len(self.video_paths)

    def get_all_comm_view_pair_names(self):
        return _get_all_comm_view_pair_names(self.video_paths)

    def _count_frames(self):
        self.frame_lengths = [[len(VideoFrameSampler(v_i))
                               for v_i in views]for views in self.video_paths]
        # assert len(self.frame_lengths[0]) ==self.n_views
    def print_frame_len_info(self):
        max_len_vid = max(max(l) for l in self.frame_lengths)
        min_len_vid = min(min(l) for l in self.frame_lengths)
        mean_len_vid = int(np.mean(self.frame_lengths))
        log.info("{} videos frame len mean : {}, min: {}, max: {}".format(
            self.vid_dir, mean_len_vid, min_len_vid, max_len_vid))
        # assert min_len_vid != 0, "dataset {} with video with no frames".format(
        #     self.video_paths)

    def _create_look_up(self):
        self.totlal_frame_lengths = 0
        self.item_idx_lookup = []  # idx -> view_pair_index and frame index
        self.idx_lookup = []  # view_pair_index and frame index -> idx
        for view_pair_index, view_lens in enumerate(self.frame_lengths):
            # take min frame len of vid pairs
            l = min(view_lens)
            frame_range = [i for i in range(0, l, self.n_frame)]
            self.totlal_frame_lengths += l
            self.idx_lookup.append(
                np.array(frame_range) + len(self.item_idx_lookup))
            for frame_index in frame_range:
                self.item_idx_lookup.append(
                    [view_pair_index, frame_index])

class ViewPairDataset(Dataset):
    """multi view pair video dataset.
        dataset with mulitple views pairs video synced in time
        frames are provided was separately
    """

    def __init__(self, vid_dir, number_views, n_frame=1,use_img_if_exists=True,
                 transform_frames=transforms.ToTensor(), filter_func=None):
        """
        Args:
            vid_dir (string): Directory with all the muliti view videos.
            number_views (int): number of views
            n_frame(int): number of sample to skip in sequence
            transform (callable, optional): Optional transform to be applied
                on a sample. if None no transformation
            filter_func: function to filter videos, input common view pair namen, and view pairs frame len list
        """
        self.transform_frames = transform_frames
        self.n_views = number_views
        self.vid_dir = vid_dir
        self.n_frame = n_frame
        self._sample_counter = 0
        self.video_paths = get_view_pair_vid_files(
            self.n_views, self.vid_dir, join_path=True)
        self._count_frames()
        self._use_labels = are_csv_files_in_dir(self.vid_dir)
        if filter_func is not None:
            # filter video based on input function
            self.video_paths, self.frame_lengths = _filter_view_pairs(
                self.video_paths, self.frame_lengths, filter_func)
        self._create_look_up()
        self._print_dataset_info_txt()
        self.use_img_if_exists=use_img_if_exists
        self.print_frame_len_info()


    def __len__(self):
        return len(self.item_idx_lookup)

    def __getitem__(self, idx):
        self._sample_counter += 1
        view_pair_index, view_index, frame_index = self.item_idx_lookup[idx]
        vid_file = self.video_paths[view_pair_index][view_index]

        _, tail = os.path.split(vid_file)
        vid_file_comm, frame = get_frame(vid_file,frame_index,self.use_img_if_exists)
        if self.transform_frames:
            frame = self.transform_frames(frame)
        vid_len=self.frame_lengths[view_pair_index][view_index]
        is_last_frame = vid_len- self.n_frame <= frame_index
        # print("idx",idx, "vid len",len(vid),"view",view_num_i,)
        # print("self._sample_counter",self._sample_counter)
        # print("self.totlal_frame_lengths",self.totlal_frame_lengths)
        samples = {"frame": frame,
                   "frame index": frame_index,
                   "video len": vid_len,
                   "common name": vid_file_comm,
                   "view": view_index,
                   "is last frame": is_last_frame}
        if self._use_labels:
            csv_file=get_video_csv_file(self.vid_dir, vid_file_comm)
            lable_state = get_state_labled_frame(csv_file, frame_index)
            samples['state lable'] = lable_state

        return samples

    def _print_dataset_info_txt(self):
        info_txt_file = os.path.join(self.vid_dir,"../../dataset_info.txt")
        if os.path.exists(info_txt_file):
            with open(info_txt_file, 'r') as f:
                log.info("dataset info:\n {}".format(f.read()))

    def _count_frames(self):
        self.frame_lengths = [[len(VideoFrameSampler(v_i))
                               for v_i in views]for views in self.video_paths]

    def print_frame_len_info(self):

        max_len_vid = max(max(l) for l in self.frame_lengths)
        min_len_vid = min(min(l) for l in self.frame_lengths)
        mean_len_vid = int(np.mean(self.frame_lengths))
        log.info("{} videos frame len mean : {}, min: {}, max: {}".format(
            self.vid_dir, mean_len_vid, min_len_vid, max_len_vid))

    def _create_look_up(self):
        self.totlal_frame_lengths = (np.sum(self.frame_lengths))
        self.item_idx_lookup = []
        for view_pair_index, view_lens in enumerate(self.frame_lengths):
            for view_index, l in enumerate(view_lens):
                frame_range = [i for i in range(0, l, self.n_frame)]
                for frame_index in frame_range:
                    self.item_idx_lookup.append(
                        [view_pair_index, view_index, frame_index])

    def get_all_comm_view_pair_names(self):
        return _get_all_comm_view_pair_names(self.video_paths)