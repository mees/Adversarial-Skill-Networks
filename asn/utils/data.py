import time

import numpy as np
import torch
from torch import multiprocessing
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from asn.utils.comm import get_view_pair_vid_files
from asn.utils.img import flip_imgs
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler


def _sampler_process_loop(queue, sampler, epochs):
    sampler.init()
    if epochs < sampler.epoch:
        log.error("No epochs to sample ")
    while epochs > sampler.epoch:
        dataset = ConcatDataset([sampler.build_set()])
        queue.put((sampler.rank, sampler.epoch, dataset))
    while True:
        time.sleep(0.1)  # TODO


class Sampler(object):
    def __init__(self, rank=0, n_samper_process=1):
        assert rank >= 0
        assert n_samper_process > 0
        self.rank = rank
        self.epoch = 0
        self.n_samper_process = n_samper_process

    def build_set(self):
        pass

    def init(self):
        pass

    def get_rank_working_part(self, list_to_split):
        """ return a list part for the rank based on the total total number of workers (n_samper_process)"""
        if self.n_samper_process == 1:
            assert self.rank < 1
            assert self.rank < self.n_samper_process
            return list_to_split
        else:
            n = len(list_to_split) // self.n_samper_process
            start_index = self.rank * n
            if not self.rank == self.n_samper_process - 1:
                end_index = start_index + n
                return list(list_to_split[start_index:end_index])
            else:
                # split untill the end
                return list(list_to_split[start_index:])


class AlignmentViewPairSampler(Sampler):
    def __init__(self, n_views, vid_dir, image_size, n_frame=1, sample_size=-1, *args, **kwargs):
        """sample_size =-1 to get all frames at once elese the vid is dividen into
        chunks ot the size sample_size
        n_frame: take every n frame of the sequence
        """
        super().__init__(*args, **kwargs)
        assert n_views >= 2 and n_frame > 0
        if n_frame != 1 and sample_size != -1:
            raise NotImplementedError("sample_size only with n_frames 1")
        self.sample_size = sample_size
        self.n_frame = n_frame
        self.frame_size = image_size
        self.n_views = n_views
        self.vid_dir = vid_dir

    def init(self):
        self.video_files = self.get_rank_working_part(get_view_pair_vid_files(self.n_views, self.vid_dir))
        self.video_count = len(self.video_files)
        assert self.video_count, "no vids for worker"
        self.video_index = 0
        self.sample_frames_index = 0

    def _sample_every_n_frame(self, vid_files):
        views = []
        for f in vid_files:
            vid = VideoFrameSampler(f, resize_shape=self.frame_size, dtype=np.float32)
            if self.n_frame == 1:
                imgs = vid.get_all()
            else:
                num_frames_sample = len(vid) // self.n_frame
                imgs = np.empty((num_frames_sample, *self.frame_size, 3))
                for frame in range(num_frames_sample):
                    imgs[frame] = vid.get_frame(frame * self.n_frame)
            t = torch.Tensor(flip_imgs(imgs))
            views.append(t.view(1, -1, 3, *self.frame_size))
        return views

    def _sample_next_frames(self, vid_files):
        views = []
        all_frames_sampled = False
        for f in vid_files:
            # get next frames for vid
            vid = VideoFrameSampler(f, resize_shape=self.frame_size, dtype=np.float32)
            max_len = self.sample_size
            if self.sample_size >= len(vid) or self.sample_size == -1:
                log.warning("vid {} wiht {}frames has less than for sample size".format(f, len(vid)))
                max_len = len(vid)
                all_frames_sampled = True
            elif self.sample_frames_index + self.sample_size >= len(vid):
                # last sample smaller
                max_len = len(vid) - self.sample_frames_index
                log.info(
                    "end {} with {} frames end size {}, index {}".format(f, len(vid), max_len, self.sample_frames_index)
                )
                all_frames_sampled = True
            imgs = np.empty((max_len, *self.frame_size, 3))
            for j in range(max_len):
                imgs[j] = vid.get_frame(j + self.sample_frames_index)
            t = torch.Tensor(flip_imgs(imgs))
            views.append(t.view(1, -1, 3, *self.frame_size))
        self.sample_frames_index += self.sample_size
        if all_frames_sampled:
            self.sample_frames_index = 0
        return all_frames_sampled, views

    def build_set(self):
        vid_files = self.video_files[self.video_index]
        views = []
        all_frames_sampled = False
        if self.sample_size == -1 or self.n_frame != 1:
            all_frames_sampled, views = True, self._sample_every_n_frame(vid_files)
        else:
            all_frames_sampled, views = self._sample_next_frames(vid_files)

        if all_frames_sampled:
            if self.video_count == (self.video_index + 1):
                self.epoch += 1
            self.video_index = (self.video_index + 1) % self.video_count
        return TensorDataset(*views)


def get_data_loader(
    use_cuda,
    path_imgs,
    epochs,
    n_process=2,
    minibatch_size=16,
    sample_size=8,
    image_size=(300, 300),
    shuffle=True,
    sampler_class=AlignmentViewPairSampler,
    **kwargs_samper,
):
    assert n_process > 0 and epochs > 0
    queue_max_size = n_process + 2
    queue = multiprocessing.Queue(queue_max_size)
    dataset_builder_process = []
    # start n Processes which add add data to the queue, will block if full
    for i in range(n_process):
        sampler = sampler_class(
            rank=i,
            vid_dir=path_imgs,
            image_size=image_size,
            sample_size=sample_size,
            n_samper_process=n_process,
            **kwargs_samper,
        )
        p = multiprocessing.Process(target=_sampler_process_loop, args=(queue, sampler, epochs), daemon=True)
        p.start()
        dataset_builder_process.append(p)

    epoch = 0
    epoch_ranks = np.zeros(n_process, dtype=np.int)
    while epoch < epochs:
        rank, epoch_rank_i, dataset = queue.get()  # blocks until there is new data
        epoch_ranks[rank] = epoch_rank_i
        epoch = epoch_ranks.min()
        data_loader = DataLoader(
            dataset=dataset, batch_size=minibatch_size, shuffle=shuffle, pin_memory=use_cuda  # shuffle in sampler
        )

        yield epoch, data_loader

    if epoch != epochs:
        log.error("wrong epoch count!")

    for p in dataset_builder_process:
        p.terminate()
        p.join()
