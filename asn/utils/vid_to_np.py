import argparse
import functools
import os.path
import timeit

import numpy as np

import cv2
import skvideo.io
import torch
from sklearn.utils import shuffle
from asn.utils.img import np_shape_to_cv
from asn.utils.log import log
from torchvision.utils import save_image


class VideoFrameSampler():
    ''' optimized if random frames are sampled from a video'''

    def __init__(self, vid_path, resize_shape=None, dtype=np.uint8, to_rgb=True, torch_transformer=None):
        self._vid_path = os.path.expanduser(vid_path)
        assert os.path.isfile(
            self._vid_path), "vid not exists: {}".format(self._vid_path)
        self.fps = None
        self.to_rgb = to_rgb
        self.approximate_frameCount = None
        self.frameWidth = None
        self.frameHeight = None
        self.resize_shape = resize_shape  # cv2 resize
        self.torch_transformer = torch_transformer  # torch vision
        if self.torch_transformer is not None:
            assert dtype == np.uint8 and self.resize_shape is None
            if not to_rgb:
                log.warn("bgr imag and torch transformer")
        self.dtype = dtype
       
    def _get_vid_info(self, cap_open=None):
        if cap_open is None:
            cap = cv2.VideoCapture(self._vid_path)
        else:
            cap = cap_open
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.approximate_frameCount = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT))  # not accurate!
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if cap_open is None:
            cap.release()

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError("VideoFrameSampler index out of range")
        return self.get_frame(item)

    def __len__(self):
        if self.approximate_frameCount is None:
            self._get_vid_info()
        return self.approximate_frameCount

    def get_fps(self):
        if self.fps is None:
            self._get_vid_info()
        return self.fps

    def count_frames(self):
        ''' count the frame to get a correct value'''
        frame_count = 0
        read_next = True
        cap = cv2.VideoCapture(self._vid_path)
        if self.approximate_frameCount is None:
            self._get_vid_info(cap)
        while(read_next):
            # set frame pos -> has different results without for some vids
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            # cap.set can lead to an inf loop
            read_next, _ = cap.read()

            if read_next:
                frame_count += 1
            # end if frame pos can t be increased
            if cap.get(cv2.CAP_PROP_POS_FRAMES) != frame_count:
                read_next = False
        cap.release()
        self.approximate_frameCount = int(frame_count)
        return frame_count

    def get_all(self):
        tmp = self.get_frame(0)
        # TODO CAP_PROP_FRAME_COUNT is a approximate
        if not isinstance(tmp, torch.Tensor):
            all_frames = np.empty((len(self),) + tmp.shape, self.dtype)
            for i, rgb in enumerate(self):
                all_frames[i, :, :, :] = rgb
        else:
            all_frames = torch.zeros((len(self),)+tmp.size(), dtype=tmp.dtype)
            try:
                backend =  "ffmpeg" if skvideo._HAS_FFMPEG else "libav"
                if not self.to_rgb:# convert to bgr not supported here TODO
                    raise ValueError()
                vid = skvideo.io.vread(self._vid_path,backend=backend)
            except ValueError as e:
                log.warn("skvideo failed, falling back to cv2")
                vid = self
            for i, rgb in enumerate(vid):
                if self.torch_transformer is not None:
                    rgb = self.torch_transformer(rgb)
                all_frames[i, :, :, :] = rgb

        return all_frames

    def get_frame(self, frame_index):
        """Gets a frame at a specified index in a video."""
        cap = cv2.VideoCapture(self._vid_path)
        if self.approximate_frameCount is None:
            self._get_vid_info(cap)
        if frame_index < 0 or frame_index >= self.approximate_frameCount:
            msg = "frame {} to high for video {} with appr. {} frames".format(
                frame_index, self._vid_path, self.approximate_frameCount)
            raise IndexError(msg)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            # approximate_frameCount was wrong
            # reduce the trame index to get last frame
            trys = 10
            try_count = 1
            while(not ok and try_count <= trys):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - try_count)
                ok, bgr = cap.read()
                try_count += 1
            msg = "read frame {} faild for video {} with appr. {} frames reduced to {}".format(
                frame_index, self._vid_path, self.approximate_frameCount, frame_index - try_count)
            log.warning(msg)
        cap.release()
        assert bgr is not None, "faild reading frame for video frame {}, header frames {}".format(
            frame_index, self.count_frames())
        if self.resize_shape is not None and self.frameWidth != self.resize_shape[1] and self.frameHeight != self.resize_shape[0]:
            # cv2 mat size is fliped -> ::-1
            bgr = cv2.resize(bgr, np_shape_to_cv(
                self.resize_shape), cv2.INTER_NEAREST)
        if self.to_rgb:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.dtype in [np.float32, np.float64]:
            return np.asarray(bgr / 255., dtype=self.dtype)
        else:
            bgr = np.asarray(bgr, dtype=self.dtype)
            if self.torch_transformer is not None:
                bgr = self.torch_transformer(bgr)
            return bgr


def random_frame_sample_timeit(vid_file):
    ''' test for time to sample differt random frames'''
    N_frames = 32

    def _sample_random_frames():
        v = VideoFrameSampler(vid_file)
        frames_rand = np.random.choice(np.arange(len(v)), size=N_frames)
        return [v.get_frame(i) for i in frames_rand]
    print("timeit differt frame sample for: ", vid_file)
    print(timeit.timeit(_sample_random_frames, number=1000))

    def _sample_random_frames_skvid():
        v = skvideo.io.vread(vid_file)
        frames_rand = np.random.choice(np.arange(v.shape[0]), size=N_frames)
        return [v[i] for i in frames_rand]
    print("timeit differt frame sk vid for: ", vid_file)
    print(timeit.timeit(_sample_random_frames_skvid, number=1000))


def main():
    from matplotlib import pyplot as plt
    plt.ion()
    plt.show()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--task-vid', type=str, default='vid.mov')
    args = parser.parse_args()
    random_frame_sample_timeit(args.task_vid)

    for rgb in VideoFrameSampler(args.task_vid):
            # plt.imshow(rgb, interpolation='nearest')
            # cv2.imwrite("env_debug/image.jpg", rgb)
        plt.draw()
        plt.pause(0.001)

    v = VideoFrameSampler(args.task_vid)
    imgs = v.get_all()
    for i in range(len(v)):
        plt.imshow(imgs[i], interpolation='nearest')
        plt.draw()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
