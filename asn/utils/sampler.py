import numpy as np
from sklearn.utils import shuffle
from torch.utils.data.sampler import Sampler

from asn.utils.dataset import (DoubleViewPairDataset,
                                    get_positive_frame_index)


class ViewPairSequenceSampler(Sampler):
    '''sampler to use with the DoubleViewPairDataset to
        sample n sequences form each view pair
    '''

    def __init__(self, dataset, examples_per_sequence, batch_size,
                 similar_frame_margin=0, shuffle_sequence=False):
        '''
            examples_per_sequence: number of example from one vid
            similar_frame_margin: window to sample a similar frame

        '''
        self.dataset = dataset
        self._examples_per_sequence = examples_per_sequence
        self._shuffle_sequence = shuffle_sequence
        if not isinstance(dataset, DoubleViewPairDataset):
            raise ValueError("unsupported dataset: {}".format(dataset.__class__.__name__))
        # main_source_len = len(self.main_source)
        #
        # how_many = int(round(main_source_len / len(self.indices)))
        # self.to_iter_from = []
        # for _ in range(how_many):
        #     self.to_iter_from.extend(self.indices)
        # Take a number of sequences for the batch.
        assert batch_size % examples_per_sequence == 0
        self._sequences_per_batch = batch_size // examples_per_sequence
        self._examples_per_sequence = examples_per_sequence
        self._similar_frame_margin = similar_frame_margin
        self._num_samples = self.dataset.get_number_view_pairs() * \
            self._examples_per_sequence
        min_len_vid = min(min(l) for l in self.dataset.frame_lengths)
        # check if vids long enough to smaple wihout replace with margins
        assert min_len_vid >= (1+self._similar_frame_margin*2) * \
            batch_size, "vid to small for batch size and margin"

    def __iter__(self):
        ''' sample n examples_per_sequence, without replace choosen
            frame index (optional with margin)
        '''
        np.random.seed()
        n = self.dataset.get_number_view_pairs()
        idx = []
        for view_pair_index in shuffle(range(n)):
            num_frames = min(self.dataset.frame_lengths[view_pair_index])

            # smaple frames random
            frames = range(num_frames)
            for _ in range(self._examples_per_sequence):
                assert len(
                    frames) >= self._similar_frame_margin, "no frames left to sample"
                # sample an frame and rm frames in margin
                anchor_index = np.random.choice(frames)
                # remove similar frames, so that not sampled again
                # -> so they can not be a negative
                frame_to_rm = get_positive_frame_index(anchor_index,
                                                       num_frames,
                                                       self._similar_frame_margin)
                frames = [f for f in frames if f not in frame_to_rm]
                sample_index = self.dataset.idx_lookup[view_pair_index][anchor_index]
                idx.append(sample_index)

        # assert self._num_samples == len(idx)
        if self._shuffle_sequence:
            idx = shuffle(idx)
        return iter(idx)

    def __len__(self):
        return self._num_samples


class RNNViewPairSequenceSampler(Sampler):
    '''sampler to use with the DoubleViewPairDataset to
        sample n sequences form each view pair
    '''

    def __init__(self, dataset,
                 sequence_length,
                 stride,
                 sequences_per_vid_in_batch,
                 batch_size,
                 allow_same_frames_in_seq=False):
        '''
            dataset: DoubleViewPairDataset
            sequence_length: length of seq
            stride: take every n frame
            sequences_per_vid_in_batch: number of example seq. from one vid on each batch
            allow_same_frames_in_seq: allows same frame the seq, start frame
                                can never be same
        '''
        self.dataset = dataset
        self._stride = stride  # frame steps
        self._sequence_length = sequence_length
        self._sequences_per_vid_in_batch = sequences_per_vid_in_batch
        if not isinstance(dataset, DoubleViewPairDataset):
            raise ValueError()
        assert dataset.n_frame == 1, "dont skip n frame, use "
        assert batch_size % sequences_per_vid_in_batch*sequence_length == 0
        assert self._stride > 0
        # nuber of all frame in one epoch

        self._num_samples = self.dataset.get_number_view_pairs() * \
            self._sequences_per_vid_in_batch * self._sequence_length
        self.allow_same_frames_in_seq = allow_same_frames_in_seq
        if not allow_same_frames_in_seq:
            min_len_vid = min(min(l) for l in self.dataset.frame_lengths)
            # check if vids long enough to smaple wihout replace with margins
            assert min_len_vid >= sequence_length*self._sequences_per_vid_in_batch * \
                self._stride, "vid to small for batch size and seq len"

    def __iter__(self):
        ''' sample n examples_per_sequence, without replace choosen
            frame index (optional with margin)
        '''
        np.random.seed()
        n = self.dataset.get_number_view_pairs()
        idx = []
        # get all dataset indexes for alll view pairs
        for view_pair_index in shuffle(range(n)):
            # get the min of the frames
            num_frames = min(self.dataset.frame_lengths[view_pair_index])
            num_frames -= self._sequence_length*self._stride  # can be sampeld end seq in vid
            # smaple frames random
            frames = range(num_frames)
            for _ in range(self._sequences_per_vid_in_batch):
                assert len(
                    frames) >= self._sequence_length, "no frames left to sample"
                # sample an frame and rm frames in margin
                anchor_index = np.random.choice(frames)
                # rm only anchor_index tot
                get_positive_frame_index(anchor_index, num_frames, 0)
                if self.allow_same_frames_in_seq:
                    # only rm anchor indexes
                    frame_to_rm = [anchor_index]
                else:
                    # remove all sampled frames
                    frame_to_rm = range(anchor_index, anchor_index +
                                        self._sequence_length*self._stride)
                    assert len(frame_to_rm) == self._sequence_length*self._stride
                frames = [f for f in frames if f not in frame_to_rm]
                assert len(frames) >= self._sequence_length
                # get the index for the frame in the dataset
                start_sample_index = self.dataset.idx_lookup[view_pair_index][anchor_index]
                n = start_sample_index+self._sequence_length*self._stride

                idx_dataset_frame_seq = range(start_sample_index, n, self._stride)
                # DEBUG check if all frames are for same view pairs vids
                # self.debug_chek_seq(idx_dataset_frame_seq, view_pair_index)
                idx.extend(idx_dataset_frame_seq)
        assert len(idx) == self._num_samples
        return iter(idx)

    def __len__(self):
        return self._num_samples

    def debug_chek_seq(self, indx_dataset, view_pair_index):
        ''' assert if index in data set are not ok'''
        prev_i_test = None
        for i in indx_dataset:
            view_pair_index_test, frame_index_test = self.dataset.item_idx_lookup[i]
            assert view_pair_index_test == view_pair_index, "error wrong vid"
            if prev_i_test is not None:
                # check strides
                assert prev_i_test + self._stride == frame_index_test, "error wrong frame"
            prev_i_test = frame_index_test

class StartEndSequenceSampler(Sampler):
    ''' sampel state and end state of a task vid
    '''

    def __init__(self, dataset, margin_start=5, margin_end=5):
        '''

        '''
        self.dataset = dataset
        if not isinstance(dataset, DoubleViewPairDataset):
            raise ValueError("unsupported dataset: {}".format(dataset.__class__.__name__))
        # main_source_len = len(self.main_source)
        #
        # how_many = int(round(main_source_len / len(self.indices)))
        # self.to_iter_from = []
        # for _ in range(how_many):
        #     self.to_iter_from.extend(self.indices)
        # Take a number of sequences for the batch.
        self._margin_start = margin_start
        self._margin_end = margin_end
        # start end end state for each vid
        self._num_samples = self.dataset.get_number_view_pairs() * 2
        min_len_vid = min(min(l) for l in self.dataset.frame_lengths)
        # check if vids long enough to smaple wihout replace with margins
        assert min_len_vid > margin_start+margin_end+2, 'min vid to small for margin'

    def __iter__(self):
        ''' sample random start and end states for mulitple vids'''
        np.random.seed()
        n = self.dataset.get_number_view_pairs()
        idx = []
        for view_pair_index in shuffle(range(n)):
            num_frames = min(self.dataset.frame_lengths[view_pair_index])
            # smaple frames random fames in start and end range
            idx_start = np.random.choice(range(self._margin_start))
            idx_end = np.random.choice(range(num_frames-self._margin_end-1,num_frames))
            sample_index_start = self.dataset.idx_lookup[view_pair_index][idx_start]
            sample_index_end = self.dataset.idx_lookup[view_pair_index][idx_end]
            vid_idx = shuffle([sample_index_start,sample_index_end])
            idx.extend(vid_idx)
        return iter(idx)

    def __len__(self):
        return self._num_samples

    def get_start_end_lable(self, batch_frames_idx):
        lable_start_end=(batch_frames_idx<=self._margin_start).long()
        return lable_start_end
