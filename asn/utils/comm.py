import functools
import os
import os.path as op
import re
from contextlib import contextmanager
import sys
import numpy as np
import itertools
import git
import torch
from torch import Tensor
from torch.autograd import Variable
from sklearn import preprocessing


@contextmanager
def ignorder_exception(*exs):
    ''' usage: with ignorder_exception(OSError): '''
    try:
        yield
    except exs:
        pass

def ls_directories(path):
    return next(os.walk(path))[1]


def create_dir_if_not_exists(file_name):
    ''' iput is file or dir creates a dir if not exits'''
    if not file_name or file_name == "":
        raise ValueError()
    if "." in file_name:
        file_name = os.path.dirname(file_name)
    path = os.path.expanduser(file_name)
    if path:
        if (sys.version_info > (3, 0)):
            os.makedirs(path, exist_ok=True)
        else:
            with ignorder_exception(OSError):
                os.makedirs(path)
    return path


def get_files(path, join_path=False, file_types=None):
    path = op.expanduser(path.strip())
    ret_files = [f for f in os.listdir(path) if op.isfile(op.join(path, f))]
    if file_types is not None:
        ret_files = [f for f in ret_files if f.lower().endswith(file_types)]
    if join_path:
        ret_files = [op.join(path, f) for f in ret_files]
    return ret_files


def split_view_file_name(filename):
    MULTI_VIEW_FILE_SPLIT_STR = ["_view", "_cam"]

    for split in MULTI_VIEW_FILE_SPLIT_STR:
        if split in filename:

            file_name_all, file_extension = op.splitext(filename)
            file_name_all = filename.split(split)
            assert len(file_name_all) > 1, "no \"{}\" in vid file name: {}".format(
                split, filename)
            assert len(file_name_all) <= 2, "multimple  \"{}\" in vid file name: {}".format(
                split, filename)

            re_num = re.findall(r'\d+', file_name_all[1])[0]
            view_num_i = int(re_num)
            vid_file_comm = file_name_all[0]
            extension = file_name_all[1].replace(re_num, "")

            return vid_file_comm, view_num_i, split, extension
    raise FileNotFoundError()


def get_other_view_files(n_views, vid_file_view_i, v_types=(".mov", ".mp4", ".avi")):
    # split name to find multi N views for the format:
    # "KukaArm_kitchen_block1_viewN.mov"
    path, filename = op.split(vid_file_view_i.strip())
    vid_file_comm, view_num_i, view_split, extension = split_view_file_name(filename)
    other_view_files = []
    # check for views in range 0 - nview +1 for differt start indexes
    for j in range(n_views+1):
        if j != view_num_i:
            for ex in itertools.chain((extension,),v_types):
                # check for other views and file types
                fullname_j = op.join(
                    path, vid_file_comm + view_split + str(j) + ex)

                if op.exists(fullname_j) and fullname_j not in other_view_files:
                    assert vid_file_view_i!=fullname_j
                    other_view_files.append(fullname_j)

    assert len(set( other_view_files )) >= n_views-1, "not all views for {}".format(filename)
    return other_view_files


def get_view_pair_vid_files(n_views, vid_dir, append=True,
                            v_types=(".mov", ".mp4", ".avi"), join_path=False):
    # TODO REFACTOR
    vid_dir = op.expanduser(vid_dir.strip())
    assert op.isdir(
        vid_dir), "path is not a directory {}".format(vid_dir)
    vid_files_to_compare = get_files(vid_dir, join_path=True, file_types=v_types)
    lr_start = len(vid_files_to_compare)
    view_pair_files = []
    while len(vid_files_to_compare):
        f = vid_files_to_compare.pop()
        view_n = get_other_view_files(n_views, f,v_types)
        assert not f in view_n, "%s : %s"% (f,view_n)
        comm_name, _, _, _ = split_view_file_name(f)
        # rm other view names
        vid_files_to_compare = [
            s for s in vid_files_to_compare if comm_name != split_view_file_name(s)[0]]
        view_n.append(f)
        view_n = sorted(view_n, key=lambda x: split_view_file_name(x)[1])
        # if join_path:
        # only return n views
        view_n = view_n[:n_views]
        if append:
            view_pair_files.append(view_n)
        else:
            view_pair_files.extend(view_n)

    return view_pair_files


def sliding_window(sequence, winSize, step=1, stride=1, drop_last=False):
    """Returns a generator that will iterate through
    the defined chunks of input sequence. Input sequence
    must be sliceable.

    usage:
        a =np.arange(16)
        for i in sliding_window(a,8,step=1):
            print(i)

    """

    # Verify the inputs
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        print("(type(winSize) == type(0))", (type(winSize) == type(0)))
        raise Exception("type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("step must not be larger than winSize.")
    if winSize*stride > len(sequence):
        raise Exception(
            "winSize*stride ={}*{} must not be larger than sequence length={}.".format(winSize, stride, len(sequence)))

    n = len(sequence)
    last_val = sequence[-1]
    for i in range(0, n, step):
        last = min(i+winSize*stride, n)
        ret = sequence[i:last:stride]
        if not drop_last or len(ret) == winSize:
            yield ret
        if len(ret) != winSize:
            return


def get_git_commit_hash(repo_path):
    repo = git.Repo(search_parent_directories=True,
                    path=os.path.dirname(repo_path))
    assert repo, "not a repo"
    # changed_files = [item.a_path for item in repo.index.diff(None)]
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommited modified files: {}".format(",".join(changed_files)))

    return repo.head.object.hexsha


def tensor_to_np(data):
    return data.cpu().numpy() if isinstance(data, torch.Tensor) else data


def create_label_func(min_val, max_val, bins, clip=False,dtype=np.float32):
    ''' genrate function to encode continus values to n classes
    return mappingf funciton to label or to one hot label
    bins(float): number of bins between min and max
    bins(array_like): boearder for new bins
    usage:
    labl, hot_l = create_lable_func(0, 10, 5)
    x_fit = np.linspace(0, 10, 11)
    print("y leables", labl(x_fit))
    print("yhot leables",  hot_l(x_fit))
    '''
    assert min_val < max_val
    max_val=dtype(max_val)
    min_val=dtype(min_val)
    x_fit = np.linspace(min_val, max_val, 5000, endpoint=True,dtype=dtype)  # TODO cnt
    if clip:
        x_fit = np.clip(x_fit, min_val, max_val)
    if isinstance(bins, int):
        bin_array = np.linspace(min_val, max_val, bins, endpoint=False,dtype=dtype)
        # remove start point
        bin_array=bin_array[1:]
    else:
        assert len(bins) > 2
        bin_array = bins

    def _digitize(x):
        return np.digitize(x, bin_array, right=True)

    x_fit = _digitize(x_fit)
    le = preprocessing.LabelEncoder()
    le.fit(x_fit)
    le_one_hot = preprocessing.LabelBinarizer()
    expected_bins_cnt=bins if isinstance(bins, int) else len(bins)
    assert len(le.classes_) == expected_bins_cnt

    le_one_hot.fit(le.classes_)
    y = le.transform(x_fit)
    yhot = le_one_hot.transform(x_fit)

    def _enc_lables(data):
        fit = tensor_to_np(data)
        if clip:
            fit = np.clip(fit, min_val, max_val)
        return le.transform(_digitize(fit))

    def _enc_labels_hot(data):
        fit = tensor_to_np(data)
        if clip:
            fit = np.clip(fit, min_val, max_val)
        return le_one_hot.transform(_digitize(fit))

    return _enc_lables, _enc_labels_hot



def start_tb_task(path_tb,port=6006):
    import tensorboard
    from tensorboard import default
    from tensorboard import program
    import logging
    try:
        class TensorBoardTool:
            '''Tensorboard V1.12 start'''
            def __init__(self, dir_path,port):
                self.dir_path = dir_path
                self.port=port
            def run(self):
                # Remove http messages
                log = logging.getLogger('werkzeug').setLevel(logging.CRITICAL)
                logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
                # Start tensorboard server
                tb = program.TensorBoard(default.get_plugins(), default.get_assets_zip_provider())
                tb.configure(argv=[None, '--logdir', self.dir_path,'--port',str(self.port)])
                url = tb.launch()
                print('TensorBoard at %s/#scalars&_smoothingWeight=0' % url)
                # Tensorboard tool launch
        tb_tool = TensorBoardTool(path_tb,port)
        tb_tool.run()
    except AttributeError:
        print("ERROR start tensorboard failed, old version V1.12 supported, disable if no data is loaded")


def data_loader_cycle(iterable):
    '''
    Using itertools.cycle has an important drawback, in that it does not shuffle the data after each iteration:
        WARNING  itertools.cycle  does not shuffle the data after each iteratio
        usage         data_iter = iter(cycle(self.data_loader))
        '''
    while True:
        for x in iterable:
            yield x


