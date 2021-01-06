import itertools

import numpy as np
import torch  # before cv2
import torch.multiprocessing as multiprocessing

from asn.utils.comm import sliding_window
from asn.utils.log import log


def get_distances(emb1, emb2):
    assert np.shape(emb1) == np.shape(emb2), "dist shape not same, " "emb1.shape{} and emb1.shape {}".format(
        emb1.shape, emb2.shape
    )
    return np.linalg.norm(emb1 - emb2)


def get_knn(task_embedding, imgs_embeddings, frame_debug, k=1):
    """ k nearest neighbour for two embedding sequences """
    if not isinstance(task_embedding, np.ndarray):
        task_embedding = np.array(task_embedding)
    if not isinstance(imgs_embeddings, np.ndarray):
        imgs_embeddings = np.array(imgs_embeddings)
    indexes_compare = np.arange(imgs_embeddings.shape[0])
    # get all distances for the task frame embedding to all other
    # imgs_embeddings frames
    dist = np.array([get_distances(task_embedding, e) for e in imgs_embeddings])
    index_sorted = dist.argsort()[:k]  # k smallest indexes
    sorted_index_imgs_k = indexes_compare[index_sorted]
    sorted_dist_k = dist[index_sorted]
    # test sorting
    assert np.min(dist) == sorted_dist_k[0]
    return (sorted_index_imgs_k, sorted_dist_k)


def get_all_knn_indexes(task_embeddings, multi_vid_embeddings_to_compare, k=1):
    """get all the knn based on the distance for a embedding compared to multiple
        videos embeddings

    Args:
        task_embeddings(nparray):  embeddings for a video
        multi_vid_embeddings_to_compare(list): list of n videos with the embeddings
                                         for the vid as a nparray

    Returns:
        The return a nparray with (n_task_fames,k,((k_index_vid,k_distance,k_frame_index)
        :param k:

    """
    assert multi_vid_embeddings_to_compare[0].shape[1] == task_embeddings.shape[1], "embedding size not same"
    min_len_comp_vids = min([len(v) for v in multi_vid_embeddings_to_compare])
    if np.abs(len(task_embeddings) - min_len_comp_vids) >= 50:
        log.error("alignment frame difference for knn over max 50 frames")

    n_task_fames = task_embeddings.shape[0]
    knn_img_indexes = np.zeros((n_task_fames, k, 3))
    # frame index, knn, file index, distane nn, predicted_frame
    for frame, task_emb in enumerate(task_embeddings):
        min_dist = None  # DEBUG
        min_frame_index = None
        frame_n_knn = []
        # find all knn to for each frame
        for index_vid, imgs_embeddings in enumerate(multi_vid_embeddings_to_compare):
            k_frame_index, k_dist = get_knn(task_emb, imgs_embeddings, frame, k)
            # k_frame_index =[knn_test[0][0]]
            # k_dist =[knn_test[0][1]]
            f = np.zeros((k, 3))
            f[:, 0] = index_vid
            f[:, 1] = k_dist
            f[:, 2] = k_frame_index
            frame_n_knn.append(f)
            # DEBUG
            if min_dist is None:
                min_dist = np.min(k_dist)
                min_frame_index = k_frame_index[0]
            if min_dist > np.min(k_dist):
                min_dist = np.min(k_dist)
                min_frame_index = k_frame_index[0]

        frame_n_knn = np.array(frame_n_knn)
        frame_n_knn = frame_n_knn.reshape((-1, 3))
        # sort all the knn bis distance column
        # and take the knn indexes
        sor = frame_n_knn[frame_n_knn[:, 1].argsort()[:k]]
        knn_img_indexes[frame] = sor

        # DEBUG
        assert min_dist == knn_img_indexes[frame, 0, 1]
        assert min_frame_index == knn_img_indexes[frame, 0, 2]

    return knn_img_indexes


def get_vid_aligment_loss_pair(embeddings, fill_frame_diff=True):
    """ embeddings(dict), key common view name and values list view embs for each video view"""
    k = 1
    loss, nn_dist, dist_view_pairs = [], [], []
    # compute the nn for all permutations
    # TODO   permutations better but combinations used in TF tImplementation
    for comm_name, view_pair_task_emb in embeddings.items():
        for emb1, emb2 in itertools.combinations(view_pair_task_emb, 2):
            if fill_frame_diff:
                # fill frame diff with the last embeddings
                # similar to the tf implementation
                max_diff = len(emb1) - len(emb2)
                size_embedding = emb1.shape[1]
                if max_diff > 0:
                    emb2 = np.concatenate((emb2, np.full((max_diff, size_embedding), emb1[-1])))
                elif max_diff < 0:
                    emb1 = np.concatenate((emb1, np.full((-max_diff, size_embedding), emb2[-1])))
            knn_img_indexes = get_all_knn_indexes(emb1, [emb2], k=k)
            # get loss assuption view paire
            n_frames = knn_img_indexes.shape[0]
            correct_index = np.arange(n_frames)
            # index for nn with smallest distance
            index_for_nn = knn_img_indexes[:, 0, 2]

            abs_frame_error = np.abs(correct_index - index_for_nn)
            loss_comp = np.mean(abs_frame_error / float(n_frames))
            loss.append(loss_comp)
            # histogram error loss frames index bin count
            error_hist_cnts = []
            for i, abs_err in enumerate(abs_frame_error):
                error_hist_cnts.extend([i] * int(abs_err))
            nn_dist.append(np.mean(knn_img_indexes[:, 0, 1]))
            # print infos
            view_pair_lens = "->".join([str(len(e)) for e in [emb1, emb2]])
            log.info(
                "aligment loss pair {:>30} with {} frames, loss {:>6.5}, mean nn dist {:>6.5}".format(
                    comm_name, view_pair_lens, loss_comp, np.mean(nn_dist)
                )
            )

        # get the distances for all view paris for the same frame
        for emb1, emb2 in itertools.combinations(view_pair_task_emb, 2):
            min_frame_len = min(np.shape(emb1)[0], np.shape(emb2)[0])
            dist_view_i = [get_distances(e1, e2) for e1, e2 in zip(emb1[:min_frame_len], emb2[:min_frame_len])]
            dist_view_pairs.append(np.mean(dist_view_i))
        loss, nn_dist, dist_view_pairs = [np.mean(i) for i in [loss, nn_dist, dist_view_pairs]]
    return loss, nn_dist, dist_view_pairs, error_hist_cnts


def _aligment_loss_process_loop(queue_data, queue_results, fill_frame_diff=True):
    while True:
        embeddings = queue_data.get()
        results = get_vid_aligment_loss_pair(embeddings)
        queue_results.put(results)


def get_embeddings(func_model_forward, data_loader, n_views, func_view_pair_emb_done=None, seq_len=None, stride=None):
    """loss for alignment for view pairs, based on knn distance
    Args:
        func_view_pair_emb_done(function): function to call if one embedded view pair was
                            computed, {comon_name:embeddings}
        data_loader(ViewPairDataset): with shuffle false
        use seq_len, stride for a sequences like mfTCN
    Returns:
        The return a nparray with (n_task_fames,k,((k_index_vid,k_distance,k_frame_index)
        :param func_model_forward:
        :param n_views:
        :param seq_len:
        :param stride:
    """
    num_view_paris = 0
    num_total_frames = 0
    view_pair_emb = {}
    for i, data in enumerate(data_loader):
        # compute the emb for a batch
        frames_batch = data["frame"]
        num_total_frames += len(frames_batch)
        if seq_len is None:
            emb = func_model_forward(frames_batch)
            # add emb to dict and to quue if all frames
            for e, name, view, last in zip(
                emb, data["common name"], data["view"].numpy(), data["is last frame"].numpy()
            ):
                if name not in view_pair_emb:
                    # empty lists for each view
                    # note: with [[]] * n_views the reference is same
                    view_pair_emb[name] = {"embs": [[] for _ in range(n_views)], "done": [False] * n_views}
                view_pair_emb[name]["embs"][view].append(e)
                view_pair_emb[name]["done"][view] = last
                # if all emb for all frames add to queue and compute knn
                if all(view_pair_emb[name]["done"]):
                    # view_pair_lens = [np.shape(e) for e in view_pair_emb[name]]
                    # print("all frame for ",name, "with frames",view_pair_lens )
                    num_view_paris += 1
                    view_pair_emb_name = view_pair_emb.pop(name, None)
                    emb_dict = {name: [np.array(e) for e in view_pair_emb_name["embs"]]}
                    if func_view_pair_emb_done is not None:
                        func_view_pair_emb_done(emb_dict)
        else:
            # get all frames for vid
            for frame, name, view, last in zip(
                frames_batch, data["common name"], data["view"].numpy(), data["is last frame"].numpy()
            ):
                if name not in view_pair_emb:
                    # empty lists for each view
                    # note: with [[]] * n_views the reference is same
                    view_pair_emb[name] = {
                        "frame": [[] for _ in range(n_views)],
                        "embs": [[] for _ in range(n_views)],
                        "done": [False] * n_views,
                    }
                view_pair_emb[name]["frame"][view].append(frame.view(1, *frame.size()))
                view_pair_emb[name]["done"][view] = last

                # compute embeds if all frames
                if last:
                    # loop over all seq as batch
                    frame_batch = torch.cat(view_pair_emb[name]["frame"][view])
                    for i, batch_seq in enumerate(
                        sliding_window(frame_batch, seq_len, step=1, stride=stride, drop_last=True)
                    ):
                        if len(batch_seq) == seq_len:
                            emb = func_model_forward(batch_seq)
                            assert len(emb) == 1
                            view_pair_emb[name]["embs"][view].append(emb[0])
                if all(view_pair_emb[name]["done"]):
                    num_view_paris += 1
                    view_pair_emb_name = view_pair_emb.pop(name, None)
                    emb_dict = {name: [np.array(e) for e in view_pair_emb_name["embs"]]}
                    if func_view_pair_emb_done is not None:
                        func_view_pair_emb_done(emb_dict)

    return num_view_paris, num_total_frames


def view_pair_alignment_loss(
    func_model_forward, n_views, data_loader, num_workers=1, frame_distribution=[], seq_len=None, stride=None
):
    """loss for alignment for view pairs, based on knn distance
        Alignment is the scaled absolute difference between the ground truth time
        and the knn aligned time.
        abs(|time_i - knn_time|) / sequence_length
    Args:
        func_model_forward(func): function to compute a the embeddins,
                                  model should be im eval modus
        data_loader(ViewPairDataset): with shuffle false
        n_process: to compute the knn

    Returns:
        [mean alignment_loss,
        distance for the nearest neighbour,
        distance for the same view pair frame]
        :param n_views:
        :param num_workers:
        :param frame_distribution:
        :param seq_len:
        :param stride:
    """
    assert n_views > 0
    losses = []
    dists_nn = []
    dists_view_pair = []
    frame_distribution_pre = []
    queue_data = multiprocessing.Queue()  # no deep copy on put
    queue_results = multiprocessing.Queue()
    knn_process = []
    # start n Processes to comupte the knn for an embedding
    # the process take data form queue_data and store the results in queue_results
    for i in range(num_workers):
        # for i in range(n_process):TODO
        p = multiprocessing.Process(
            target=_aligment_loss_process_loop, args=(queue_data, queue_results, frame_distribution), daemon=True
        )
        knn_process.append(p)
        p.start()

    def func_emb_done(emb_dict):
        return queue_data.put(emb_dict)

    # compute embbedings and add to the queue
    num_view_paris, num_total_frames = get_embeddings(
        func_model_forward, data_loader, n_views, func_emb_done, seq_len=seq_len, stride=stride
    )

    # wait for all knn Processes
    while num_view_paris != len(losses):
        loss_i, dist_nn_i, dist_view_pair_i, error_hist_cnts = queue_results.get()
        losses.append(loss_i)
        dists_nn.append(dist_nn_i)
        dists_view_pair.append(dist_view_pair_i)
        frame_distribution_pre.extend(error_hist_cnts)
    for p in knn_process:
        p.terminate()
        p.join()  # One must call close() or terminate() before using join().
    return np.mean(losses), np.mean(dists_nn), np.mean(dists_view_pair), frame_distribution_pre
