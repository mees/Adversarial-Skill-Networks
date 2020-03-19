

import argparse
import functools
import os
from multiprocessing import Process, Queue

import numpy as np
import torch #before cv2
import cv2
import sklearn.utils
from sklearn.utils import shuffle
from torch import multiprocessing
from torch.autograd import Variable
from asn.model.asn import create_model
from asn.utils.comm import get_view_pair_vid_files,create_dir_if_not_exists
from asn.utils.img import (convert_to_uint8, montage, np_shape_to_cv,
                                resize_with_border)
from asn.utils.log import log
from asn.utils.vid_to_np import VideoFrameSampler
from tqdm import tqdm
import time
from contextlib import contextmanager
from itertools import combinations
import subprocess
from collections import defaultdict
'''
show connected web cams:
ls -ltrh /dev/video*

# start recording for n views
py utils/webcam_dataset_creater.py --ports 0,1 --tag test --display
# Hit Ctrl-C when done collecting, upon which the script will compile videos for each view
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-imgs', type=str,
                        help="directory of file separated with \",\" like vid1.mov,vid2.mov",
                        default="~/tcn_data/can_stacking_3_V6/videos/train")
    parser.add_argument('--output-vid-name', type=str,
                        default="out.mp4")
    parser.add_argument(
        '--ports', help="webcam_ports of views with ',' as a seperator, use ls -ltrh /dev/video* to find ports", type=str,required=True)
    parser.add_argument(
        '--fps', help="vido fpsw", type=int, default=24)
    parser.add_argument(
        '--img-height', help="vid output height", type=int, default=150)
    parser.add_argument(
        '--img-width', help="vid output width", type=int, default=150)
    parser.add_argument(
        '--num-frames', help="num of frames", type=int, default=500)
    parser.add_argument(
        '--tag', help="tag the video", type=str,required=True)
    parser.add_argument(
        '--out-dir', help="outout folder", type=str, default="/tmp/tcn_data/webcam")
    parser.add_argument(
        '--set-name', help="folder name of the datset val, train  folder", type=str, default="train")
    parser.add_argument(
        '--max-frame', help="max frames", type=int, default=1000)
    parser.add_argument(
        '--display', help="show frames", action='store_true')
    return parser.parse_args()

@contextmanager
def web_cam_samper(port):
    ''' cv2 webcam manger '''
    video_capture = cv2.VideoCapture(port)
    if not video_capture.isOpened():
        log.error("port is open {}".format(port))
        close_open_web_cams()
        video_capture = cv2.VideoCapture(port)

    # test vid sample
    assert sample_image(video_capture) is not None," cam failed port {} ".format(port)
    # When everything is done, release the capture
    try:
        yield video_capture
    except Exception:
        video_capture.release()
        log.info('release video_capture: {} port {}'.format(video_capture,port))

@contextmanager
def vid_writer(output_vid_name,fps,frame_shape,frame_count=None):
    ''' mange vid cam '''
    fourcc = get_fourcc(output_vid_name)
    # vid writer if shape isknows after rot
    out_shape_cv = np_shape_to_cv(frame_shape[:2])
    vid_writer = cv2.VideoWriter(
            output_vid_name, fourcc, fps, out_shape_cv)
    # vid_writer.write(montage_image)
    yield vid_writer
    if frame_count is not None:
        vid_writer.set(cv2.CAP_PROP_FRAME_COUNT, frame_count)
    vid_writer.set(cv2.CAP_PROP_FPS, fps)
    log.info("output vid: {}".format(output_vid_name))
    vid_writer.release()

def sample_image(camera):
  """Captures a single image from the camera and returns it in PIL format."""
  data = camera.read()
  _, im = data
  return im

def close_open_web_cams():
    # TODO checko for port her
    # Try to find and kill hanging cv2 process_ids.
    try:
        output = subprocess.check_output(['lsof -t /dev/video*'], shell=True)
        log.info('Found hanging cv2 process_ids:')
        log.info(output)
        log.info('Killing hanging processes...')
        output=str(output)
        for process_id in output.split('\n')[:-1]:
            subprocess.call(['kill %s' % process_id], shell=True)
            time.sleep(3)
        # Recapture webcams.
    except subprocess.CalledProcessError:
      raise ValueError(
          'Cannot connect to cameras. Try running: \n'
          'ls -ltrh /dev/video* \n '
          'to see which ports your webcams are connected to. Then hand those '
          'ports as a comma-separated list to --webcam_ports, e.g. '
          '--webcam_ports 0,1')

def adjust_brightness(camera):
     # Take some ramp images to allow cams to adjust for brightness etc.
    for i in range(30):
        sample_image(camera)

def sample_frames(webcam_ports, sample_events,num_frames):
    '''
        generator to sample for each webcam in a different processs and
        yield synchronized, sample of a new frame is triggered with
        sample event
    '''
    process_frames = []
    result_frames_qs = []

    def save_webcam_frames(p_ranke, port, event_sync, result_frames_q):
        ''' sample frame on event set '''
        frame_cnt = 0
        with web_cam_samper(port) as camera:
            log.info('port: {}'.format(port))
            adjust_brightness(camera)
            while True:
                frame=sample_image(camera)
                sample_time=time.time()
                # log.info('port {} sample_time: {},frame_cnt {}'.format(port,sample_time,frame_cnt))
                result_frames_q.put({"frame":frame,
                                     "time":sample_time,
                                     "num":frame_cnt})
                frame_cnt += 1
                event_sync.wait()
                event_sync.clear()


    for p_ranke,(port,e) in enumerate(zip(webcam_ports,sample_events)):
        result_frames_q = multiprocessing.Queue()
        result_frames_qs.append(result_frames_q)
        args_worker = (p_ranke, port, e, result_frames_q)
        p = Process(target=save_webcam_frames, args=args_worker)
        p.daemon = True
        process_frames.append(p)
        p.start()

    for frame_i in range(num_frames):
        # get frames for process
        fames = {port:q_res.get() for port,q_res in zip(webcam_ports,result_frames_qs)}
        yield fames

    for p in process_frames:
        p.join()


def get_fourcc(file_name):
    ''' input file_name or extension '''
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


def display_worker(port_data_queue):
    while True:
        port_data=port_data_queue.get()
        imgs = [[d["frame"] for d in port_data.values()]]
        titels = [[str(d["num"]) for d in port_data.values()]]
        margin = imgs[0][0].shape[0]//20

        montage_image = montage(imgs,
                                margin_color_bgr=[255]*3,
                                margin_top=margin, margin_bottom=margin,
                                margin_left=margin, margin_right=margin,
                                margin_seperat_vertical=margin,
                                margin_seperat_horizontal=margin,
                                titels=titels, fontScale=0.8)

        cv2.imshow('Video', montage_image)
        cv2.waitKey(1)

def save_img_worker(ports,save_dir,tag, port_data_queue,result_file_name_que):
    # port_data soted in view order
    create_dir_if_not_exists(save_dir)
    while True:
        port_data=port_data_queue.get()
        files_names={}
        for view_i, p in enumerate(ports):
            f_cnt=port_data[p]["num"]
            frame=port_data[p]["frame"]
            filenamne="frame{}_{}_view{}.png".format(f_cnt,tag,view_i)
            filenamne=os.path.join(save_dir,filenamne)
            files_names[p]=filenamne
            cv2.imwrite(filenamne,frame)
        result_file_name_que.put(files_names)

def get_all_queue_result(queue):
    ''' queue data to list '''
    result_list = []
    while not queue.empty():
        result_list.append(queue.get())

    return result_list

def save_vid_worker(img_files,view_i, save_dir,tag, img_size,fps):
    # port_data soted in view order
    create_dir_if_not_exists(save_dir)
    output_vid_name="{}_view{}.mov".format(tag,view_i)
    frame_count=len(img_files)
    out_shape_cv = img_size[:2]
    # out_shape_cv = np_shape_to_cv(img_size[:2])
    output_vid_name=os.path.join(save_dir,output_vid_name)
    with vid_writer(output_vid_name,fps,out_shape_cv,frame_count) as vid:
        for im_f in img_files:
            img = cv2.imread(im_f)
            vid.write(img)

def main():


    args = get_args()
    args.out_dir = os.path.expanduser(args.out_dir)
    image_size = (args.img_height, args.img_width)

    ports= list(map(int, args.ports.split(',')))
    log.info('ports: {}'.format(ports))
    sample_events=[multiprocessing.Event() for _ in ports]
    num_frames=args.max_frame
    if args.display:
        disp_q = multiprocessing.Queue()
        p = Process(target=display_worker, args=(disp_q,),daemon=True)
        p.start()
    # process to save images as a file
    im_data_q,im_file_q = multiprocessing.Queue(),multiprocessing.Queue()
    img_folder = os.path.join(args.out_dir,"images",args.set_name,args.tag)
    vid_folder = os.path.join(args.out_dir,"videos",args.set_name)
    img_args=(ports, img_folder,args.tag,im_data_q,im_file_q)
    p = Process(target=save_img_worker, args=img_args,daemon=True)
    p.start()

    log.info('img_folder: {}'.format(img_folder))
    log.info('vid_folder: {}'.format(vid_folder))
    log.info('fps: {}'.format(args.fps))

    try:
        time_prev=time.time()
        # loop to sample frames with events
        for frame_cnt,port_data in enumerate(sample_frames(ports,sample_events,num_frames)):
            sample_time_dt=time.time()-time_prev
            if frame_cnt % 10==0:
                log.info('frame {} time_prev: {}'.format(frame_cnt,time.time()-time_prev))

            time_prev=time.time()
            # set events to trigger cams
            for e in sample_events:
                e.set()

            if frame_cnt==0:
                # skip first frame because not  synchronized with event
                log.info('START: {}'.format(frame_cnt))
                continue
            elif (sample_time_dt-1./args.fps) > 0.1:
                log.warn("sampling frame taks too long for fps")
            # check sampel time diff
            if len(ports)>1:
                dt = [np.abs(p1["time"]-p2['time']) for p1,p2 in combinations(port_data.values(),2)]
                # log.info('dt: {}'.format(np.mean(dt)))
                if np.max(dt)>0.1:
                    log.warn('camera sample max time dt: {}, check light condition and camera models'.format(np.max(dt)))
            assert all(frame_cnt==d["num"] for d in port_data.values()), "out of sync"

            im_data_q.put(port_data)
            if args.display:
                disp_q.put(port_data)

            time.sleep(1./args.fps)
    except KeyboardInterrupt:
        # create vids form images save before
        im_shape={p:d["frame"].shape for p,d in port_data.items()}
        img_files=defaultdict(list)
        for d in get_all_queue_result(im_file_q):
            for p, f in d.items():
                img_files[p].append(f)
        # TODO start for each a procresss and join
        for view_i,p in enumerate(port_data.keys()):
            save_vid_worker(img_files[p],view_i,vid_folder,args.tag,im_shape[p],args.fps)

    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
