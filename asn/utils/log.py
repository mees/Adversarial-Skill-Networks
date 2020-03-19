#!/usr/bin/env python

""" logging at one place with color and mpi
usage:
    from bulletrobotgym.utils.blogging import log,set_log_file
    import logging

    if __name__ =="__main__":
        log.setLevel(logging.INFO)
        set_log_file("/tmp/test/test.log",mpi_only_log_rank=0)
        log.debug("debug msg")
        log.info("info msg")
        log.warn("warn msg")
        log.error("err msg")

set the log level:
export BR_GYM_LOG_LEVEL=10
Level 	Numeric value
RITICAL 	50
ERROR   	40
WARNING 	30
INFO 	        20
DEBUG 	        10
NOTSET 	        0
"""

import datetime
import logging
import os
from contextlib import contextmanager
from torchtcn.utils.comm import create_dir_if_not_exists

LOG_LEVEL = int(os.getenv('BR_GYM_LOG_LEVEL', logging.INFO))
datefmt = "%H:%M:%S"
# log msg format with mpi rank if used
_mpi_rank_info = ""
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank() if MPI.COMM_WORLD.Get_size() > 1 else None
    if rank is not None:
        _mpi_rank_info = "{:<10}".format("MPI np {} ".format(rank))
except ImportError:
    rank = None
_format = '%(asctime)s %(levelname)-7s {}%(module)-8s - %(message)s'.format(_mpi_rank_info)

log = logging.getLogger()
try:
    # color log format
    from colorlog import ColoredFormatter
    _format_color = '%(asctime)s %(log_color)s%(levelname)-7s %(reset) s%(module)-8s {}- %(message)s'.format(_mpi_rank_info)
    colors = {'DEBUG': 'reset',
              'INFO': 'reset',
                      'WARNING': 'bold_yellow',
                      'ERROR': 'bold_red',
                      'CRITICAL': 'bold_red'}

    # logging.root.setLevel(logging.DEBUG)
    formatter = ColoredFormatter(
        _format_color, log_colors=colors, datefmt=datefmt)

except ImportError:
    formatter = logging.Formatter(_format,datefmt=datefmt)

_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(formatter)
log.addHandler(_stdout_handler)
_file_handler = None
log.setLevel(LOG_LEVEL)

def disable_log_prints():
    global _stdout_handler
    if _stdout_handler is not None:
        log.removeHandler(_stdout_handler)
    _stdout_handler=None


@contextmanager
def suppress_logging():
    global _stdout_handler
    old = _stdout_handler
    disable_log_prints()
    try:
        yield
    finally:
        _stdout_handler=old
        log.addHandler(_stdout_handler)

def set_log_file(log_file, mpi_only_log_rank=None):
    '''
        mpi_only_rank(int or NOne), mpi rank only to log to the file
    '''
    global _file_handler
    # create file handler which logs even debug messages
    # always overte file unless multi mpi log
    mode = "a" if mpi_only_log_rank is None else "w"
    mpi_info = "only mpi rank {}".format(
        mpi_only_log_rank) if mpi_only_log_rank is not None and rank is not None else ""
    if rank is None or mpi_only_log_rank is None or rank == mpi_only_log_rank:
        if _file_handler is not None:
            log.removeHandler(_file_handler)
        create_dir_if_not_exists(log_file)
        _file_handler = logging.FileHandler(log_file, mode=mode)
        formatter = logging.Formatter(_format, datefmt=datefmt)
        _file_handler.setFormatter(formatter)
        log.addHandler(_file_handler)
        start_dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log.info("Start logging {}, saved to file: {} {}".format(start_dt, log_file, mpi_info))
