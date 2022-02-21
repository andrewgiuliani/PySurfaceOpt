#import logging
#import os
#from mpi4py import MPI
#comm = MPI.COMM_WORLD

#__all__ = ("debug", "info", "warning", "error")
#
#logger = logging.getLogger('PySurfaceOpt')
#handler = logging.StreamHandler()
#formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
#handler.setFormatter(formatter)
#
##if comm is not None and comm.rank != 0:
##    handler = logging.NullHandler()
##logger.addHandler(handler)
#
#if comm.rank != 0:
#    logger.disabled = True
#
#
#def set_file_logger(path):
#    fileHandler = logging.FileHandler(path, mode='w')
#    logger.addHandler(fileHandler)
#
#logger.setLevel(logging.INFO)
#
#debug = logger.debug
#info = logger.info
#warning = logger.warning
#error = logger.error
