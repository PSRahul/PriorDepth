# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""Logging utilities for training
"""
import os

from termcolor import colored
#import horovod.torch as hvd
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter


#from kp2d.utils.wandb import WandBLogger


def printcolor_single(message, color="white"):
    """Print a message in a certain color"""
    print(colored(message, color))


def printcolor(message, color="white"):
    "Print a message in a certain color (only rank 0)"
    if 0 == 0:
        print(colored(message, color))


class SummaryWriter_:
    """Wrapper class for tensorboard and WandB logging"""
    def __init__(self, log_path, params,
                 description=None,
                 project='monodepth',
                 entity='tri',
                 mode='run',
                 job_type='train',
                 log_wb=False):
        self.log_wb = log_wb
        self._global_step = 0        
        if self.log_wb:
            os.environ['WANDB_DIR'] = log_path
            self.wb_logger = WandBLogger(
                params, description=description,
                project=project, entity=entity, mode=mode, job_type=job_type)
        else:
            print("Test 4")
            self.wb_logger = SummaryWriter(log_path)

    @property
    def run_name(self):
        return self.wb_logger.run_name

    @property
    def run_url(self):
        return self.wb_logger.run_url

    @property
    def global_step(self):
        return self._global_step

    def log_wandb(self, value):
        self.log_wb = value

    def add_scalar_(self, tag, scalar_value, epoch=None):
        if self.log_wb:
            self.wb_logger.log_values(tag, scalar_value, now=False)
        else:
            self.wb_logger.add_scalar(tag, scalar_value, epoch)

    def add_image_(self, tag, img_tensor, epoch=None):
        assert img_tensor.max() <= 1.0
        assert (isinstance(img_tensor, torch.Tensor) and img_tensor.device == torch.device(
            'cpu')) or isinstance(img_tensor, np.ndarray)
        if self.log_wb:
            caption = tag
            if isinstance(img_tensor, torch.Tensor):
                # shape: (C, H, W)
                size = tuple(img_tensor.shape[-2:][::-1])
                assert img_tensor.shape[0] == 1 or img_tensor.shape[0] == 3, \
                    'Expects CHW with C=1 or 3, provided {}'.format(img_tensor.shape)
                self.wb_logger.log_tensor_image(img_tensor * 255, tag, caption, size=size, now=False)
            else:
                # shape: (H, W, C)
                size = tuple(img_tensor.shape[:2][::-1])
                assert img_tensor.shape[-1] == 1 or img_tensor.shape[-1] == 3, \
                    'Expects HWC with C=1 or 3, provided {}'.format(img_tensor.shape)
                self.wb_logger.log_numpy_image((img_tensor * 255).astype(np.uint8), tag, caption, size=size, now=False)
        else:
            self.wb_logger.add_image(tag, img_tensor.transpose(2,0,1), epoch)

    def commit_log(self):
        if self.log_wb and self._global_step >= 0:
            self.wb_logger.commit_log()
        self._global_step += 1