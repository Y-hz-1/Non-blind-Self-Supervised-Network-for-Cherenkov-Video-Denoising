import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tensorboardX import SummaryWriter

def transform(gt_sample,raw_sample):
    # define transformations
    do_nothing = lambda x: x
    do_nothing.__name__ = 'do_nothing'
    flipud = lambda x: torch.flip(x, dims=[2])
    flipud.__name__ = 'flipup'
    rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
    rot90.__name__ = 'rot90'
    rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
    rot90_flipud.__name__ = 'rot90_flipud'
    rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
    rot180.__name__ = 'rot180'
    rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
    rot180_flipud.__name__ = 'rot180_flipud'
    rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
    rot270.__name__ = 'rot270'
    rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
    rot270_flipud.__name__ = 'rot270_flipud'
    add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
                                          std=(5 / 255.)).expand_as(x).to(x.device)
    add_csnt.__name__ = 'add_csnt'

    # define transformations and their frequency, then pick one.
    aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
                rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
    w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12]  # one fourth chances to do_nothing
    transf = choices(aug_list, w_aug)
    # transform all images in array
    return transf[0](gt_sample),transf[0](raw_sample)