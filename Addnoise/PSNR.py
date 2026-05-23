import torch
import cv2
import os
from skimage import measure
from skimage.metrics import structural_similarity as compare_ssim
import scipy.io as scio
from torch.nn import functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np

def matchmat2(out,gt):
    out = out.cpu().detach().numpy().astype(np.float32).squeeze(0).squeeze(0)
    gt = gt.cpu().detach().numpy().astype(np.float32).squeeze(0).squeeze(0)
    ssim_out=compare_ssim(out, gt ,data_range=gt.max() - out.min(), win_size=7)
    psnr_out=psnr(out, gt)
    #print(compare_ssim(img1, img2 ,data_range=img1.max() - img2.min()))
    #print(compare_ssim(img1, img2, data_range=1.0)) #也可以
    #print(psnr(img1, img2))
    return ssim_out,psnr_out