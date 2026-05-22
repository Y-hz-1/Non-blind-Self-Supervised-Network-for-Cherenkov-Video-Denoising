#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.
"""
import os
import argparse
import random
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import time
import cv2
from scipy.io import savemat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model.models_unsupervised import UnsupervisedDVDnet
from RefastDVDnet1 import open_sequence
from model.model_fast_1 import define_F,LossNetwork,hubber_loss, filter_on_channel
from data.util import batch_psnr, init_logger_test, \
				variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence_1, close_logger
# from fastdvdnet import denoise_seq_fastdvdnet, clean_seq_fastdvdnet
# from utils import batch_psnr, init_logger_test, \
#     variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence, close_logger, LossNetwork, median_filter
# from models import filter_on_channel

# torch.backends.cudnn.benchmark = True  # 允许搜索最优算法
# torch.backends.cudnn.deterministic = False
NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm
OUTIMGEXT = '.tiff'  # output images format


def load_random_patch(patch_dir, patchsize):
    patch_filenames = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
    patch_file = random.choice(patch_filenames)
    patch = Image.open(os.path.join(patch_dir, patch_file)).convert('L')
    patch = transforms.Resize((patchsize, patchsize))(patch)
    patch_tensor = transforms.ToTensor()(patch)[0]
    # a = torch.max(patch_tensor)
    # b = torch.min(patch_tensor)
    patch_tensor = (patch_tensor.clamp(0,1))*2-1
    return patch_tensor.unsqueeze(0).unsqueeze(0)


def compute_psnr_ignore_mask(img1, img2, mask):
    # mask: 1 表示替换区域，不参与 PSNR
    # print(img1.max(), img1.min(), img2.max(), img2.min())
    # img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    # img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    mse = F.mse_loss(img1[mask == 0], img2[mask == 0])
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def compute_ssim_ignore_mask(img1, img2, mask):
    """
    计算 img1 和 img2 的 SSIM（忽略 mask==1 区域）
    img1, img2, mask 均为 PyTorch Tensor，shape: (1, 1, H, W)
    """
    img1_np = img1.squeeze().detach().cpu().numpy()
    img2_np = img2.squeeze().detach().cpu().numpy()
    mask_np = mask.squeeze().detach().cpu().numpy()

    # 将 mask==1 区域设置为 NaN，排除计算
    img1_masked = np.where(mask_np == 1, np.nan, img1_np)
    img2_masked = np.where(mask_np == 1, np.nan, img2_np)

    # 找到有效区域（非 NaN）索引
    valid = ~np.isnan(img1_masked) & ~np.isnan(img2_masked)

    # 只在有效区域计算 SSIM
    if valid.sum() < 9:  # 太小无法计算窗口
        return 0.0

    ssim_val = ssim(img1_masked[valid], img2_masked[valid], data_range=1.0)
    return ssim_val


def CEM_calculation(**args):
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])
    torch.cuda.set_device(0)
    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # Create models
    print('Loading models ...')
    model_temp = UnsupervisedDVDnet(num_input_frames=NUM_IN_FR_EXT)
    #model_temp = OriginFastDVDnet(num_input_frames=NUM_IN_FR_EXT)
    #percepture = LossNetwork().to(device)

    # Load saved weights
    state_temp_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
    else:
        # CPU mode: remove the DataParallel wrapper
        print("no cuda available")
        #state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)

    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    n_accu = 1
    snr = torch.full((1, 1), n_accu, dtype=torch.float32)
    # seq_c, _, _ = open_sequence(args['test_path_clean'], \
    #                             args['gray'], \
    #                             expand_if_needed=False, \
    #                             max_num_fr=1)
    # seq_c = torch.from_numpy(seq_c).to(device)
    #
    # seq_clean = torch.unsqueeze(seq_c, 1)

    # process data
    seq, _,_, _ = open_sequence_1(args['test_path'], \
                              args['gray'], \
                              expand_if_needed=False, \
                              max_num_fr=n_accu * 5)
    seq = seq.to(device)
    seqn = torch.unsqueeze(seq, 1)
    _, C, H, W = seq.shape
    seq_noise = torch.empty((5, C, H, W), device=seq.device)

    # featureExtractor = LossNetwork()
    # x = featureExtractor(seq_noise, seq_noise)

    for p in range(0, 5):
        start = p * n_accu
        end = start + n_accu
        chunk = seq[start:end].squeeze(1).unsqueeze(0)
        chunk = filter_on_channel(chunk, 'sum')

        # tensor_img = chunk.squeeze().cpu().numpy()  # 去掉 batch 和 channel 变成 (1088,1600)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(tensor_img, cmap='gray')
        # plt.title("Tensor Visualization")
        # plt.axis("off")
        # plt.colorbar()
        # plt.show()
        chunk = (chunk / 2048.).clamp(0., 1.)
        chunk = chunk*2-1
        # chunk = (chunk-chunk.min()) / (chunk.max() - chunk.min())
        # chunk[chunk >= 1] = 1
        seq_noise[p] = chunk
    seq_noise = seq_noise.squeeze(1).unsqueeze(0)

    mask_SigArea = torch.ones(1, 1, H, W)
    #mask_SigArea[:, :, 800:800+160, 800:800+160] = 0
    mask_SigArea[:, :, 256:256+160, 756:756+160] = 0
    mask_HalfSigArea = torch.ones(1, 1, H, W)
    mask_HalfSigArea[:, :, 256:256+160, 960:960+160] = 0  #
    patchsize = 32

    loop = 2
    # print(seq_noise.device)
    # print(next(model_temp.parameters()).device)
    mask = torch.zeros_like(seq_noise)
    final_evaluate_map_Global = torch.zeros(loop, 5, H // patchsize, W // patchsize)
    final_evaluate_map_SigArea = torch.zeros(loop, 5, H // patchsize, W // patchsize)
    final_evaluate_map_HalfSigArea = torch.zeros(loop, 5, H // patchsize, W // patchsize)

    with torch.no_grad():
        denframe_ref_Global = model_temp(seq_noise, snr)
        denframe_ref_Global = (denframe_ref_Global + 1) / 2
        # tensor_img = denframe.squeeze().cpu().numpy()  # 去掉 batch 和 channel 变成 (1088,1600)
        # plt.figure(figsize=(8, 6))
        # plt.imshow(tensor_img, cmap='gray')
        # plt.title("Tensor Visualization")
        # plt.axis("off")
        # plt.colorbar()
        # plt.show()
    # denframe_ref_Global = median_filter(denframe_ref_Global, 3)
    maximum_beam = denframe_ref_Global.max()

    for k in range(0, loop):
        temp_evaluate_map_Global = torch.zeros(1, 5, H // patchsize, W // patchsize)
        temp_evaluate_map_SigArea = torch.zeros(1, 5, H // patchsize, W // patchsize)
        temp_evaluate_map_HalfSigArea = torch.zeros(1, 5, H // patchsize, W // patchsize)
        for i in range(0, H, patchsize):
            for j in range(0, W, patchsize):
                if i + patchsize > H or j + patchsize > W:
                    continue

                mask = torch.zeros(1, 1, H, W)
                mask[:, :, i:i + patchsize, j:j + patchsize] = 1
                #plt.imshow(mask[0][0], cmap='gray')
                #plt.show()
                temp_patch = load_random_patch(
                    '/mnt/home/yuhuizhen/project/CEM/patch-DIV2K-8-Demo', patchsize).to(device)
                for m in range(0, 5):
                    input = seq_noise.clone()
                    sum_patch = input[:, m, i:i + patchsize, j:j + patchsize].sum()
                    #t = temp_patch.repeat(1, 1, 1, 1)/sum_patch
                    #tensor_img = t.squeeze().cpu().numpy()  # 去掉 batch 和 channel 变成 (1088,1600)
                    # plt.figure(figsize=(8, 6))
                    # plt.imshow(tensor_img, cmap='gray')
                    # plt.title("Tensor Visualization")
                    # plt.axis("off")
                    # plt.colorbar()
                    # plt.show()
                    input[:, m, i:i + patchsize, j:j + patchsize] = (temp_patch.repeat(1, 1, 1, 1)/sum_patch)
                    # fig, axes = plt.subplots(2, 5, figsize=(18, 6))
                    # fig.suptitle("Patch Replacement Visualization", fontsize=18)
                    #
                    # for m in range(5):
                    #     # Before replacement
                    #     axes[0, m].imshow(input[0, m].cpu().numpy(), cmap="gray")
                    #     axes[0, m].set_title(f"Before - Channel {m + 1}")
                    #     axes[0, m].axis("off")
                    #
                    #     # After replacement
                    #     axes[1, m].imshow(input[0, m].cpu().numpy(), cmap="gray")
                    #     axes[1, m].set_title(f"After - Channel {m + 1}")
                    #     axes[1, m].axis("off")
                    #
                    # plt.tight_layout()
                    # plt.show()
                    with torch.no_grad():
                        denframe = model_temp(input, snr)

                    denframe = (denframe+1)/2
                    # tensor_img = denframe.squeeze().cpu().numpy()  # 去掉 batch 和 channel 变成 (1088,1600)
                    # plt.figure(figsize=(8, 6))
                    # plt.imshow(tensor_img, cmap='gray')
                    # plt.title("Tensor Visualization")
                    # plt.axis("off")
                    # plt.colorbar()
                    # plt.show()

                    # denframe = median_filter(denframe, 3)
                    # denframe = denframes[2, :, :, :].unsqueeze(0)

                    temp_psnr_index_Global = compute_psnr_ignore_mask(denframe, denframe_ref_Global, mask)
                    temp_psnr_index_SigArea = compute_psnr_ignore_mask(denframe, denframe_ref_Global, mask_SigArea)
                    temp_psnr_index_HalfSigArea = compute_psnr_ignore_mask(denframe, denframe_ref_Global, mask_HalfSigArea)
                    temp_evaluate_map_Global[0][m][i // patchsize][j // patchsize] = temp_psnr_index_Global
                    temp_evaluate_map_SigArea[0][m][i // patchsize][j // patchsize] = temp_psnr_index_SigArea
                    temp_evaluate_map_HalfSigArea[0][m][i // patchsize][j // patchsize] = temp_psnr_index_HalfSigArea
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(mask[0][0].detach().cpu().numpy(), cmap='jet')
                    # plt.subplot(1, 3, 2)
                    # plt.imshow(mask_SigArea[0][0].detach().cpu().numpy(), cmap='jet')
                    # plt.subplot(1, 3, 3)
                    # plt.imshow(mask_HalfSigArea[0][0].detach().cpu().numpy(), cmap='jet')
                    # plt.show()
                    # plt.subplot(1, 3, 1)
                    # plt.imshow(input[0][0].detach().cpu().numpy(), cmap='jet')
                    # plt.subplot(1, 3, 2)
                    # plt.imshow(denframe[0][0].detach().cpu().numpy(), cmap='jet')
                    # plt.subplot(1, 3, 3)
                    # plt.imshow(denframe_ref_Global[0][0].detach().cpu().numpy(), cmap='jet')
                    #plt.show()
                print('temp_index: ', temp_psnr_index_Global, temp_psnr_index_SigArea, temp_psnr_index_HalfSigArea)

        final_evaluate_map_Global[k, :, :, :] = temp_evaluate_map_Global
        final_evaluate_map_SigArea[k, :, :, :] = temp_evaluate_map_SigArea
        final_evaluate_map_HalfSigArea[k, :, :, :] = temp_evaluate_map_HalfSigArea

    final_evaluate_map_Global = temp_evaluate_map_Global.squeeze(0).cpu().numpy()
    final_evaluate_map_SigArea = temp_evaluate_map_SigArea.squeeze(0).cpu().numpy()
    final_evaluate_map_HalfSigArea = temp_evaluate_map_HalfSigArea.squeeze(0).cpu().numpy()

    savemat(f'fastDVDnet_final_evaluate_map_Global_nlcl_{n_accu}.mat', {'data': final_evaluate_map_Global})
    savemat(f'fastDVDnet_final_evaluate_map_SigArea_nlcl_{n_accu}.mat', {'data': final_evaluate_map_SigArea})
    savemat(f'fastDVDnet_final_evaluate_map_HalfSigArea_nlcl_{n_accu}.mat', {'data': final_evaluate_map_HalfSigArea})


    # final_evaluate_map = filter_on_channel(final_evaluate_map_Global, 'med')
    # final_evaluate_map = final_evaluate_map.squeeze()
    # data = final_evaluate_map.cpu().numpy()
    # data_norm = (data - data.min()) / (data.max() - data.min())
    # data_uint8 = (data_norm * 255).astype(np.uint8)
    # Image.fromarray(data_uint8).save('final_evaluate_map.tiff')

    # plt.subplot(1, 3, 1)
    # plt.imshow(temp_evaluate_map_Global[0][2].detach().cpu().numpy(), cmap='jet')
    # plt.subplot(1, 3, 2)
    # plt.imshow(temp_evaluate_map_SigArea[0][2].detach().cpu().numpy(), cmap='jet')
    # plt.subplot(1, 3, 3)
    # plt.imshow(temp_evaluate_map_HalfSigArea[0][2].detach().cpu().numpy(), cmap='jet')
    # plt.show()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    # parser.add_argument("--model_file", type=str, \
    #                     default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net0918_1.pth", \
    #                     help='path to model of the pretrained denoiser')
    parser.add_argument("--model_file", type=str, \
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net.pth", \
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--test_path", type=str,
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/cem_1010", \
                        help='path to sequence to denoise')
    parser.add_argument("--test_path_clean", type=str,
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/cem_1010_c", \
                        help='path to sequence to denoise')
    parser.add_argument("--reference", default=True, help='if True, use reference sequence')
    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=500, \
                        help='max number of frames to load per sequence')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/cem_results', \
                        help='where to save outputs as png')
    parser.add_argument("--gray", action='store_true', \
                        help='perform denoising of grayscale images instead of RGB')

    argspar = parser.parse_args()
    argspar.gray = True
    argspar.save_noisy = False
    # Normalize noises ot [0, 1]
    argspar.noise_sigma /= 255.

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    CEM_calculation(**vars(argspar))
