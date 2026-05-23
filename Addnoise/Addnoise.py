import os
import cv2
import glob
import yaml
import torch
import visdom
import random
import logging
import argparse
import functools
import subprocess
import numpy as np
from PIL import Image
import torch.nn as nn
from util import option
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from Dnet import NLayerDiscriminator
from PSNR import matchmat2
import torchvision.utils as vutils
from data_argu import transform
from random import choices  # requires Python >= 3.6
from Degmodel import DegModel
from losses import define_F,LossNetwork,visualize_feature_map, SSIM
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.checkpoint import checkpoint
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from Addnoise_model import ReFastDVDnet, ResnetGenerator, init_net
import torch.nn.init as init

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None and hasattr(net, 'parameters'):
            for param in net.parameters():
                param.requires_grad = requires_grad
        else:
            print(f"Warning: {net} 不是一个网络对象。")
'''加权损失'''
def create_weight_mask(image, threshold=0.1, alpha=10, beta=1):
    """
    根据图像生成权重掩码。

    :param image: 输入的图像（灰度图）。
    :param threshold: 判断纯黑区域的阈值。
    :param alpha: 纯黑区域的权重。
    :param beta: 其他区域的权重。
    :return: 权重掩码。
    """
    # 创建一个与图像大小相同的权重掩码
    weight_mask = np.where(image < threshold, alpha, beta)
    return weight_mask


'''save_model_checkpoint'''
def save_model_checkpoint(model, log_dir, optimizer, train_pars, epoch):
	"""Stores the model parameters under 'argdict['log_dir'] + '/net.pth'
	Also saves a checkpoint under 'argdict['log_dir'] + '/ckpt.pth'
	"""
	torch.save(model.state_dict(), os.path.join(log_dir, 'net.pth'))
	save_dict = { \
		'state_dict': model.state_dict(), \
		'optimizer' : optimizer.state_dict(), \
		'training_params': train_pars, \
		'args': argdict\
		}
	torch.save(save_dict, os.path.join(log_dir, 'ckpt.pth'))

	if epoch % 5 == 0:
		torch.save(save_dict, os.path.join(log_dir, 'ckpt_e{}.pth'.format(epoch+1)))
	del save_dict

def weighted_mse_loss(pred, target, weight_mask):
    """
    计算加权 MSE 损失。

    :param pred: 模型预测。
    :param target: 目标图像。
    :param weight_mask: 权重掩码。
    :return: 加权 MSE 损失。
    """
    mse_loss = (pred - target) ** 2
    weighted_loss = mse_loss * weight_mask
    return weighted_loss.mean()


def weighted_bce_loss(pred, target, weight_mask):
    """
    计算加权 BCE 损失。

    :param pred: 模型预测。
    :param target: 目标图像。
    :param weight_mask: 权重掩码。
    :return: 加权 BCE 损失。
    """
    bce_loss = nn.functional.binary_cross_entropy(pred, target, reduction='none')
    weighted_loss = bce_loss * weight_mask
    return weighted_loss.mean()


'''noise_map'''
def generate_hotpixel_noise_map(a, image_shape, num_Hotpixels_range=(300,400)):

    s = image_shape
    m = a.item()  # 设置最大噪声值为1
    #print(m)
    noise_num = np.random.randint(num_Hotpixels_range[0], num_Hotpixels_range[1] + 1)
    noise_matrix = np.zeros(s, dtype=np.float32)

    # 生成随机的噪声位置
    linear_indices = np.random.choice(s[-2] * s[-1], noise_num, replace=False)
    noise_matrix_flat = noise_matrix.reshape(-1)
    noise_matrix_flat[linear_indices] = m
    noise_matrix = noise_matrix_flat.reshape(s)

    randr = [0, -4, 4]
    l = len(randr)
    row, col = np.where(noise_matrix[0] == m if len(s) == 3 else noise_matrix == m)

    for r, c in zip(row, col):
        gaussian_height = 5 + np.random.choice(randr)  #高斯核的高度
        gaussian_width = 5 + np.random.choice(randr)
        # impulse = np.zeros((gaussian_height, gaussian_width))
        # impulse[gaussian_height // 2, gaussian_width // 2] = 1
        # #h = gaussian_filter(np.zeros((gaussian_height, gaussian_width)), sigma=10000000)
        # h = gaussian_filter(impulse, sigma=10000000)
        # h = (h - np.min(h)) / (np.max(h) - np.min(h))
        h = np.multiply(cv2.getGaussianKernel(gaussian_height, sigma=10000000),
                        (cv2.getGaussianKernel(gaussian_width, sigma=10000000)).T)#0000000
        if np.max(h) != np.min(h):
            h = (h - np.min(h)) / (np.max(h) - np.min(h))  # 正规化到 [0, 1]

        if r > gaussian_height // 2 and r < s[-2] - gaussian_height // 2 and c > gaussian_width // 2 and c < s[-1] - gaussian_width // 2:
            r_range = slice(max(0, r - gaussian_height // 2), min(s[-2], r + gaussian_height // 2 + 1))
            c_range = slice(max(0, c - gaussian_width // 2), min(s[-1], c + gaussian_width // 2 + 1))
            # if len(s) == 3:
            #     noise_matrix[:, r_range, c_range] = m * h
            # else:
            #     noise_matrix[r_range, c_range] = m * h
            if len(s) == 3:
                noise_matrix[:, r_range, c_range] = np.maximum(noise_matrix[:, r_range, c_range], m * h)
            else:
                noise_matrix[r_range, c_range] = np.maximum(noise_matrix[r_range, c_range], m * h)
    # plt.imshow(noise_matrix.squeeze(0), cmap='gray')
    # print(noise_matrix.max(),noise_matrix.min())
    # plt.colorbar()
    # plt.title('Uniform Noise Map')
    # plt.show()
    noise_map = np.tile(np.expand_dims(noise_matrix, axis=0), (1, 1, 1, 1))
    noise_map = torch.tensor(noise_map, dtype=torch.float32)
    del noise_matrix
    return noise_map
'''discriminator'''
def backward_D(netD, real, fake, criterion):
    """Calculate GAN loss for the discriminator

    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator

    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    # Real
    criterion = nn.BCEWithLogitsLoss()
    pred_real = netD(real)
    #print(pred_real)
    real_labels = torch.ones_like(pred_real)
    loss_D_real = criterion(pred_real, real_labels)
    # Fake
    pred_fake = netD(fake.detach())
    #print(pred_fake)
    fake_labels = torch.zeros_like(pred_fake)
    loss_D_fake = criterion(pred_fake, fake_labels)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    #("loss_D:{}".format(loss_D))
    loss_D.backward()
    return loss_D

'''PSNR'''
def batch_psnr(img, imclean, data_range):
    img_cpu = img.data.cpu().numpy()
    imgclean = imclean.data.cpu().numpy()
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
    return psnr / img_cpu.shape[0]

'''Dataloader'''
IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
def get_imagenames(seq_dir, pattern=None):
    files = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))
    if not pattern is None:
        ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
        files = ffiltered
        del ffiltered
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files
def open_sequence(stat, gt_seq_dir,noise_se q_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    files = get_imagenames(gt_seq_dir)
    gt_seq_list = []
    raw_seq_list = []
    print("\tOpen sequence in folder: ", gt_seq_dir)
    # 随机选择 num_samples 张图像
    Ground_truth_files = random.sample(files, max_num_fr)
    if stat == 'train':
        Raw_files = [fpath.replace("Groundtruth_pool", "Raw_pool") for fpath in Ground_truth_files]
    else:
        Raw_files = [fpath.replace("img", "raw_img") for fpath in Ground_truth_files]
    #for fpath in files[0:max_num_fr]:
    for fpath in Ground_truth_files:
        img, expanded_h, expanded_w = open_image(fpath, gray_mode=gray_mode, expand_if_needed=expand_if_needed,
                                                 expand_axis0=False)
        gt_seq_list.append(img)
    for fpath in Raw_files:
        img, expanded_h, expanded_w = open_image(fpath, gray_mode=gray_mode, expand_if_needed=expand_if_needed,
                                                 expand_axis0=False)
        raw_seq_list.append(img)
    gt_seq = np.stack(gt_seq_list, axis=0)
    raw_seq = np.stack(raw_seq_list, axis=0)
    return gt_seq, raw_seq, expanded_h, expanded_w


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    if not gray_mode:
        img = cv2.imread(fpath)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        img = np.expand_dims(img, axis=0)
    if expand_axis0:
        img = np.expand_dims(img, 0)
    expanded_h = False
    expanded_w = False
    sh_im = img.shape
    if expand_if_needed:
        if sh_im[-2] % 2 == 1:
            expanded_h = True
            if expand_axis0:
                img = np.concatenate((img, img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
            else:
                img = np.concatenate((img, img[:, -1, :][:, np.newaxis, :]), axis=1)
        if sh_im[-1] % 2 == 1:
            expanded_w = True
            if expand_axis0:
                img = np.concatenate((img, img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
            else:
                img = np.concatenate((img, img[:, :, -1][:, :, np.newaxis]), axis=2)
    if normalize_data:
        # img = (img / 255).clip(0,255.)  #.astype(np.uint8)
        # img = img.clip(0, 1.).astype(np.float32)
        # #img = (img / 255).astype(np.float32)
        # #img = normalize(img)
        img = (img - img.min()) / (img.max() - img.min())
        img = img.clip(0, 1)
        img = img.astype(np.float32)
    return img, expanded_h, expanded_w

VALSEQPATT = '*'
class TrainDataset(Dataset):
    def __init__(self, stat, gt_dir=None, raw_dir=None, gray_mode=True, num_input_frames=10):
        self.gray_mode = gray_mode
        # gt_seqs_dirs = sorted(glob.glob(os.path.join(val_gt_dir, VALSEQPATT)))
        # noisy_seqs_dirs = sorted(glob.glob(os.path.join(val_raw_dir, VALSEQPATT)))
        # assert len(gt_seqs_dirs) == len(
        #     noisy_seqs_dirs), "Mismatch in number of sequences between ground truth and noisy directories."
        # gt_sequences = []
        # noisy_sequences = []
        # for gt_seq_dir, noisy_seq_dir in zip(gt_seqs_dirs, noisy_seqs_dirs):
        #     gt_seq,noisy_seq,_,_ = open_sequence(gt_seq_dir,noisy_seq_dir, gray_mode, expand_if_needed=False, max_num_fr=num_input_frames)
        #     #a = noisy_seq-gt_seq
        #     gt_sequences.append(gt_seq)
        #     noisy_sequences.append(noisy_seq)
        gt_sequences, noisy_sequences, _, _ = open_sequence(stat, gt_dir, raw_dir, gray_mode, expand_if_needed=False,max_num_fr=num_input_frames)
        self.gt_sequences = gt_sequences
        self.noisy_sequences = noisy_sequences

    def __getitem__(self, index):
        gt_seq = torch.from_numpy(self.gt_sequences[index])
        noisy_seq = torch.from_numpy(self.noisy_sequences[index])
        return gt_seq, noisy_seq

    def __len__(self):
        return len(self.gt_sequences)

'''切割成小块'''

def split_image(image, patch_size, overlap=16):  # overlap重叠的像素数
    batch_size, channels, height, width = image.size()
    stride = patch_size - overlap  #图像大小减去重叠部分
    patches = image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  #在给定维度上对张量进行展开第2维和第3维度
    patches = patches.contiguous().view(batch_size, -1, patch_size, patch_size)    #确保张量在内存中是连续的
    return patches, stride


# def combine_patches(patches, image_size, patch_size, stride):
#     batch_size, channels, height, width = image_size
#     device = patches.device
#     output = torch.zeros(image_size).to(device)
#     count = torch.zeros(image_size).to(device)
#
#     patch_index = 0
#     for i in range(0, height - patch_size + 1, stride):
#         for j in range(0, width - patch_size + 1, stride):
#             output[:, :, i:i + patch_size, j:j + patch_size] += patches[patch_index]
#             count[:, :, i:i + patch_size, j:j + patch_size] += 1
#             patch_index += 1
#
#     # Replace zero values in count with a small value
#     count = torch.where(count == 0, torch.tensor(1e-10).to(device), count)
#     combined = output / count
#
#     # Check and handle NaN and Inf values
#     if torch.isnan(combined).any():
#         print("NaN values found in combined tensor before fixing.")
#         combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
#     if torch.isinf(combined).any():
#         print("Infinite values found in combined tensor before fixing.")
#         combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
#
#     combined = torch.clamp(combined, 0, 1)
#
#     return combined
def combine_patches(patches, image_size, patch_size, stride):
    batch_size, channels, height, width = image_size
    device = patches.device

    # 初始化输出图像和计数张量
    output = torch.zeros((batch_size, channels, height, width), device=device)
    count = torch.zeros((batch_size, channels, height, width), device=device)

    patch_index = 0
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            output[:, :, i:i + patch_size, j:j + patch_size] += patches[patch_index]
            count[:, :, i:i + patch_size, j:j + patch_size] += 1
            patch_index += 1

    # 处理 count 张量中的零值，避免除以零
    count = torch.where(count == 0, torch.tensor(1e-10, device=device), count)
    combined = output / count

    # 检查并处理 NaN 和 Inf 值
    if torch.isnan(combined).any():
        print("NaN values found in combined tensor before fixing.")
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)
    if torch.isinf(combined).any():
        print("Infinite values found in combined tensor before fixing.")
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=0.0)

    # 将结果裁剪到 0 和 1 之间
    combined = torch.clamp(combined, 0, 1)
    return combined

'''train_model'''
def train_model(viz, writer, model, net_D, net_F, net_S, train_loader, criterion, criterion_G, optimizer_G, optimizer_D, num_epochs, patch_size, scheduler):
    all_epoch_losses = []
    save_dir = '/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Addnoise/train_out'
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_loss_D = 0.0
        min_loss = 999999.0
        for data in train_loader:
            gt_seq, noisy_seq = data
            # hist, bin_edges = np.histogram(noisy_seq.squeeze(0).squeeze(0).numpy(), bins=256, range=(0, 1))
            # # 绘制直方图
            # plt.figure(figsize=(10, 5))
            # # 绘制直方图
            # plt.subplot(1, 2, 1)
            # plt.plot(bin_edges[0:-1], hist, lw=2)
            # plt.title('Histogram')
            # plt.xlabel('Pixel value')
            # plt.ylabel('Frequency')
            # # 绘制图像
            # plt.subplot(1, 2, 2)
            # plt.imshow(noisy_seq.squeeze(0).squeeze(0).numpy(), cmap='gray')
            # plt.title('Image')
            # plt.axis('off')
            # plt.show()
            gt_seq, noisy_seq = transform(gt_seq, noisy_seq)
            gt_seq = gt_seq.cuda()
            noisy_seq = noisy_seq.cuda()
            numframes, C, H, W = gt_seq.shape
            #padding
            # expanded_h = (128 - H % 128) % 128
            # expanded_w = (128 - W % 128) % 128
            # padexp = (0, expanded_w, 0, expanded_h)
            # gt_seq = F.pad(input=gt_seq.squeeze(0), pad=padexp, mode='constant')
            # noisy_seq = F.pad(input=noisy_seq.squeeze(0), pad=padexp, mode='constant')  #反射填充-reflect；常数填充-constant
            # numframes, C, H, W = noisy_seq.shape
            #split
            # gt_seq_patchs, stride = split_image(gt_seq, patch_size)
            # patch_means = gt_seq_patchs.var(dim=(2, 3))
            # for i in range(gt_seq_patchs.shape[0]):
            #     for j in range(gt_seq_patchs.shape[1]):
            #         if patch_means[i,j] <=0.0001:
            #             gt_seq_patchs = torch.cat((gt_seq_patchs[:, :i, :, :], gt_seq_patchs[:, i + 1:, :, :]), dim=1)
            # # 可视化
            # for i in range(gt_seq_patchs.shape[0]):
            #     for j in range(gt_seq_patchs.shape[1]):
            #         image = gt_seq_patchs[i][j]#.squeeze(0)
            #         viz.image(image.cpu().numpy(), opts={'title': f'Image {i + 1}_{j + 1}mean{patch_means[i,j]}'})
            # # 打印结果
            # print(patch_means.shape)
            # print(patch_means)
            # plt.imshow(gt_seq.cpu()[0].squeeze(0), cmap='gray')
            # plt.colorbar()
            # plt.title('Uniform Noise Map')
            # plt.show()
            # loss
            # noisy_map_real = generate_hotpixel_noise_map(torch.max(gt_seq), (C, H, W)).cuda()
            # weight_mask = create_weight_mask(noisy_seq.cpu().numpy(), threshold=0.07, alpha=5, beta=2)#纯黑区域权重和其他区域# 生成权重掩码
            # weight_mask = torch.tensor(weight_mask).cuda()
            # loss_G = weighted_bce_loss(noisy_out, noisy_seq, weight_mask)
            #noisy_out_seq = model(gt_seq_patchs)
            #trans_normalize = transforms.Normalize((0.5,), (0.5,))
            #noisy_out_seq = model(trans_normalize(gt_seq))
            #noisy_out_seq = noisy_out_seq * 0.5 + 0.5
            #gt_seq_a = torch.cat((a.cuda(),gt_seq),1)
            #gt_seq_transformed = transforms.RandomResizedCrop(256)(gt_seq)
            crop_size = (256,256)
            optimizer_G.zero_grad()
            img_loss = 0.0
            gt_seq_patchs, stride = split_image(gt_seq, patch_size)
            noise_seq_patchs, stride = split_image(noisy_seq, patch_size)
            for z in range(gt_seq_patchs.shape[1]):
                # random_crop = transforms.RandomResizedCrop(crop_size)
                # # 手动应用裁剪变换，获取裁剪的位置和大小
                # i,j,h,w = random_crop.get_params(gt_seq, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.))
                # def crop_image(img, i, j, h, w):
                #     """裁剪图像并返回裁剪后的部分"""
                #     return TF.crop(img, i, j, h, w)
                # gt_seq_cropped = crop_image(gt_seq[0, 0], i, j, h, w).unsqueeze(0) .unsqueeze(0)
                # noisy_seq_cropped = crop_image(noisy_seq[0, 0], i, j, h, w).unsqueeze(0) .unsqueeze(0)
                # def resize_image(img, size):
                #     """缩放图像到指定的大小"""
                #     return TF.resize(img, size)
                # gt_seq = resize_image(gt_seq_cropped, crop_size)  # 添加 batch 维度
                # noisy_seq = resize_image(noisy_seq_cropped, crop_size)   # 添加 batch 维度
                # #添加noise_map
                # # 设置图像大小
                # image_shape = (1, 256, 256)
                # # 生成泊松噪声图像
                # lam = 30  # 控制泊松噪声的强度
                # poisson_noise_image = np.random.poisson(lam=lam, size=image_shape)
                # # 生成稀疏掩码
                # sparsity = 0.05  # 控制稀疏度，0.01表示1%的像素位置有噪声
                # mask = np.random.choice([0, 1], size=image_shape, p=[1 - sparsity, sparsity])
                # # 应用稀疏掩码
                # sparse_poisson_noise_image = poisson_noise_image * mask
                # a = torch.tensor(sparse_poisson_noise_image).unsqueeze(0).to(torch.float32)
                # a = a / a.max()
                # gt_seq_a = torch.cat((a.cuda(), gt_seq), 1)
                # 判断是否有效（是否存在值大于 0 的像素）
                gt_seq = gt_seq_patchs[0,z].unsqueeze(0).unsqueeze(0)
                noisy_seq = noise_seq_patchs[0, z].unsqueeze(0).unsqueeze(0)
                if gt_seq.max().item() > 0:
                    # 如果有效，将其输入模型
                    noisy_out_seq = model(gt_seq)
                else:
                    print("非有效图像。")
                    continue
                #noisy_out_seq = model(gt_seq)
                # noisy_out_seq = F.interpolate(noisy_out_seq, size=(H,W), mode='bilinear', align_corners=False)
                # discriminator
                # n_critic = 5  # Number of times to train the discriminator per generator step
                # for _ in range(n_critic):
                #     set_requires_grad(net_D, True)
                #     optimizer_D.zero_grad()
                #     loss_D = backward_D(net_D, noisy_seq, noisy_out_seq, criterion)
                #     optimizer_D.step()
                # set_requires_grad(net_D, True)
                # optimizer_D.zero_grad()
                # loss_D = backward_D(net_D, noisy_seq, noisy_out_seq, criterion)
                # optimizer_D.step()
                #generator
                # set_requires_grad(net_D, False)
                '''loss'''
                noisy_out_seq_G = noisy_out_seq.repeat(1,3,1,1)
                per_noisy_out_seq = net_F(noisy_out_seq_G)
                noisy_seq_G = noisy_seq.repeat(1, 3, 1, 1)
                per_noisy_seq = net_F(noisy_seq_G)
                '''loss新'''
                loss_G_1 = net_F(noisy_out_seq_G,noisy_seq_G)[0]
                loss_G_1 = criterion(per_noisy_out_seq,per_noisy_seq)
                loss_G_2 = criterion_G(noisy_out_seq, noisy_seq)
                #visualize_feature_map(feature_map)
                # # 可视化
                # for i in range(per_noisy_seq.shape[0]):
                #     for j in range(per_noisy_seq.shape[1]):
                #         image = per_noisy_seq[i][j]#.squeeze(0)
                #         viz.image(image.cpu().numpy(), opts={'title': f'Image {i + 1}_{j + 1}'})
                # # 打印结果
                # plt.imshow(image.cpu()[0].squeeze(0), cmap='gray')
                # plt.colorbar()
                # plt.title('Uniform Noise Map')
                # plt.show()

                #loss_G_1 = criterion_G(per_noisy_out_seq, per_noisy_seq)
                #loss_G_1 = criterion_G(noisy_out_seq, noisy_seq)
                #out_D = net_D(noisy_out_seq)
                #real_labels = torch.ones_like(out_D)
                #loss_G_2 = criterion(out_D, real_labels)
                weight_1=0.8
                weight_2=0.2
                loss_G_0 = loss_G_1 * weight_1 + loss_G_2 * weight_2
                img_loss += loss_G_0
            img_loss.backward()
            optimizer_G.step()
            torch.cuda.empty_cache()
            #img_loss = img_loss + loss_G_0
            ''''#view mask
            weight_mask_patchs = split_image(weight_mask, patch_size)
            plt.imshow(kernel[0, 0].cpu().detach().numpy(), cmap='hot')
            plt.title(f'Weight Mask at Epoch {epoch + 1}')
            plt.axis('off')
            plt.show()'''
            epoch_loss +=  img_loss.item()
            loss_value = img_loss.item()
            #epoch_loss_D += loss_D.item()
            #loss_value_D = loss_D.item()
            if loss_value < min_loss:
                min_loss = loss_value
                out_noisy = noisy_out_seq
                #real_noisy_map = noisy_map_real
                best_noisy_seq = noisy_seq
                best_gt = gt_seq
            if epoch >= 0:
                i = 0
                vutils.save_image((noisy_out_seq[i]).cpu().detach(), os.path.join(save_dir, f'Noisy_out_{epoch + 1}.tiff'))
                vutils.save_image((noisy_seq[i]).cpu().detach().clip(0, 1.),
                                  os.path.join(save_dir, f'Image_noisy_{epoch + 1}.tiff'))
                vutils.save_image((gt_seq[i]).cpu().detach().clip(0, 1.),
                                  os.path.join(save_dir, f'Image_gt_{epoch + 1}.tiff'))
        '''view'''
        window_size = dict(width=400, height=272)
        all_epoch_losses.append(epoch_loss/100)
        i = 0
        out = matchmat2(out_noisy, best_noisy_seq)
        print(out)
        viz.image(noisy_out_seq[0].cpu().detach().numpy(), opts=dict(title=f'Noisy_out_{epoch+1}', **window_size))
        viz.image(noisy_seq[0].cpu().detach().numpy(), opts=dict(title=f'Image_noisy_{epoch+1}', **window_size))
        viz.image(gt_seq[0].cpu().detach().numpy(), opts=dict(title=f'Image_gt_{epoch+1}', **window_size))
        writer.add_image(f'Noisy_out', (out_noisy[i]).cpu().detach().clamp(0, 1.), epoch)
        writer.add_image(f'Image_noisy', (best_noisy_seq[i]).cpu().detach().clamp(0, 1.), epoch)
        writer.add_image(f'Image_gt', (best_gt[i]).cpu().detach().clamp(0, 1.), epoch)  #_{epoch + 1}
        # 关闭 SummaryWriter
        writer.close()
        scheduler.step()
        if(epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), f'logs/model_epoch_{epoch + 1}.pth')
        #torch.save(net_D.state_dict(), f'net_D_epoch_{epoch + 1}.pth')
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/num_epochs},Loss_D: {epoch_loss_D}")


def test_model(viz, writer, model, test_loader, criterion_G, patch_size, model_path):
    # 加载保存的模型权重
    torch.set_printoptions(threshold=float('inf'))
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式
    test_loss = 0.0
    ssim_sum = 0
    ssim_sum_list = []
    psnr_sum = 0
    psnr_sum_list = []
    j = 0
    ind = []
    with torch.no_grad():  # 禁用梯度计算
        i = 0
        for data in test_loader:
            gt_seq, noisy_seq = data
            gt_seq = gt_seq.cuda()
            noisy_seq = noisy_seq.cuda()
            numframes, C, H, W = gt_seq.shape
            #trans_normalize = transforms.Normalize((0.5,), (0.5,))
            # # 添加noise_map
            # # 设置图像大小
            # image_shape = (1, H, W)
            # # 生成泊松噪声图像
            # lam = 30  # 控制泊松噪声的强度
            # poisson_noise_image = np.random.poisson(lam=lam, size=image_shape)
            # # 生成稀疏掩码
            # sparsity = 0.05  # 控制稀疏度，0.01表示1%的像素位置有噪声
            # mask = np.random.choice([0, 1], size=image_shape, p=[1 - sparsity, sparsity])
            # # 应用稀疏掩码
            # sparse_poisson_noise_image = poisson_noise_image * mask
            # a = torch.tensor(sparse_poisson_noise_image).unsqueeze(0).to(torch.float32)
            # a = a / a.max()
            # gt_seq_a = torch.cat((a.cuda(), gt_seq), 1)
            noisy_out_seq = model(gt_seq)
            #noisy_out_seq = noisy_out_seq * 0.5 + 0.5
            # 损失计算
            loss_G = criterion_G(noisy_out_seq, noisy_seq)
            test_loss += loss_G.item()
            # 将灰度图像拼接在一起
            concatenated_images = torch.cat((noisy_out_seq, noisy_seq), dim=-1)  # 按宽度拼接
            # 将拼接后的图像转换为CPU并分离计算图
            concatenated_images = concatenated_images[0, 0].cpu().detach().numpy()
            # 保存拼接后的图像
            plt.imshow(concatenated_images, cmap='gray')
            plt.title('Concatenated Noisy Output and Noisy Input')
            plt.axis('off')
            plt.savefig(f'concatenated_images{i}.png')
            i+=1
            # 可视化
            window_size = dict(width=400, height=272)
            viz.image(noisy_out_seq.cpu().detach().numpy(), opts=dict(title=f'Noisy_out_{i}', **window_size))
            viz.image(noisy_seq.cpu().detach().numpy(), opts=dict(title=f'Image_noisy_{i}', **window_size))
            viz.image(gt_seq.cpu().detach().numpy(), opts=dict(title=f'Image_gt_{i}', **window_size))
            writer.add_image(f'Noisy_out_{i}', noisy_out_seq.squeeze(0).cpu().detach().clamp(0, 1.))
            writer.add_image(f'Image_noisy_{i}', noisy_seq.squeeze(0).cpu().detach().clamp(0, 1.))
            writer.add_image(f'Image_gt_{i}', gt_seq.cpu().squeeze(0).detach().clamp(0, 1.))  # _{epoch + 1}
            writer.close()
            #PSNR
            #print(noisy_out_seq.max())
            out = matchmat2(noisy_out_seq, noisy_seq)
            print(out)
            ssim_sum = ssim_sum + out[0]
            ssim_sum_list.append(out[0])
            psnr_sum = psnr_sum + out[1]
            psnr_sum_list.append(out[1])
            ind.append(i)
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}, Test Loss_D: {0}")
    return avg_test_loss
def main():
    optimizer=[]
    state = "train"
    gt_dir = "/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Groundtruth_pool_1"
    raw_dir = "/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Raw_pool_1"
    test_gt_dir = "/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/test_img"
    test_raw_dir = "/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/test_raw_img"
    writer = SummaryWriter(log_dir='/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Addnoise/tensorboard_logs')
    gray_mode = True
    viz = visdom.Visdom()
    viz.close(env=None)
    assert viz.check_connection(), "Visdom server not running. Please run 'python -m visdom.server'"
    torch.cuda.set_device(1)
    device = torch.device("cuda:1")  # 指定gpu1为主GPU
    torch.backends.cudnn.benchmark = True  # CUDNN optimization
    Train_dataset = TrainDataset(stat = 'train', gt_dir=gt_dir, raw_dir=raw_dir, gray_mode=gray_mode, num_input_frames=100)
    Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset, batch_size=1, shuffle=True)
    test_dataset = TrainDataset(stat = 'test', gt_dir=test_gt_dir, raw_dir=test_raw_dir, gray_mode=gray_mode, num_input_frames=6)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    reFastDVDnet = ResnetGenerator().to(device)
    #reFastDVDnet = ReFastDVDnet().to(device)
    reFastDVDnet = init_net(reFastDVDnet)
    model = nn.DataParallel(reFastDVDnet, device_ids=[1, 0], output_device=1)
    #reFastDVDnet = SMNet(in_channels=1, out_channels=1, num_srg=3, num_mab=2, channels=8).to(device)
    #model = nn.DataParallel(reFastDVDnet, device_ids=[1, 0], output_device=1)
    lr = 0.001
    lr_D = 0.001
    #(reFastDVDnet)
    net_D = NLayerDiscriminator(1,nf=64,n_layers=3).to(device)
    net_D = nn.DataParallel(net_D, device_ids=[1, 0], output_device=1)
    net_F = define_F().to(device)
    net_F = nn.DataParallel(net_F, device_ids=[1, 0], output_device=1)
    # net_F = LossNetwork().to(device)
    # net_F.eval()  # No need to train
    # net_F = nn.DataParallel(net_F, device_ids=[1, 0], output_device=1)
    net_S = SSIM().to(device)
    net_S = nn.DataParallel(net_S, device_ids=[1, 0], output_device=1)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr_D)
    optimizer_G = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.1)
    optimizer.append(optimizer_G)
    optimizer.append(optimizer_D)
    criterion = nn.MSELoss() #reduction='sum'
    criterion_G = nn.L1Loss()
    num_epochs = 300
    patch_size = 256
    torch.cuda.empty_cache()
    if state == "train":
        train_model(viz, writer, model, net_D, net_F, net_S, Train_loader, criterion, criterion_G, optimizer_G, optimizer_D, num_epochs, patch_size, scheduler)
    else:
        model_path = '/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Addnoise/logs/model_epoch_300.pth'
        test_model(viz, writer,model, test_loader, criterion_G, patch_size, model_path)

if __name__ == "__main__":
    main()
