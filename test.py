#!/bin/sh
"""
Denoise all the sequences existent in a given folder using FastDVDnet.
"""
import os
import glob
import argparse
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import tifffile as tiff
import numpy as np
import torch.nn as nn
from data.util import *
from model.models_unsupervised import UnsupervisedDVDnet
from fastdvdnet import denoise_seq_fastdvdnet, denoise_seq_fastdvdnet_all
from data.util import batch_psnr, init_logger_test, \
    variable_to_cv2_image, remove_dataparallel_wrapper, open_sequence_1, close_logger

NUM_IN_FR_EXT = 5  # temporal size of patch
MC_ALGO = 'DeepFlow'  # motion estimation algorithm
OUTIMGEXT = '.tiff'  # output images format


def save_out_seq(seqnoisy, seqclean, renframes, save_dir, sigmaval, suffix, save_noisy):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqclean.size()[0]
    seqclean = (seqclean + 1.) / 2.
    renframes = (renframes + 1.) / 2.
    for idx in range(seq_len):
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir, \
                                  ('n{}_{}').format(sigmaval, idx) + fext)
        reout_name = os.path.join(save_dir, \
                                  ('n{}_FastDVDnet_{}_{}').format(sigmaval, idx, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir, \
                                    ('n{}_FastDVDnet_{}').format(sigmaval, idx) + fext)
        else:
            out_name = os.path.join(save_dir, \
                                    ('n{}_FastDVDnet_{}_{}').format(sigmaval, suffix, idx) + fext)

        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        reout_img = variable_to_cv2_image(renframes[idx].unsqueeze(dim=0))
        cv2.imwrite(reout_name, reout_img)
        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)

def save_out_seq_all(seqnoisy, seqclean, renframes, save_dir, sigmaval, suffix, save_noisy, beam):
    """Saves the denoised and noisy sequences under save_dir
    """
    seq_len = seqclean.size()[0]
    seqclean = (seqclean + 1.) / 2.
    renframes = (renframes + 1.) / 2.
    for idx in range(seq_len):
        # Build Outname
        fext = OUTIMGEXT
        noisy_name = os.path.join(save_dir, \
                                  ('n{}_{}').format(sigmaval, idx) + fext)
        reout_name = os.path.join(save_dir, \
                                  ('n{}_FastDVDnet_{}_{}').format(sigmaval, idx, idx) + fext)
        if len(suffix) == 0:
            out_name = os.path.join(save_dir, \
                                    ('{}').format(beam)+ fext)
        else:
            out_name = os.path.join(save_dir, \
                                    ('{}').format(beam)+ fext)

        # Save result
        if save_noisy:
            noisyimg = variable_to_cv2_image(seqnoisy[idx].clamp(0., 1.))
            cv2.imwrite(noisy_name, noisyimg)

        reout_img = variable_to_cv2_image(renframes[idx].unsqueeze(dim=0))
        cv2.imwrite(reout_name, reout_img)
        outimg = variable_to_cv2_image(seqclean[idx].unsqueeze(dim=0))
        cv2.imwrite(out_name, outimg)


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re


def natural_sort_key(s):
    """提取字符串中的数字用于自然排序"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def load_tif_image(tif_file):
    img = cv2.imread(tif_file, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, axis=2)
    img, S = normalize(img, 20)
    return tif_file, img, S


def load_all_tif_images(base_dir, sub_dirs1, max_workers=8):
    all_images1 = []
    total_count = 0
    # sub_dirs1 = [d for d in sorted(os.listdir(base_dir), key=natural_sort_key)
    #              if os.path.isdir(os.path.join(base_dir, d))
    #              and d.startswith("beam1") and "_cp_" in d]
    # sub_dirs2 = [d for d in sorted(os.listdir(base_dir), key=natural_sort_key)
    #              if os.path.isdir(os.path.join(base_dir, d))
    #              and d.startswith("beam2") and "_cp_" in d]

    print(f"共找到 {len(sub_dirs1)} 个文件夹。")

    for sub_dir in sub_dirs1:
        sub_path = os.path.join(base_dir, sub_dir)
        tif_files = sorted(glob.glob(os.path.join(sub_path, "*.tif")))
        if not tif_files:
            continue
        print(f"\n正在读取文件夹：{sub_dir} ({len(tif_files)} 张)")
        folder_images = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(load_tif_image, f): f for f in tif_files}
            # tqdm 进度条
            for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
                try:
                    folder_images.append(future.result())
                except Exception as e:
                    print(f"无法读取图像: {futures[future]}, 错误: {e}")

        all_images1.extend(folder_images)
        total_count += len(folder_images)
        print(f"文件夹 {sub_dir} 读取完成，共 {len(folder_images)} 张。累计 {total_count} 张。")
    # for sub_dir in sub_dirs2:
    #     sub_path = os.path.join(base_dir, sub_dir)
    #     tif_files = sorted(glob.glob(os.path.join(sub_path, "*.tif")))
    #     if not tif_files:
    #         continue
    #
    #     print(f"\n正在读取文件夹：{sub_dir} ({len(tif_files)} 张)")
    #     folder_images = []
    #
    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         futures = {executor.submit(load_tif_image, f): f for f in tif_files}
    #         # tqdm 进度条
    #         for future in tqdm(as_completed(futures), total=len(futures), ncols=80):
    #             try:
    #                 folder_images.append(future.result())
    #             except Exception as e:
    #                 print(f"无法读取图像: {futures[future]}, 错误: {e}")
    #     all_images2.extend(folder_images)
    #     total_count += len(folder_images)
    #     print(f"文件夹 {sub_dir} 读取完成，共 {len(folder_images)} 张。累计 {total_count} 张。")
    print(f"\n全部读取完成，beam总计 {len(all_images1)} 张")
    return all_images1  # , all_images2


def load_all_tif_images_cached(base_dir, beam_list, index,
                               cache_path="/home/fastdvdnet-master-unsupervised/fastdvdnet-master",
                               max_workers=8):
    cache_path = os.path.join(cache_path, index)
    cache_path = os.path.join(cache_path, 'cached_images.npz')
    print(cache_path)
    if os.path.exists(cache_path):
        print(f"检测到缓存文件 {cache_path}，直接加载缓存...")
        data = np.load(cache_path, allow_pickle=True)
        return data["all_images1"]
    print("未检测到缓存，开始读取 .tif 文件...")
    all_images1 = load_all_tif_images(base_dir, beam_list, max_workers=max_workers)
    # 保存缓存
    np.savez_compressed(cache_path, all_images1=np.array(all_images1, dtype=object))
    print(f"图像已缓存到 {cache_path}")
    return all_images1


import os, re


def natural_sort_key(s):
    """将字符串中的数字按数值排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def test_fastdvdnet(**args):
    """Denoises all sequences present in a given folder. Sequences must be stored as numbered
    image sequences. The different sequences must be stored in subfolders under the "test_path" folder.

    Inputs:
        args (dict) fields:
            "model_file": path to model
            "test_path": path to sequence to denoise
            "suffix": suffix to add to output name
            "max_num_fr_per_seq": max number of frames to load per sequence
            "noise_sigma": noise level used on test set
            "dont_save_results: if True, don't save output images
            "no_gpu": if True, run model on CPU
            "save_path": where to save outputs as png
            "gray": if True, perform denoising of grayscale images instead of RGB
    """
    # Start time
    start_time = time.time()
    # If save_path does not exist, create it
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])
    logger = init_logger_test(args['save_path'])

    # Sets data type according to CPU or GPU modes
    if args['cuda']:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")  # 指定gpu1为主GPU
        torch.backends.cudnn.benchmark = True  # CUDNN optimization
    else:
        device = torch.device('cpu')
    # Create model
    model_temp = UnsupervisedDVDnet().to(device)  # num_input_frames=NUM_IN_FR_EXT
    # Load saved weights
    state_temp_dict = torch.load(args['model_file'], map_location=device)
    if args['cuda']:
        device_ids = [0]
        model_temp = nn.DataParallel(model_temp, device_ids=[0], output_device=0)
    else:
        # CPU mode: remove the DataParallel wrapper
        state_temp_dict = remove_dataparallel_wrapper(state_temp_dict)
    model_temp.load_state_dict(state_temp_dict)
    # Sets the model in evaluation mode (e.g. it removes BN)
    model_temp.eval()

    with torch.no_grad():
        # process data
        if args['seq']:
            print(args['test_clin_path'])
            list_all = sorted(
                [d for d in os.listdir(args['test_clin_path'])
                 if os.path.isdir(os.path.join(args['test_clin_path'], d))],
                key = natural_sort_key
            )
            mid = len(list_all) // 2
            list_beam1 = list_all[:mid]
            list_beam2 = list_all[mid:]
            #list_beam1 = list_beam2
            # beam1 = load_all_tif_images_cached(args['test_clin_path'],list_beam1, index = 'beam1')
            # beam2 = load_all_tif_images_cached(args['test_clin_path'],list_beam2, index = 'beam2')
            for idx in range(len(list_beam1)):
                beam = list_beam1[idx]
                beam_path = os.path.join(args['test_clin_path'], beam)
                tif_files = sorted(glob.glob(os.path.join(beam_path, "*.tif")))
                if len(tif_files) < 5:
                    center_num_images = len(tif_files)
                    if idx - 1>=0 and idx + 1<len(list_beam1):
                        beam0 = list_beam1[idx - 1]
                        beam1 = list_beam1[idx + 1]
                        beam_path0 = os.path.join(args['test_clin_path'], beam0)
                        tif_files0 = sorted(glob.glob(os.path.join(beam_path0, "*.tif")))
                        beam_path1 = os.path.join(args['test_clin_path'], beam1)
                        tif_files1 = sorted(glob.glob(os.path.join(beam_path1, "*.tif")))
                        tif_files = tif_files0 + tif_files + tif_files1
                        center_start_index = len(tif_files0)
                    elif idx - 1<0 :
                        beam1 = list_beam1[idx + 1]
                        beam_path1 = os.path.join(args['test_clin_path'], beam1)
                        tif_files1 = sorted(glob.glob(os.path.join(beam_path1, "*.tif")))
                        tif_files = tif_files + tif_files1
                        center_start_index = 0
                    if idx - 1>=0 and idx + 1>=len(list_beam1):
                        beam0 = list_beam1[idx - 1]
                        beam_path0 = os.path.join(args['test_clin_path'], beam0)
                        tif_files0 = sorted(glob.glob(os.path.join(beam_path0, "*.tif")))
                        tif_files = tif_files0 + tif_files
                        center_start_index = len(tif_files0)
                    all_images = []
                    all_S = []
                    file_names = []

                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(load_tif_image, f): f for f in tif_files}
                        for future in as_completed(futures):
                            try:
                                tif_file, img, S = future.result()
                                all_images.append(img)
                                all_S.append(S)
                                file_names.append(tif_file)
                            except Exception as e:
                                print(f"读取失败: {futures[future]}, 错误: {e}")

                    seq = torch.stack(all_images, dim=0).cuda()
                    S = torch.stack(all_S, dim=0).cuda()
                    denframes, renframes = denoise_seq_fastdvdnet_all(seq=seq, \
                                                                      S=S, \
                                                                      temp_psz=NUM_IN_FR_EXT, \
                                                                      model_temporal=model_temp)

                    denframes = torch.sum(denframes[center_start_index:center_start_index+center_num_images,:,:,:], dim=0)/ denframes.size(0)
                    if not args['dont_save_results']:
                        save_out_seq_all(seq, denframes, renframes, args['save_path'], \
                                     int(args['noise_sigma'] * 255), args['suffix'], args['save_noisy'],beam)
                    print(len(tif_files))
                else:
                    all_images = []
                    all_S = []
                    file_names = []
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(load_tif_image, f): f for f in tif_files}
                        for future in as_completed(futures):
                            try:
                                tif_file, img, S = future.result()
                                all_images.append(img)
                                all_S.append(S)
                                file_names.append(tif_file)
                            except Exception as e:
                                print(f"读取失败: {futures[future]}, 错误: {e}")
                    seq = torch.stack(all_images, dim=0).cuda()
                    S = torch.stack(all_S, dim=0).cuda()
                    denframes, renframes = denoise_seq_fastdvdnet_all(seq=seq, \
                                                                      S=S, \
                                                                      temp_psz=NUM_IN_FR_EXT, \
                                                                      model_temporal=model_temp)
                    #save_path = os.path.join(args['save_path'], beam)
                    denframes = torch.sum(denframes, dim=0)/ denframes.size(0)
                    if not args['dont_save_results']:
                        save_out_seq_all(seq, denframes, renframes, args['save_path'], \
                                     int(args['noise_sigma'] * 255), args['suffix'], args['save_noisy'],beam)
                    print(len(tif_files))

            seq_time = time.time()
        else:
            seq, S, _, _ = open_sequence_1(args['test_path'], \
                                           args['gray'], \
                                           expand_if_needed=False, \
                                           max_num_fr=args['max_num_fr_per_seq'])
            seq_gt, S_gt, _, _ = open_sequence_1(args['test_path_gt'], \
                                                 args['gray'], \
                                                 expand_if_needed=False, \
                                                 max_num_fr=args['max_num_fr_per_seq'])
            seq = seq.to(device)
            S = S.to(device)
            seq_time = time.time()
            denframes, renframes = denoise_seq_fastdvdnet(seq=seq, \
                                                          S=S, \
                                                          seq_gt=seq_gt, \
                                                          temp_psz=NUM_IN_FR_EXT, \
                                                          model_temporal=model_temp)
            # Save outputs
            if not args['dont_save_results']:
                # Save sequence
                save_out_seq(seq, denframes, renframes, args['save_path'], \
                             int(args['noise_sigma'] * 255), args['suffix'], args['save_noisy'])

    # Compute PSNR and log it
    stop_time = time.time()
    loadtime = (seq_time - start_time)
    runtime = (stop_time - seq_time)
    seq_length = seq.size()[0]
    logger.info("Finished denoising {}".format(args['test_path']))
    logger.info("\tDenoised {} frames in {:.3f}s, loaded seq in {:.3f}s". \
                format(seq_length, runtime, loadtime))

    # close logger
    close_logger(logger)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,\
    					default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/logs/net.pth", \
    					help='path to model of the pretra ined denoiser')
    # parser.add_argument("--model_file", type=str, \
    #                     default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net0.pth", \
    #                     help='path to model of the pretrained denoiser')
    # parser.add_argument("--model_file", type=str, \
    #                     default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net.pth", \
    #                     help='path to model of the pretrained denoiser')
    # parser.add_argument("--model_file", type=str, \
    #                     default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net0918_1.pth", \
    #                     help='path to model of the pretrained denoiser')
    # /home/yuhuizhen_1/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/clinical_sequence/2
    parser.add_argument("--test_path", type=str,
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/0410_1010", \
                        help='path to sequence to denoise')
    parser.add_argument("--test_clin_path", type=str,
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_all/cherenkov_cps_gxz", \
                        help='path to clinical sequence to denoise')
    parser.add_argument("--test_path_gt", type=str,
                        default="/home/fastdvdnet-master-unsupervised/fastdvdnet-master/test_seq/0410_1010", \
                        help='path to sequence to denoise')
    parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=32, help='max number of frames to load per sequence')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    parser.add_argument("--save_path", type=str,
                        default='/home/fastdvdnet-master-unsupervised/fastdvdnet-master/results', \
                        help='where to save outputs as png')
    parser.add_argument("--gray", action='store_true', default=True, \
                        help='perform denoising of grayscale images instead of RGB')
    parser.add_argument("--seq", action='store_true', default=False, \
                        help='是否cp序列')
    argspar = parser.parse_args()
    argspar.noise_sigma /= 255.
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    test_fastdvdnet(**vars(argspar))
