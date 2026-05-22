import argparse
import yaml
from tqdm import tqdm
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from model.models_unsupervised import UnsupervisedDVDnet
from concurrent.futures import ThreadPoolExecutor
import math
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from CEM_Demo.ModelZoo.utils import Tensor2PIL
from CEM_Demo.ModelZoo import load_model,load_denoise_model,load_derain_model
from CEM_Demo.SaliencyModel.utils import pil_to_cv2, calculate_psnr
from CEM_Demo.utils import Get_Intervention_Dataset, load_psnr_data, InterventionDataset
from CEM_Demo.utils import RainGenerator,extract_first_number_from_image_name,LossNetwork

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.data)  # Use len() instead of .shape[0]

    def __getitem__(self, idx):
        img = self.data[idx]
        img_tensor = self.transform(img)
        return img_tensor


def main(config):
    # Load the configuration
    Imagelist = config['Imagelist']
    Modellist = config['Modellist']
    GtImage = config['GtImage']
    prefix = config['prefix']
    patch_path = config['patch_path']
    coarse_num = config['coarse_num']
    fine_num = config['fine_num']
    ROI_size = config['ROI_size']
    h = config['h']
    w = config['w']
    mask_block_size = config['mask_block_size']
    load_previous = config['load_previous']
    tol = config['tol']
    sr = config['sr']
    task = config['task']
    TextImg_path = config['TextImg_path']
    OutImg_path = config['OutImg_path']
    mask_block_size = mask_block_size // sr

    # Iterate over the models and images
    for Model_name in Modellist:
        batch_size = config['batch_size_dict'].get(Model_name, 100)
        if task == 'SR':
            model = load_model(f'{Model_name}@Base').eval().cuda()
        elif task == 'DN':  #模型加载
            #model = load_denoise_model(f'{Model_name}@Base').eval().cuda()
            device = torch.device("cuda:0")  # 指定gpu1为主GPU
            model_path = "/mnt/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/model_1/net.pth"
            model_temp = UnsupervisedDVDnet().to(device)
            state_temp_dict = torch.load(model_path, map_location=device)
            model = nn.DataParallel(model_temp, device_ids=[0], output_device=0)
            model.load_state_dict(state_temp_dict)
        elif task == 'DR':
            model = load_derain_model(f'{Model_name}@Base').eval().cuda()
        trans_totensor = transforms.ToTensor()
        for Image_name in Imagelist:
            gt = cv2.imread(GtImage, cv2.IMREAD_UNCHANGED)
            #gt = np.expand_dims(gt, axis=2)
            gt = gt/255.0/500
            gt = gt.astype(np.float32)
            gt = trans_totensor(gt).squeeze(0)
            basename, ext = os.path.splitext(Image_name)  # ('image0002', '.tif')
            prefix = basename[:-4]  # 'image'
            num_str = basename[-4:]  # '0002'
            start_idx = int(num_str)
            #读取连续5帧作为输入
            A = []
            for i in range(start_idx, start_idx + 5):
                fname = f"{prefix}{i:04d}{ext}"
                img_path = os.path.join(TextImg_path, fname)
                hr_pil = Image.open(img_path)
                img = np.asarray(hr_pil)
                if img.ndim == 2:
                    img = np.array(hr_pil, dtype=np.uint16)
                    img = np.expand_dims(img, axis=2)
                img = img.astype(np.float32)
                tensor_img = trans_totensor(img).unsqueeze(0)  # [1, C, H, W]
                A.append(tensor_img)
            img = torch.cat(A, dim=1)
            sizex, sizey = hr_pil.size
            hr_pil = hr_pil.crop((0, 0, sizex - sizex % sr, sizey - sizey % sr))
            sizex, sizey = hr_pil.size
            pil2tensor = transforms.ToTensor()
            if task == 'SR':
                lr_pil = hr_pil.resize((sizex // sr, sizey // sr), Image.BICUBIC)
                lr_pil.save(os.path.join(TextImg_path, f'{Image_name}-lr.png'))
                lr_origin = pil2tensor(lr_pil)
                beta = None
                rain = None
                input_image = lr_pil
            elif task == 'DN':
                img = np.array(img / 400, dtype=float)
                #img = img * 2-1
                beta = None
                rain = None
                input_image = img

            elif task == 'DR':
                first_number = extract_first_number_from_image_name(Image_name)
                RG = RainGenerator(first_number)
                beta, rain, img_rain = RG(img)
                lr_pil = Image.fromarray(img_rain)
                lr_pil.save(os.path.join(TextImg_path, f'{Image_name}-rain.png'))
                lr_origin = pil2tensor(lr_pil)
                input_image = hr_pil

            #lr_origin = lr_origin.unsqueeze(0).cuda()
            num_blocks_size_h = math.ceil(input_image.shape[2] / mask_block_size)
            num_blocks_size_w = math.ceil(input_image.shape[3] / mask_block_size)
            #得到原始输入 baseline结果
            n = 1
            S = torch.full((1, 1), n, dtype=torch.float32)
            input_image = torch.from_numpy(input_image).float()
            input = input_image*2-1
            sr_origin = model(input,S)
            sr_origin = (sr_origin+1)/2
            sr_for_retangle = sr_origin.detach().clone()
            sr_for_retangle = Tensor2PIL(torch.clamp(sr_for_retangle, min=0., max=1.))
            sr_origin = sr_origin.squeeze(0).detach().cpu().numpy()
            sr_origin = np.transpose(sr_origin, (1, 2, 0))#.clip(0, 1)

            output_folder = os.path.join(f'./{task}-CEM', f'{prefix}-R{ROI_size}M{mask_block_size}', Model_name)
            os.makedirs(output_folder, exist_ok=True)   #'./DN-CEM/image-R128M8/DnCNN'

            # Define the image output paths
            origin_output_path = os.path.join(output_folder, f"{task}_{Image_name}-Origin.png") #'./DN-CEM/image-R128M8/DnCNN/DN_image0093.tiff-Origin.png'
            rectangle_folder = os.path.join(output_folder, Image_name) #'./DN-CEM/image-R128M8/DnCNN/image0093.tiff'
            rectangle_output_file = os.path.join(rectangle_folder, "rectangle-result-origin.png") #'./DN-CEM/image-R128M8/DnCNN/image0093.tiff/rectangle-result-origin.png'
            gt_rectangle_output_file = os.path.join(rectangle_folder, "gt_rectangle-result-origin.png")

            # 保存baseline结果
            sr_origin_uint8 = np.uint8(np.clip(sr_origin * 255.0, 0, 255))
            cv2.imwrite(origin_output_path, sr_origin_uint8.squeeze())
            # 在image上画一个矩形，再保存
            draw_img = pil_to_cv2(sr_for_retangle)
            cv2.rectangle(draw_img, (w, h), (w + ROI_size, h + ROI_size), 255, 2)#画一条绿色框线
            os.makedirs(rectangle_folder, exist_ok=True)
            cv2.imwrite(rectangle_output_file, draw_img)

            #计算baseline去噪结果的PSNR
            img = img.sum(axis=1)
            img = np.transpose(img, (1, 2, 0))
            gt = gt.unsqueeze(-1).numpy()
            # 保存gt结果
            gt_origin_uint8 = np.uint8(np.clip(gt * 255.0, 0, 255))
            cv2.imwrite(gt_rectangle_output_file, gt_origin_uint8.squeeze())
            # 在gt上画一个矩形，再保存
            gt_tensor = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0)
            gt_tensor = Tensor2PIL(torch.clamp(gt_tensor, min=0., max=1.))
            draw_img = pil_to_cv2(gt_tensor)
            cv2.rectangle(draw_img, (w, h), (w + ROI_size, h + ROI_size), 255, 2)
            cv2.imwrite(gt_rectangle_output_file, draw_img)
            GT_ROI = gt[h: h + ROI_size, w: w + ROI_size, :]
            sr_origin_ROI = sr_origin[h: h + ROI_size, w: w + ROI_size, :]
            # net_F = LossNetwork()
            # net_F = net_F.eval()
            # a = torch.from_numpy(sr_origin_ROI).float().squeeze(2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            # b = torch.from_numpy(GT_ROI).float().squeeze(2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            #LPIPS_loss = net_F(torch.from_numpy(sr_origin_ROI).float().squeeze(2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), torch.from_numpy(GT_ROI).squeeze(2).unsqueeze(0).unsqueeze(0).float().repeat(1, 3, 1, 1))
            b = calculate_psnr(sr_origin_ROI, GT_ROI)
            psnr_origin = b    #LPIPS_loss.numpy()
            print(psnr_origin)
            print(f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR-Origin.mat')
            sio.savemat(
                f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR-Origin.mat',
                {'PSNR': psnr_origin})

            # load previous data PSNR_C:粗扰动；PSNR_F：细扰动
            PSNR_C, PSNR_F = load_psnr_data(prefix, ROI_size, mask_block_size, Model_name, Image_name,
                                            coarse_num, fine_num,
                                            load_previous,num_blocks_size_h,num_blocks_size_w,task)
            coarse_coordinates_dict = {}
            fine_coordinates_dict = {}

            #创建粗粒度block坐
            for z in range(5):
                for i in range(num_blocks_size_h):
                    for j in range(num_blocks_size_w):
                        if PSNR_C[z,i, j, -1] != 0:  # block 在 coarse 的最后两个扰动条件下的 PSNR 值  and PSNR_C[i, j, -2] != 0
                            continue
                        for k in range(coarse_num):
                            for z in range(5):
                                coarse_coordinates_dict[len(coarse_coordinates_dict)] = ((i, j),k,z)
            dataset = InterventionDataset(coarse_coordinates_dict, input_image, mask_block_size, patch_path, task=task,
                                          beta=beta, rain=rain)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            ith_batch = 0
            with torch.no_grad():
                for lr,index in tqdm(dataloader):
                    (X,Y,K,Z) = index
                    s = torch.full((lr.shape[0], 1), n, dtype=torch.float32)
                    print(f'Coarse stage: {patch_path}  {Image_name}/{Model_name}')
                    lr = lr[:,0,:,:,:]
                    sr_batch = model(lr.cuda(),s)
                    sr_batch = (sr_batch+1)/2
                    def process_coarse_image(i):
                        x = X[i]
                        y = Y[i]
                        z = Z[i]
                        sr_one = sr_batch[i].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                        #sr_one = np.uint8(np.round(sr_one * 255))
                        sr_ROI = sr_one[h: h + ROI_size, w: w + ROI_size, :]
                        mini_ind = (ith_batch * batch_size + i) % coarse_num
                        #a = net_F(torch.from_numpy(sr_ROI).float().squeeze(2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1), torch.from_numpy(GT_ROI).squeeze(2).unsqueeze(0).unsqueeze(0).float().repeat(1, 3, 1, 1))
                        PSNR_C[z, x, y, mini_ind] =  calculate_psnr(sr_ROI, sr_origin_ROI ) #sr_origin_ROI  a.numpy()
                    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以调整max_workers的值
                        futures = [executor.submit(process_coarse_image, i) for i in range(sr_batch.shape[0])]
                    for future in futures:
                        future.result()
                    sio.savemat(
                        f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-C{coarse_num}.mat',
                        {'PSNR': PSNR_C})
                    ith_batch += 1
            torch.cuda.empty_cache()
            # 创建细粒度block坐标
            PSNR_error_C = np.abs(psnr_origin - PSNR_C)
            for z in range(5):
                for cx in range(num_blocks_size_h):
                    for cy in range(num_blocks_size_w):
                        if PSNR_F[z, cx, cy, -1] != 0:  # and PSNR_F[cx, cy, -2] != 0
                            print(f'{Model_name}-block {cx}-{cy} fine is processed')
                            continue
                        elif np.max(PSNR_error_C[z, cx, cy, :]) >= tol:
                            for m in range(fine_num):
                                for n in range(5):
                                    fine_coordinates_dict[len(fine_coordinates_dict)] = ((cx, cy),m,n)
            dataset = InterventionDataset(fine_coordinates_dict, input_image, mask_block_size, patch_path,task=task,beta=beta,rain=rain)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            ith_batch = 0
            with torch.no_grad():
                for lr,index in tqdm(dataloader):
                    lr = lr[:,0,:,:,:]
                    (X,Y,K,Z)=index
                    s = torch.full((lr.shape[0], 1), n, dtype=torch.float32)
                    print(f'Fine stage: {patch_path}  {Image_name}/{Model_name}')  #lr[50,1,5,1088,1600]
                    sr_batch = model(lr.cuda(),s)
                    sr_batch = (sr_batch+1)/2
                    out_0 = sr_batch[0, 0, :, :]
                    def process_fine_image(i):
                        x = X[i]
                        y = Y[i]
                        z = Z[i]
                        k = K[i]
                        sr_one = sr_batch[i].detach().cpu().numpy().transpose(1, 2, 0).clip(0, 1)
                        sr_ROI = sr_one[h: h + ROI_size, w: w + ROI_size, :]
                        # mini_ind = (ith_batch * batch_size + i) % fine_num #当前干预次数，用来记录第几次干预
                        # print(z,mini_ind)
                        #a = net_F(torch.from_numpy(sr_ROI).float().squeeze(2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1),torch.from_numpy(GT_ROI).squeeze(2).unsqueeze(0).unsqueeze(0).float().repeat(1, 3, 1, 1))
                        PSNR_F[z, x, y, k] = calculate_psnr(sr_ROI, sr_origin_ROI )
                    with ThreadPoolExecutor(max_workers=5) as executor:  # 可以调整max_workers的值
                        futures = [executor.submit(process_fine_image, i) for i in range(sr_batch.shape[0])]
                    for future in futures:
                        future.result()
                    ith_batch += 1
                    sio.savemat(
                        f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-F{fine_num}.mat',
                        {'PSNR': PSNR_F})
                    print(f'./{task}-CEM/{prefix}-R{ROI_size}M{mask_block_size}/{Model_name}/{Image_name}-PSNR_all_perturb-F{fine_num}.mat')
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEM Configuration")
    parser.add_argument("--config", type=str, default="demo-DN_CEM.yml", help="Path to the configuration file")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    main(config)
