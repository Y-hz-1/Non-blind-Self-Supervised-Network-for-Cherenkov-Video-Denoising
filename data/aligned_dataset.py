import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import re
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import kornia

import torch.nn.functional as F
def random_crop_with_coords(img, crop_size=512):
    """
    Performs random crop on the image and returns the cropped image along with the top-left corner coordinates.

    Args:
    - img (PIL.Image or Tensor): The image to be cropped.
    - crop_size (int or tuple): The size of the crop.

    Returns:
    - cropped_img (Tensor): The cropped image.
    - top, left (int, int): The top-left corner of the crop.
    """
    # Ensure crop_size is a tuple (height, width)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    # Get image dimensions
    if isinstance(img, torch.Tensor):
        img_height, img_width = img.shape[-2], img.shape[-1]
    else:
        img_width, img_height = img.size

    crop_height, crop_width = crop_size

    # Ensure the crop size is not larger than the image size
    assert crop_height <= img_height and crop_width <= img_width, "Crop size must be smaller than the image size."

    # Randomly select the top-left corner of the crop
    top = random.randint(0, img_height - crop_height)
    left = random.randint(0, img_width - crop_width)

    # Perform the crop
    if isinstance(img, torch.Tensor):
        cropped_img = img[..., top:top + crop_height, left:left + crop_width]
    else:
        cropped_img = img.crop((left, top, left + crop_width, top + crop_height))
    cropped_img = F.interpolate(cropped_img, size=(512, 512), mode='bilinear', align_corners=False)

    return cropped_img, top, left
def crop_image_with_coords(img, top, left, crop_size):
    """
    Crops the image based on the provided coordinates and crop size.
    Args:
    - img (PIL.Image or Tensor): The image to be cropped.
    - top, left (int, int): The top-left corner coordinates for cropping.
    - crop_size (int or tuple): The size of the crop.

    Returns:
    - cropped_img (PIL.Image or Tensor): The cropped image.
    """
    # Ensure crop_size is a tuple (height, width)
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    crop_height, crop_width = crop_size

    # Perform the crop
    if isinstance(img, torch.Tensor):
        cropped_img = img[..., top:top + crop_height, left:left + crop_width]
    else:
        cropped_img = img.crop((left, top, left + crop_width, top + crop_height))
    cropped_img = F.interpolate(cropped_img, size=(512, 512), mode='bilinear', align_corners=False)
    return cropped_img



def median_filter_3x3(input_tensor):
    """Apply 3×3 median filtering using Kornia.
    Args:
        input_tensor: Input tensor with shape (B, C, H, W)
    Returns:
        Filtered tensor preserving original dimensions
    """
    # Ensure 4D input format (batch, channel, height, width)
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    # Apply median filtering with 3×3 kernel
    filtered = kornia.filters.median_blur(input_tensor, kernel_size=(3, 3))
    # figsize = (12, 6)
    # plt.figure(figsize=figsize, dpi=150)
    # # Original image (normalized to 3σ for radiation images)
    # plt.subplot(1, 2, 1)
    # plt.imshow(input_tensor[0, 0].cpu().numpy(),cmap='gray')
    # plt.title('Original Image\n[SNR: %.1f dB]' % (10 * torch.log10(input_tensor.var() / filtered.var()).item()))
    # plt.axis('off')
    # # Filtered result (window leveled to original)
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered[0, 0].cpu().numpy(),cmap='gray')
    # plt.title('3×3 Median Filtered\n[Noise Reduction: %.1f%%]' %
    #           (100 * (input_tensor.std().item() - filtered.std().item()) / input_tensor.std().item()))
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
    return filtered.squeeze() if input_tensor.dim() == 2 else filtered

'''noise_map'''
def generate_hotpixel_noise_map(a, image_shape, num_Hotpixels_range=(10,20)):
    s = image_shape
    m = a.item() # 设置最大噪声值为1
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
        gaussian_height = 7 + np.random.choice(randr)  #高斯核的高度
        gaussian_width = 7 + np.random.choice(randr)
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

import glob
def load_and_sum_images_from_folder(folder_path,m , n):
    image_paths = sorted(glob.glob(os.path.join(folder_path, str(m),'*.tif')))
    if len(image_paths) - n[0]<0:
        print(folder_path,m)
    start_index = random.randint(0, len(image_paths) - n[0])
    # for path in image_paths:
    #     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    #     print(path, type(img), None if img is None else img.shape, None if img is None else img.dtype)
    images = [cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED) for i in range(start_index, start_index + n[0])]
    sum_image = np.zeros_like(np.array(images[0]))  # 创建一个与图像大小相同的零数组
    for img in images:
        sum_image += np.array(img)
    return sum_image
import torch.nn.functional as F


def outliers_rmv_circle(img, radius=1, threshold=10, img_bit=16):
    """
    Remove outliers using a circular neighborhood (mimicking ImageJ).

    Parameters:
    - img: Tensor of shape (B, 1, H, W), values in [0, 1]
    - radius: Integer, radius of the circular neighborhood (ImageJ style)
    - threshold: Threshold in uint16 scale (e.g., 10 for 16-bit image)
    - img_bit: Bit depth of original image, used for scaling threshold (default: 16-bit)
    """
    assert img.dim() == 4 and img.shape[1] == 1, "Input must be (B,1,H,W) tensor"
    B, _, H, W = img.shape
    img = img.squeeze(1)  # (B, H, W)
    # Convert threshold to float scale based on bit depth
    max_val = float(2 ** img_bit - 1)
    th = threshold / max_val
    # Build circular mask
    size = 2 * radius + 1
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = radius
    circular_mask = ((x - center) ** 2 + (y - center) ** 2) <= radius ** 2
    circular_mask = circular_mask.flatten()  # Shape (K*K,)
    # Reflect pad and unfold
    padded = F.pad(img, (radius, radius, radius, radius), mode='reflect')  # (B, H+2r, W+2r)
    unfolded = padded.unfold(1, size, 1).unfold(2, size, 1)  # (B, H, W, K, K)
    neighborhoods = unfolded.contiguous().view(B, H, W, -1)  # (B, H, W, K*K)
    # Apply circular mask
    neighborhoods = neighborhoods[..., circular_mask.to(neighborhoods.device)]  # (B, H, W, N)
    # Compute median of circular neighborhood
    median_values, _ = torch.median(neighborhoods, dim=-1)  # (B, H, W)
    # Identify outliers
    outlier_mask = torch.abs(img - median_values) > th
    cleaned = torch.where(outlier_mask, median_values, img)  # (B, H, W)
    return cleaned.unsqueeze(1)  # Back to (B, 1, H, W)
def outliers_rmv(img, th=10, r=1):
    assert img.dim() == 4 and img.shape[1] == 1, "Input must be (B,1,H,W) tensor"

    B, C, H, W = img.shape  # 批次、通道、高度、宽度
    img = img.squeeze(1)  # 变为 (B, H, W)，只对空间维度进行操作
    padded_image = F.pad(img, (r, r, r, r), mode='reflect')  # (B, H+2r, W+2r)
    unfolded = padded_image.unfold(1, 2 * r + 1, 1).unfold(2, 2 * r + 1, 1)  # (B, H, W, K, K)
    neighborhoods = unfolded.contiguous().view(B, H, W, -1)  # (B, H, W, K*K)
    median_values, _ = torch.median(neighborhoods, dim=-1)  # (B, H, W)
    outlier_mask = torch.abs(img - median_values) > th  # (B, H, W)
    Clean_cherenkov = torch.where(outlier_mask, median_values, img)  # (B, H, W)
    return Clean_cherenkov.unsqueeze(1)  # 还原为 (B,1,H,W)


import torchvision.transforms.functional as TF
def apply_same_transform(A, B):
    # Random Horizontal Flip
    if random.random() > 0.5:
        A = TF.hflip(A)
        B = TF.hflip(B)
    # Random Vertical Flip
    if random.random() > 0.5:
        A = TF.vflip(A)
        B = TF.vflip(B)
    # Random Rotation
    # angle = random.uniform(-10, 10)
    # A = TF.rotate(A, angle)
    # B = TF.rotate(B, angle)
    k = random.choice([1, 2, 3])  # Rotate 90, 180, or 270 degrees
    A = TF.rotate(A, angle=90 * k)
    B = TF.rotate(B, angle=90 * k)
    return A, B
class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_AB = os.path.join(opt.dataroot)  # get the image directory, opt.phase
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        trans_normalize = transforms.Normalize((0.5,), (0.5,))
        trans_totensor = transforms.ToTensor()
        self.transform_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
        ])
        ind = list(range(1, 21))
        B_n = random.choices(ind)
        self.dir_B = "/home/vipuser/project/pytorch-CycleGAN-and-pix2pix-master/datasets/che_end/trainB"
        B_m =len(os.listdir(self.dir_B))
        datast_i = [1,2]
        p = random.choice(datast_i)
        if p<=1:
            j = random.choice([x for x in range(2, B_m - 2) if x != 27])
            A_path = os.path.join(self.dir_A,f"image({j}).tiff")
            # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
            A_img = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
            A_img = np.expand_dims(A_img, axis=2)
            A_img = A_img.astype(np.float32)
            A_img = trans_totensor(A_img).unsqueeze(0)
            # BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB
            B_img = load_and_sum_images_from_folder(self.dir_B, j, B_n)
        else:
            # j = random.randint(59,60)
            # num = 20
            # A_path_s = os.path.join(self.dir_A, f"{j}")
            # subfolders = [f.path for f in os.scandir(A_path_s) if f.is_dir()]
            # selected_folder = random.choice(subfolders)
            # if selected_folder == "/zhangyanqiu/yuhuizhen_tju/pytorch-CycleGAN-and-pix2pix-master/pytorch-CycleGAN-and-pix2pix-master/datasets/trainA/60/1":
            #     num = 21
            # image_paths = sorted(glob.glob(os.path.join(selected_folder, '*.tiff')),key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
            # a = max(B_n[0],num)
            # start_index = random.randint((a // 2), len(image_paths) - (a - a // 2))
            # selected_images = image_paths[start_index - (num // 2): start_index + (num // 2) + (num % 2)]
            # image_stack = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in selected_images]
            # image_stack = np.stack(image_stack, axis=2)
            # A_img = np.median(image_stack, axis=2).astype(np.uint16)
            j = random.randint(61, 62)
            A_path_s = os.path.join(self.dir_A, f"{j}")
            subfolders = [f.path for f in os.scandir(A_path_s) if f.is_dir()]
            selected_folder = random.choice(subfolders)
            image_paths = sorted(glob.glob(os.path.join(selected_folder, '*.tiff')),key=lambda x: int(re.search(r'\((\d+)\)', x).group(1)))
            #start_index = random.randint(0, len(image_paths))
            candidate_indices = list(range(B_n[0]//2, len(image_paths)-B_n[0]//2, 5))
            start_index = random.choice(candidate_indices)
            #selected_image = image_paths[start_index]
            #A_img = cv2.imread(selected_image, cv2.IMREAD_UNCHANGED)
            selected_images_A = image_paths[start_index - (B_n[0] // 2): start_index + (B_n[0] // 2) + (B_n[0] % 2)]
            image_stack_A = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in selected_images_A]
            #image_stack_A = np.array(image_stack_A)
            try:
                image_stack_A = np.array(image_stack_A)
            except ValueError as e:
                print("Error when converting image_stack_A to numpy array:")
                print("selected_images_A =", selected_images_A)
                for i, img in enumerate(image_stack_A):
                    print(
                        f"Index {i}: type={type(img)}, shape={None if img is None else img.shape}, dtype={None if img is None else img.dtype}")
                raise e  # 保持原始错误抛出，方便调试
            A_img = np.sum(image_stack_A, axis=0).astype(np.uint16)
            A_img = np.expand_dims(A_img, axis=2)
            A_img = A_img.astype(np.float32)
            A_img = trans_totensor(A_img).unsqueeze(0)
            A_img = outliers_rmv_circle(A_img)
            B_path_s = selected_folder.replace('/trainA/61', '/trainB/61').replace('/trainA/62', '/trainB/62')
            #B_path_s = selected_folder.replace('/trainA/61', '/trainB/trainB/61').replace('/trainA/62', '/trainB/trainB/62')
            #a = os.path.join(selected_folder, '*.tif')
            image_paths_B = sorted(glob.glob(os.path.join(B_path_s, '*.tif')))
            selected_images_B = image_paths_B[start_index - (B_n[0] // 2): start_index + (B_n[0] // 2) + (B_n[0] % 2)]
            image_stack_B = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in selected_images_B]
            image_stack_B = np.array(image_stack_B)
            B_img = np.sum(image_stack_B, axis=0).astype(np.uint16)
            if B_img.shape == ():
                print(B_path_s , start_index - (B_n[0] // 2),start_index + (B_n[0] // 2) + (B_n[0] % 2))
        B_img = np.expand_dims(B_img, axis=2)
        B_img = B_img.astype(np.float32)
        B_img = trans_totensor(B_img).unsqueeze(0)
        _, C, H, W = B_img.shape
        a = torch.tensor(B_n[0])
        metrics_vector_I = torch.full((1,1), a, dtype=torch.float32)
        rand_num = random.randint(512, 1024)
        i = 0
        b_size = 1
        while i < b_size:
            A_img_crop, top, left = random_crop_with_coords(A_img, crop_size=rand_num)
            B_img_crop = crop_image_with_coords(B_img, top, left, crop_size=rand_num)
            A_img_crop,B_img_crop = apply_same_transform(A_img_crop,B_img_crop)
            A_img_crop = A_img_crop.squeeze().numpy()
            #A_img_crop = (A_img_crop - A_img_crop.min()) / (A_img_crop.max() - A_img_crop.min())
            # A_img_crop = (A_img_crop/300.).clamp(0,1.)
            #A_img_crop = A_img_crop / 255.
            if A_img_crop.max()>0 and i == 0:  #.mean()>=0.025 .max()>0 .mean()>=20
                A = A_img_crop
                A = trans_totensor(A).unsqueeze(0)
                if p == 1:
                    if A.max()>512:
                        print(A.max())
                    A = (A / 512.).clamp(0, 1.)
                else:
                    A = (A / (a * 512.)).clamp(0, 1.)
                A = trans_normalize(A)
                B = B_img_crop
                #B = B.squeeze().numpy()
                B = (B/(a*512.)).clamp(0, 1.)
                #B = trans_totensor(B).unsqueeze(0)
                B = trans_normalize(B)
                i += 1
            elif A_img_crop.max()>0 and i != 0:
                A_c = A_img_crop
                B_c = B_img_crop
                #B_c = B_c.squeeze().numpy()
                B_c = (B_c/(a*512.)).clamp(0, 1.)
                #B_c = trans_totensor(B_c).unsqueeze(0)
                B_c = trans_normalize(B_c)
                A_c = trans_totensor(A_c).unsqueeze(0)
                if p == 1:
                    print(A_c.max())
                    A_c = (A_c / 512.).clamp(0, 1.)
                else:
                    A_c = (A_c / (a * 512.)).clamp(0, 1.)
                A_c = trans_normalize(A_c)
                A = torch.cat((A, A_c))
                B = torch.cat((B, B_c))
                i += 1
        return {'A': A, 'B': B, 'S':metrics_vector_I} #, 'A_paths': A_path, 'B_gabor': B_gabor

    def __len__(self):
        """Return the total number of images in the dataset."""
        return int(len(self.A_paths)/10)
