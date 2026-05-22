from data.base_dataset import BaseDataset
from data.image_folder import get_sorted_image_paths
import numpy as np
import random
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import tifffile
import torch
from skimage.  morphology import disk
from scipy.ndimage import median_filter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import kornia
import torch.nn.functional as F
import random

def outliers_rmv_circle(img, radius=1, threshold=10, img_bit=16):
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
def outliers_rmv(img, th=0.1, r=1):
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
    #
    # plt.tight_layout()
    # plt.show()
    return filtered.squeeze() if input_tensor.dim() == 2 else filtered

def load_random_image(gt_root, n=5):
    all_items = sorted(os.listdir(gt_root))
    choice = random.choice(all_items)
    choice_path = os.path.join(gt_root, choice)

    if os.path.isdir(choice_path):
        # 如果是文件夹
        gt_path_files = sorted(os.listdir(choice_path))
        gt_path = random.choice(gt_path_files)
        gt_path = os.path.join(choice_path, gt_path)
        all_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.tiff', '.tif'))])
        start_idx = random.randint(0, len(all_files) - n)
        selected_files = all_files[start_idx:start_idx + n]

        accumulated_img = None
        for fname in selected_files:
            img_path = os.path.join(gt_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if accumulated_img is None:
                accumulated_img = img
            else:
                accumulated_img += img
        return accumulated_img

    elif choice.endswith(('.tiff', '.tif')):
        # 如果是单张图像
        img = cv2.imread(choice_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        return img
    else:
        raise RuntimeError(f"选中了无效文件: {choice_path}")

def random_crop_with_coords(img, crop_size):
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
import os
class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot # create a path '/path/to/data/trainA'
        self.dir_gt = opt.gtroot
        self.gt_path = sorted(os.listdir(self.dir_gt))
        self.gt_paths,self.folder_size_gt,self.image_files_size_gt = get_sorted_image_paths(self.dir_gt)
        self.A_paths,self.folder_size,self.image_files_size = get_sorted_image_paths(self.dir_A)
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.patch_size = opt.patch_size
        self.epoch_size = self.opt.epoch_size

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        trans_totensor = transforms.ToTensor()
        trans_normalize = transforms.Normalize((0.5,), (0.5,))
        valid_starts = []
        ind = list(range(1, 21))
        # ind = [1]#, 15, 20]
        n = random.choices(ind)[0]
        num = n*5
        current_index = 0
        for folder_idx in range(self.folder_size):
            folder_image_count = self.image_files_size[folder_idx]
            for start in range(current_index, current_index + folder_image_count, 1):
                if start + num-1 < current_index + folder_image_count:
                    valid_starts.append(start)
                    #valid_starts.extend(range(start, start + folder_image_count - 4))
            current_index += folder_image_count
        start = random.choice(valid_starts) # 选择一个起始点
        consecutive_numbers = [start + i for i in range(num)] # 生成5个连续的数字
        A = []
        iso = []
        for i in range(0,5):
            if consecutive_numbers[i*n+(n-1)] < len(self.A_paths):
                image_paths = [self.A_paths[consecutive_numbers[i * n + j]] for j in range(n)]
                images = [cv2.imread(img_path, cv2.IMREAD_UNCHANGED) for img_path in image_paths]
                if any(img is None for img in images):
                    raise ValueError(f"部分图像无法读取: {image_paths}")
                image_stack = np.stack(images, axis=0)  # 形状 (ind, H, W)
                A_img = np.sum(image_stack, axis=0).astype(images[0].dtype)
                A_img =A_img.astype(images[0].dtype)
            A_img = np.expand_dims(A_img, axis=2)
            A_img = A_img.astype(np.float32)
            A_img = trans_totensor(A_img).unsqueeze(0)
            A.append(A_img)
        A = torch.cat(A, dim=1)
        #proposed
        gt_root = "/home/vipuser/project/pytorch-CycleGAN-and-pix2pix-master/datasets/che_end/trainA"
        img = load_random_image(gt_root, n)
        gt_img = trans_totensor(img).unsqueeze(0)
        gt_img = outliers_rmv_circle(gt_img)
        # #fastdvdnet
        # num_groups = len(consecutive_numbers) // 5
        # gt_sum = None
        # for i in range(num_groups):
        #     mid_idx = consecutive_numbers[i * 5 + 2]
        #     if mid_idx >= len(self.gt_paths):
        #         raise IndexError("GT index 超出范围")
        #     # 2. 读取 GT 图像
        #     gt_img = cv2.imread(self.gt_paths[mid_idx], cv2.IMREAD_UNCHANGED)
        #     if gt_img is None:
        #         raise ValueError(f"GT 图像无法读取: {self.gt_paths[mid_idx]}")
        #     gt_img = gt_img.astype(np.float32)
        #     if gt_sum is None:
        #         gt_sum = gt_img
        #     else:
        #         gt_sum += gt_img
        # #gt_avg = gt_sum / 5  # 这里 num_groups = 5
        # gt_avg = np.expand_dims(gt_sum, axis=2) # (H, W, 1)
        # gt_avg = trans_totensor(gt_avg)  # (1, H, W)
        # gt_img = gt_avg.unsqueeze(0)  # (1, 1, H, W)
        gt_img = outliers_rmv_circle(gt_img)

        rand_num = random.randint(512, 1024)
        i = 0
        a_max = 0
        gt_max = 0
        while i < 2:
            m = (1,1)
            A_img_crop, top, left = random_crop_with_coords(A,crop_size=rand_num)
            gt_img_crop, _, _ = random_crop_with_coords(gt_img, crop_size=rand_num)
            #A_img_m_crop = crop_image_with_coords(A_img_m.unsqueeze(0).unsqueeze(0),top, left, crop_size=rand_num)
            if A_img_crop.max() >= 0 and gt_img_crop.mean()>2:  # and i<self.epoch_size/self.A_size:  2
                if i == 0:
                    metrics_vector_I_B = torch.full((1, 1), n, dtype=torch.float32)   #num
                    # A_img_crop = A_img_crop.squeeze().numpy()
                    # A_img_crop = torch.from_numpy(A_img_crop).unsqueeze(0)
                    A_img_crop = (A_img_crop / (n*512)).clamp(0, 1.)  #400
                    A_img_crop =  trans_normalize(A_img_crop)
                    gt_img_crop = (gt_img_crop/ (n*512)).clamp(0, 1.)
                    gt_img_crop = trans_normalize(gt_img_crop)
                    # A_img_m_crop = A_img_m_crop/255.
                    # A_img_m_crop = trans_normalize(A_img_m_crop)
                    gt = gt_img_crop
                    img = A_img_crop
                    #im = A_img_m_crop
                    i += 1
                elif i!=0 and gt_img_crop.mean()>2:
                    A_img_crop = (A_img_crop / (n*512)).clamp(0, 1.)  #400
                    A_img_crop = trans_normalize(A_img_crop)
                    gt_img_crop = (gt_img_crop/ (n*512)).clamp(0, 1.)
                    gt_img_crop = trans_normalize(gt_img_crop)
                    # A_img_m_crop = A_img_m_crop/255.
                    # A_img_m_crop = trans_normalize(A_img_m_crop)
                    img = torch.cat((img, A_img_crop))
                    gt = torch.cat((gt, gt_img_crop))
                    i +=1
        return {'A': img,'S':metrics_vector_I_B,'gt':gt,'m':m}
    def __len__(self):
        #print(int(self.A_size))5253
        return int(self.A_size/10)




