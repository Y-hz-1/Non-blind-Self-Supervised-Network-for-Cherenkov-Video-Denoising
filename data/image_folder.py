"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import os
import re
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def extract_numbers(text):
    """提取字符串中的数字并转换为整数列表"""
    numbers = re.findall(r'\d+', text)
    return list(map(int, numbers))

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]

def get_sorted_image_paths(root_dir):
    """从根目录中提取图像路径，并按文件夹和图像编号排序"""
    sorted_image_paths = []
    # 获取所有文件夹的路径，并按文件夹名的数字排序
    subfolders = sorted(os.listdir(root_dir), key=lambda x: extract_numbers(x))
    folder_size = 0
    image_files_size = []
    for folder in subfolders:
        folder_path = os.path.join(root_dir, folder)
        folder_size +=1
        if os.path.isdir(folder_path):
            # 获取当前文件夹中的所有图像文件并按图像编号排序
            image_files = sorted(os.listdir(folder_path), key=lambda x: extract_numbers(x))
            image_files_size_n = len(image_files)
            # 将完整路径添加到列表中
            for image_file in image_files:
                sorted_image_paths.append(os.path.join(folder_path, image_file))
            image_files_size.append(image_files_size_n)
    return sorted_image_paths,folder_size,image_files_size

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


