import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import cv2

def generate_hotpixel_noise_map(a, image_shape, num_Hotpixels_range=(300,400)):

    s = image_shape
    m = a.item()  # 设置最大噪声值为1
    #print(m)
    noise_num = np.random.randint(num_Hotpixels_range[0], num_Hotpixels_range[1] + 1) #噪声点数量
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
    noise_map = np.tile(np.expand_dims(noise_matrix, axis=0), (4, 1, 1, 1))
    noise_map = torch.tensor(noise_map, dtype=torch.float32)
    del noise_matrix
    return noise_map

class ResBlock(nn.Module):
    def __init__(self, nf, ksize, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2),
            norm(nf), act(),
            nn.Conv2d(nf, nf, ksize, 1, ksize // 2)
        )

    def forward(self, x):
        return torch.add(x, self.body(x))


class KernelModel(nn.Module):
    def __init__(self, scale=1):
        super().__init__()

        # Default options (can be modified based on requirements)
        self.opt = {
            "nc": 1,  # Number of channels for random noise (zk)
            "nf": 64,  # Number of filters in convolution layers
            "nb": 5,  # Number of ResBlocks
            "ksize": 7,  # Kernel size
            "spatial": False,  # Whether to use spatial noise
            "mix": True,  # Whether to mix noise with input
            "zero_init": True  # Whether to zero initialize the last conv layer
        }
        self.scale = scale

        nc, nf, nb = self.opt["nc"], self.opt["nf"], self.opt["nb"]
        ksize = self.opt["ksize"]

        if self.opt["spatial"]:
            head_k = body_k = ksize
        else:
            head_k = body_k = 1

        if self.opt["mix"]:
            in_nc = 1 + nc
        else:
            in_nc = nc

        deg_kernel = [
            nn.Conv2d(in_nc, nf, head_k, 1, head_k // 2),
            nn.BatchNorm2d(nf), nn.ReLU(True),
            *[ResBlock(nf=nf, ksize=body_k) for _ in range(nb)],
            nn.Conv2d(nf, ksize ** 2, 1, 1, 0),
            nn.Softmax(dim=1)
        ]
        self.deg_kernel = nn.Sequential(*deg_kernel)

        if self.opt["zero_init"]:
            nn.init.constant_(self.deg_kernel[-2].weight, 0)
            nn.init.constant_(self.deg_kernel[-2].bias, 0)
            self.deg_kernel[-2].bias.data[ksize ** 2 // 2] = 1

        self.pad = nn.ReflectionPad2d(ksize // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        h = H // self.scale
        w = W // self.scale

        if self.opt["nc"] > 0:
            if self.opt["spatial"]:
                #zk = torch.randn(B, self.opt["nc"], H, W).to(x.device)
                zk = generate_hotpixel_noise_map(torch.tensor([1.0]), (self.opt["nc"], H, W)).to(x.device)
            else:
                #zk = torch.randn(B, self.opt["nc"], 1, 1).to(x.device)
                zk = generate_hotpixel_noise_map(torch.tensor([1.0]), (self.opt["nc"], H,W)).to(x.device)
                # if self.opt["mix"]:
                #     zk = zk.repeat(1, 1, H, W)

        if self.opt["mix"]:
            if self.opt["nc"] > 0:
                inp = torch.cat([x, zk], 1)
            else:
                inp = x
        else:
            inp = zk

        ksize = self.opt["ksize"]

        kernel = self.deg_kernel(inp).view(B, 1, ksize ** 2, H, W)

        x = x.view(B * C, 1, H, W)
        x = F.unfold(
            self.pad(x), kernel_size=ksize, stride=self.scale, padding=0
        ).view(B, C, ksize ** 2, h, w)

        # 调试信息：打印 x 和 kernel 的尺寸
        # print(f'x shape: {x.shape}')
        # print(f'kernel shape: {kernel.shape}')

        # 确保 kernel 在展开后与 x 的尺寸匹配
        kernel = kernel.view(B, 1, ksize ** 2, h, w)
        x = torch.mul(x, kernel).sum(2).view(B, C, h, w) #逐元素相乘，在第2维度上进行求和，特征图与卷积核的卷积操作
        #x = x.view(B, C, h, w)
        kernel = kernel.view(B, ksize, ksize, h, w).squeeze()

        return x, kernel


class NoiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.deg_noise = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            *[ResBlock(64, 1) for _ in range(5)],
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.deg_noise(x)


class DegModel(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.deg_kernel = KernelModel(scale=scale)
        self.deg_noise = NoiseModel()

    def forward(self, x):
        x, kernel = self.deg_kernel(x)
        #del kernel
        #a = kernel[0,0,:,:]
        noise = self.deg_noise(x) #a.unsqueeze(0).unsqueeze(0)
        torch.cuda.empty_cache()
        x = x + noise
        del noise
        del kernel
        return x   #,noise   #, kernel, noise

# torch.cuda.set_device(1)
# device = torch.device("cuda:1")  # 指定gpu1为主GPU
# torch.backends.cudnn.benchmark = True  # CUDNN optimization
# # 初始化模型
# model = DegModel(scale=1).to(device)
#
# # 使用 DataParallel 并行化模型
# model = nn.DataParallel(model, device_ids=[1, 0], output_device=1)
# # 示例输入张量
# x = torch.randn(4, 1, 256, 256)
#
# # 前向传播
# output, kernel, noise = model(x)
# print(output.shape)  # 输出形状
# print(kernel.shape)  # 内核形状
