import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision
from math import exp
import numpy as np
import matplotlib.pyplot as plt

'''SSIMloss'''
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(weights=True)
        else:
            model = torchvision.models.vgg19(weights=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        x = x - x.min()
        x = x / x.max()
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output
#### Define Network used for Perceptual Loss
def define_F(use_bn=False):
    gpu_ids = [0,1]
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 28   #34   28
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)

    netF.eval()  # No need to train
    return netF

#perceptual loss
class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        vgg_model = torchvision.models.vgg19(weights=True).features
        # 冻结VGG19的参数
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': "relu3_2",
            '22': "relu4_2",
            '28': "conv_28",
            '31': "relu5_2"
        }
        self.weight = [1/2.6,1/4.8,1/3.7,10/1.5,10/1.5]
        #self.weight = [0.0, 0.0, 0.0, 0.0, 10/1.5, 0.0]
        #self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            #print("vgg_layers name:",name,module)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        #print(output.keys())
        return list(output.values())

    def forward(self, output, gt):
        l1_loss = nn.L1Loss()  #l1_loss
        loss = []
        output_features = self.output_features(output)
        gt_features = self.output_features(gt)
        for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
                zip(output_features, gt_features, self.weight)):
            loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)  #F.mse_loss
        return sum(loss), output_features  # /len(loss)


# 输入的应该时feature_maps.shape = (H,W,Channels)
# 下图对relu1_2 进行了可视化，有64channels，拼了个了8*8的图
import math
def get_row_col(num_maps):
    """
    根据特征图数量计算适合的行数和列数
    """
    row = int(math.sqrt(num_maps))
    col = math.ceil(num_maps / row)
    return row, col
def visualize_feature_map(feature_maps):
    # 创建特征子图，创建叠加后的特征图
    # param feature_batch: 一个卷积层所有特征图
    # np.squeeze(feature_maps, axis=0)
    for j in range(0,6):
        print("visualize_feature_map shape:{},dtype:{}".format(feature_maps[j].shape, feature_maps[j].dtype))
        num_maps = feature_maps[j].shape[1]
        feature_map = feature_maps[j].squeeze(0)
        feature_map_combination = []
        plt.figure(figsize=(8, 7))
        # 取出 featurn map 的数量，因为特征图数量很多，这里直接手动指定了。
        # num_pic = feature_map.shape[2]
        row, col = get_row_col(num_maps)
        # 将 每一层卷积的特征图，拼接层 5 × 5
        for i in range(0, num_maps):
            feature_map_split = feature_map[i, :, :]
            feature_map_combination.append(feature_map_split)
            plt.subplot(row, col, i + 1)
            plt.imshow(feature_map_split.cpu().detach().numpy())
            plt.axis('off')

        plt.savefig(f'/home/yuhuizhen/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/Addnoise/feature_map/relu{j}_feature_map.png')  # 保存图像到本地
        plt.show()
