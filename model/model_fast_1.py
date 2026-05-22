"""
Definition of the FastDVDnet model
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.autograd import Function
def filter_on_channel(x: torch.Tensor, mode: str = 'min', keepdim: bool = True) -> torch.Tensor:
    # x: (B=4, C=3, H=512, W=512)
    # 变为: (B, H, W, C)
    if mode == 'max':
        out, _ = x.max(dim=1, keepdim=keepdim)
    elif mode == 'min':
        out, _ = x.min(dim=1, keepdim=keepdim)
    elif mode == 'med':
        out, _ = x.median(dim=1, keepdim=keepdim)
    elif mode == 'mean':
        out = x.mean(dim=1, keepdim=keepdim)
    elif mode == 'sum':
        out = x.sum(dim=1, keepdim=keepdim)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from ['max', 'min', 'median', 'mean']")

    return out
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class EnhancedCBAM(nn.Module):
	def __init__(self, channel, reduction=8):  # 降低压缩率
		super().__init__()

		# 改进的通道注意力（引入动态阈值）
		self.channel_attention = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channel, channel // reduction, 1),
			nn.ReLU(),
			nn.Conv2d(channel // reduction, channel, 1),
			nn.Sigmoid()
		)

		# 双模态空间注意力（抑制高光区域）
		self.spatial_attention = nn.Sequential(
			nn.Conv2d(2, 1, 7, padding=3),
			nn.Sigmoid(),
			nn.Conv2d(1, 1, 3, padding=1),  # 增加高斯模糊层
			nn.Sigmoid()
		)

	def forward(self, x):
		# 分阶段应用注意力
		ca = self.channel_attention(x)
		x = x * ca  # 先做通道加权

		# 空间注意力前增加亮度归一化
		mean = x.mean(dim=1, keepdim=True)
		std = x.std(dim=1, keepdim=True) + 1e-5
		normalized = (x - mean) / std
		sa = self.spatial_attention(normalized)

		return x * sa  # 再做空间加权

def hook_fn(module, input, output):
	# Store the output in a dictionary
	if hasattr(module, 'name'):
		features[module.name] = output.detach().cpu()

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''

	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm2d(out_ch), #eps=1e-03),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm2d(out_ch), #eps=1e-03),
			nn.LeakyReLU(inplace=True),
		)

	def forward(self, x):
		return self.convblock(x)


class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''

	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30 #30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, num_in_frames * self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.InstanceNorm2d(num_in_frames * self.interm_ch), #eps=1e-03),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),  # num_in_frames*self.interm_ch
			nn.InstanceNorm2d(out_ch), #eps=1e-03),
			nn.LeakyReLU(inplace=True),
		)
		#self.norm1 = AdaIN_1(style_dim=2, num_features=2)
	def forward(self, x):
		return self.convblock(x)

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(64, num_features*2) #*2
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        #beta = self.act(beta)
        a = self.norm(x)
        b = (1 + gamma) * a + beta
        return b

class AdaIN_1(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        a = self.norm(x)
        b = (1 + gamma) * a + beta
        return b

class AdaIN_fc(nn.Module):
    def __init__(self, style_dim):
        super().__init__()
        self.fc_0 = nn.Linear(style_dim,8)
        self.fc_1 = nn.Linear(8,16)
        self.fc_2 = nn.Linear(16,32)
        self.fc_3 = nn.Linear(32,64)
    def forward(self, s):
        h = self.fc_0(s)
        h = self.fc_1(h)
        h = self.fc_2(h)
        h = self.fc_3(h)
        return h
class AdaIN_out(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(64, num_features*2) #*2
        self.act = nn.ReLU()
    def forward(self, x, h):
        h = self.fc(h)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        beta = self.act(beta)
        a = self.norm(x)
        b = (1 + gamma) * a + beta
        return b
class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''

	def __init__(self, in_ch, out_ch, use_maxpool=False, norm_layer=nn.BatchNorm2d):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),  # , stride=2
			nn.InstanceNorm2d(out_ch), #eps=1e-03),
			nn.LeakyReLU(inplace=True),  # negative_slope=0.1,
			CvBlock(out_ch, out_ch)
		)
	def forward(self, x):
		return self.convblock(x)

class DownBlock_1(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''

	def __init__(self, in_ch, out_ch, use_maxpool=False, norm_layer=nn.BatchNorm2d):
		super(DownBlock_1, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),  # , stride=2
		)
	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''

	def __init__(self, in_ch, out_ch, use_interpolate=True):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Dropout(0.5),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), #kernel_size=3, padding=1,
			nn.InstanceNorm2d(out_ch),
			nn.LeakyReLU(inplace=True),
		)
	def forward(self, x):
		return self.convblock(x)


class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''

	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm2d(in_ch),  # eps=1e-03),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
		)

	def forward(self, x):
		return self.convblock(x)

import torchvision.transforms.functional as TF
class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x):
        return TF.gaussian_blur(x, [self.kernel_size, self.kernel_size], [self.sigma, self.sigma])
import torch
import torch.nn as nn
import torch.nn.functional as F
class MedianBlurLayer(nn.Module):
	def __init__(self, kernel_size=3):
		super(MedianBlurLayer, self).__init__()
		assert kernel_size % 2 == 1, "Kernel size must be an odd number"
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2

	def forward(self, x):
		b, c, h, w = x.shape
		x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
		x_unfold = F.unfold(x_padded, kernel_size=self.kernel_size)
		x_unfold = x_unfold.view(b, c, self.kernel_size * self.kernel_size, -1)
		x_median, _ = torch.median(x_unfold, dim=2)
		x_median = x_median.view(b, c, h, w)
		return x_median
class DenBlock_0509(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=3, out_s=False):
		super(DenBlock_0509, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_1(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_1(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.con_1 = CvBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr1)
		self.con_2 = CvBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
		self.act1 = nn.LeakyReLU()
		self.act2 = nn.LeakyReLU()
		#self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.upc3 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.adin1 = AdaIN(style_dim=self.style_dim,num_features=64)
		self.adin2 = AdaIN(style_dim=self.style_dim,num_features=128)
		self.out = nn.Tanh()
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2 ,S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.out_s == False:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x1 = self.adin1(x1,S)
			x1 = self.act1(x1)
			x1 = self.con_1(x1)
			x2 = self.downc1(x1)
			x2 = self.adin2(x2,S)
			x2 = self.act2(x2)
			x2 = self.con_2(x2)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x
class DenBlock_0605(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=3, out_s=False):
		super(DenBlock_0605, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_1(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_1(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.downc2 = DownBlock_1(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.con_1 = CvBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr1)
		self.con_2 = CvBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
		self.con_3 = CvBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr3)
		self.act1 = nn.LeakyReLU()
		self.act2 = nn.LeakyReLU()
		self.act3 = nn.LeakyReLU()
		#self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.upc3 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.AdaIN = AdaIN_fc(1)
		self.adin1 = AdaIN(style_dim=self.style_dim,num_features=64)
		self.adin2 = AdaIN(style_dim=self.style_dim,num_features=128)
		self.adin3 = AdaIN(style_dim=self.style_dim,num_features=256)
		self.out = nn.Tanh()
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2 ,S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		S = self.AdaIN(S)
		if self.out_s == False:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x1 = self.adin1(x1,S)
			x1 = self.act1(x1)
			x1 = self.con_1(x1)
			x2 = self.downc1(x1)
			x2 = self.adin2(x2,S)
			x2 = self.act2(x2)
			x2 = self.con_2(x2)
			x3 = self.downc2(x2)
			x3 = self.adin3(x3, S)
			x3 = self.act3(x3)
			x3 = self.con_3(x3)
			# Upsampling
			x3 = self.upc3(x3)
			x2 = self.upc2(x2 + x3)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x1 = self.adin1(x1,S)
			x1 = self.act1(x1)
			x1 = self.con_1(x1)
			x2 = self.downc1(x1)
			x2 = self.adin2(x2,S)
			x2 = self.act2(x2)
			x2 = self.con_2(x2)
			x3 = self.downc2(x2)
			x3 = self.adin3(x3, S)
			x3 = self.act3(x3)
			x3 = self.con_3(x3)
			# Upsampling
			x3 = self.upc3(x3)
			x2 = self.upc2(x2 + x3)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x
class DenBlock_0512(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=3, out_s=False):
		super(DenBlock_0512, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_1(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock_1(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.con_1 = CvBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr1)
		self.con_2 = CvBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
		self.act1 = nn.LeakyReLU()
		self.act2 = nn.LeakyReLU()
		#self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.upc3 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.AdaIN = AdaIN_fc(1)
		self.adin1 = AdaIN(style_dim=self.style_dim,num_features=64)
		self.adin2 = AdaIN(style_dim=self.style_dim,num_features=128)
		self.out = nn.Tanh()
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2 ,S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.out_s == False:
			S = self.AdaIN(S)#dddd
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x1 = self.adin1(x1,S)
			x1 = self.act1(x1)
			x1 = self.con_1(x1)
			x2 = self.downc1(x1)
			x2 = self.adin2(x2,S)
			x2 = self.act2(x2)
			x2 = self.con_2(x2)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x

class DenBlock_0322(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=3, out_s=False):
		super(DenBlock_0322, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.upc3 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.norm_0 = AdaIN_1(style_dim=self.style_dim,num_features=3)
		self.norm_1 = AdaIN_1(style_dim=self.style_dim,num_features=32)
		self.norm_2 = AdaIN_1(style_dim=self.style_dim,num_features=64)
		self.norm_3 = AdaIN_1(style_dim=self.style_dim,num_features=128)
		self.out = nn.Tanh()
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2 ,S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.out_s == False:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x
class DenBlock_0506(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames=3, out_s=False):
		super(DenBlock_0506, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc_0 = InputCvBlock(num_in_frames=4, out_ch=self.chs_lyr0)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.downc2 = DownBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr3)
		self.upc3 = UpBlock(in_ch=self.chs_lyr3, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.norm_0 = AdaIN_1(style_dim=self.style_dim,num_features=3)
		self.norm_1 = AdaIN_1(style_dim=self.style_dim,num_features=32)
		self.norm_2 = AdaIN_1(style_dim=self.style_dim,num_features=64)
		self.norm_3 = AdaIN_1(style_dim=self.style_dim,num_features=128)
		self.out = nn.Tanh()
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2 ,S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.out_s == False:
			x0_0 = torch.cat((in0, in1, in2), dim=1)
			S = S.view(S.size(0), S.size(1), 1, 1)
			S = S.repeat(x0_0.size(0), 1, x0_0.size(2), x0_0.size(3))
			x0_0 = torch.cat([x0_0, S], dim=1)
			x0 = self.inc_0(x0_0)
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			# Estimation
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			x2 = self.upc2(x2)
			x1 = self.upc1(x1 + x2)
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x
class DenBlock_0410(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""
	def __init__(self, num_input_frames, out_s=False):
		super(DenBlock_0410, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.chs_lyr3 = 256
		self.out_s = out_s
		self.style_dim =1
		self.Blur_N = GaussianBlurLayer(kernel_size=5, sigma=2.0)
		self.Blur_M = MedianBlurLayer(kernel_size=5)
		self.inc = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		#self.inc_0 = InputCvBlock(num_in_frames=3, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock_1(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		#self.downc_0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.con_1 = CvBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr1)
		self.act1 = nn.LeakyReLU()
		self.act2 = nn.LeakyReLU()
		self.downc1 = DownBlock_1(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		#self.downc_1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.con_2 = CvBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)
		self.ada = AdaIN_fc(style_dim=self.style_dim)
		self.norm_0 = AdaIN_out(num_features=3)
		self.norm_1 = AdaIN_out(num_features=32)
		self.norm_2 = AdaIN_out(num_features=64)
		self.norm_3 = AdaIN_out(num_features=128)
		self.cbam1 = CBAM(channel=64)
		self.cbam2 = CBAM(channel=32)
		self.cbam3 = CBAM(channel=256)
		self.cbam4 = CBAM(channel=512)
		# self.norm_0 = AdaIN_1(style_dim=self.style_dim, num_features=3)
		# self.norm_1 = AdaIN_1(style_dim=self.style_dim,num_features=32)
		# self.norm_2 = AdaIN_1(style_dim=self.style_dim,num_features=64)
		# self.norm_3 = AdaIN_1(style_dim=self.style_dim,num_features=128)
		self.out = nn.Tanh()

		self.reset_params()

		global features
		features = {}
		# Register hooks
		self._register_hooks()

	def _register_hooks(self):
		self.inc.name = 'InputCvBlock'
		self.downc0.name = 'DownBlock_0'
		self.downc1.name = 'DownBlock_1'
		self.upc2.name = 'UpBlock_2'
		self.upc1.name = 'UpBlock_1'
		self.outc.name = 'OutputCvBlock'

		self.inc.register_forward_hook(hook_fn)
		self.downc0.register_forward_hook(hook_fn)
		self.downc1.register_forward_hook(hook_fn)
		self.upc2.register_forward_hook(hook_fn)
		self.upc1.register_forward_hook(hook_fn)
		self.outc.register_forward_hook(hook_fn)

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2, S): # ,S
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		if self.out_s == False:
			x0_0 = torch.cat((in0, in1, in2), dim=1)
			# S = S.view(S.size(0), S.size(1), 1, 1)
			# S = S.repeat(x0_0.size(0), 1, x0_0.size(2), x0_0.size(3))
			# x0_0 = torch.cat([x0_0, min_map, S], dim=1)
			x0 = self.inc(x0_0)
			# Downsampling
			#x0 = self.norm_0(x0,S)
			h = self.ada(S)
			x1 = self.downc0(x0)
			x1 = self.norm_2(x1, h)
			x1 = self.act1(x1)
			x1 = self.con_1(x1)
			x2 = self.downc1(x1)
			x2 = self.norm_3(x2, h)
			x2 = self.act2(x2)
			x2 = self.con_2(x2)
			# Upsampling
			x2 = self.upc2(x2)
			#x1 = self.cbam1(x1)
			x1 = self.upc1(x1 + x2)
			# Estimation
			#x0 =self.cbam2(x0)
			x = self.outc(x0 + x1)
			x = self.out(x)
		else:
			x0 = self.inc( torch.cat((in0, in1, in2), dim=1))
			# Downsampling
			x1 = self.downc0(x0)
			x2 = self.downc1(x1)
			# Upsampling
			#x2 = self.Blur_N(x2)
			x2 = self.upc2(x2)
			#x2 = self.Blur_N(x2)
			#x1 = self.cbam1(x1)
			x1 = self.upc1(x1 + x2)
			#x1 = self.Blur_N(x1)
			# Estimation
			#x0 = self.cbam2(x0)
			x = self.outc(x0 + x1)
			x = self.out(x)
		return x

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		# self.temp1 = DenBlock_0512(num_input_frames)
		# encoder_path = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_encoder.pth'
		# state_encoder = torch.load(encoder_path)
		# state_dict = state_encoder["model_state_dict"]
		# new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
		# self.temp1.load_state_dict(new_state_dict, strict=True)
		# for module in [self.temp1.inc, self.temp1.downc0, self.temp1.con_1]:
		# 	for param in module.parameters():
		# 		param.requires_grad = False
		self.temp1 = DenBlock_0605(num_input_frames, out_s=False)
		self.temp2 = DenBlock_0605(num_input_frames, out_s=True)  # num_input_frames=3
		# model_np = [
        #         nn.Conv2d(1, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 32, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        #         nn.ReLU(),
        #         nn.Conv2d(32, 1, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
        #         nn.ReLU()
        # ]
		# self.model_np = nn.Sequential(*model_np)
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, S): # , S
		(x0, x1, x2, x3, x4) = tuple(x[:, m:m + 1, :, :] for m in range(self.num_input_frames))
		# First stage
		# min_map = filter_on_channel(torch.cat((x0, x1, x2, x3, x4), dim=1))
		# min_map = self.model_np(min_map)
		x20 = self.temp1(x0, x1, x2, S)# , S
		x21 = self.temp1(x1, x2, x3, S)# , S
		x22 = self.temp1(x2, x3, x4, S)# , S
		# Second stage
		x = self.temp2(x20, x21, x22, S)# , S
		# # # Plot feature maps
		# def plot_feature_maps(features, layer_name):
		# 	if layer_name in features:
		# 		feature_maps = features[layer_name]
		# 		num_feature_maps = feature_maps.shape[1]
		# 		plt.figure(figsize=(15, 15))
		# 		for i in range(min(num_feature_maps, 64)):  # Limit to first 64 feature maps
		# 			plt.subplot(8, 8, i + 1)
		# 			plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray') #, cmap='gray'
		# 			plt.axis('off')
		# 		plt.title(layer_name)
		# 		plt.show()
		#
		# # Example usage to plot feature maps from each layer
		# for layer_name in features.keys():
		# 	plot_feature_maps(features, layer_name)
		# x=x/2
		# x = (x - x.min()) / (x.max() - x.min())
		# x = x * 0.05
		# x22 = x22 * 0.1
		# x21 = x21 * 0.1
		# x20 = x20 * 0.1
		# x2 = x2 * 0.1
		# x22 = torch.exp(-x22)
		# x21 = torch.exp(-x21)
		# x20 = torch.exp(-x20)
		# x2 = torch.exp(-x2)
		# x = torch.exp(-x)
		# #x = -torch.log10(torch.abs(x))+torch.log10(x2)
		# x22 = (x22 - x22.min()) / (x22.max() - x22.min())
		# x21 = (x21 - x21.min()) / (x21.max() - x21.min())
		# x20 = (x20 - x20.min()) / (x20.max() - x20.min())
		# i =0
		# plt.figure(figsize=(10, 5))
		# plt.subplot(2, 3, 1)
		# plt.imshow(x20[i].squeeze(0).cpu().detach().numpy(), cmap='gray')  # 使用灰度图显示
		# plt.axis('off')  # 隐藏坐标轴
		# plt.subplot(2, 3, 2)
		# plt.imshow(x21[i].squeeze(0).cpu().detach().numpy(), cmap='gray')  # 使用灰度图显示
		# plt.axis('off')  # 隐藏坐标轴
		# plt.subplot(2, 3, 3)
		# plt.imshow(x22[i].squeeze(0).cpu().detach().numpy(), cmap='gray')  # 使用灰度图显示
		# plt.axis('off')  # 隐藏坐标轴
		# plt.subplot(2, 3, 4)
		# plt.imshow(x2[i].squeeze(0).cpu().detach().numpy(), cmap='gray')  # 使用灰度图显示
		# plt.axis('off')  # 隐藏坐标轴
		# plt.subplot(2, 3, 5)
		# plt.imshow(x[i].squeeze(0).cpu().detach().numpy(), cmap='gray')  # 使用灰度图显示
		# plt.axis('off')  # 隐藏坐标轴
		# # 显示图像
		# plt.show()
		# plt.close()
		return x


############################################################################################################
# Perceptual loss
############################################################################################################
import torchvision


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
	gpu_ids = [0, 1]
	device = torch.device('cuda' if gpu_ids else 'cpu')
	# PyTorch pretrained VGG19-54, before ReLU.
	if use_bn:
		feature_layer = 49
	else:
		feature_layer = 26  # 34   28  29
	netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
							   use_input_norm=True, device=device)
	netF.eval()  # No need to train
	return netF


def hubber_loss(y, y_predicted):
    error = y_predicted - y
    mae = torch.mean(torch.abs(error))
    delta = 1.35 * mae.item()
    abs_error = torch.abs(error)
    mask = abs_error < delta
    loss = torch.where(mask,0.5 * error ** 2,delta * (abs_error - 0.5 * delta))
    return loss.mean()

# def hubber_loss(y, y_predicted): #, delta
# 	error = y_predicted - y
# 	absolute_error = np.absolute(error.cpu().detach().numpy())
# 	total_absolute_error = np.sum(absolute_error)
# 	y_size = y.size()
# 	mae = total_absolute_error / y_size
# 	delta = 1.35 * mae
# 	y_size = y_size[0]
# 	total_error = 0
# 	for i in range(y_size):
# 		error = np.absolute(y_predicted[i] - y[i])
# 		if error < delta:
# 			hubber_error = (error * error) / 2
# 		else:
# 			hubber_error = (delta * error) / (0.5 * (delta * delta))
# 		total_error += hubber_error
# 	total_hubber_error = total_error / y.size
# 	return total_hubber_error


# perceptual loss——2
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
			'31': "relu5_2"
		}
		self.weight = [1/1500, 1/150, 1/15, 1/1.5, 10 / 1.5]
		# self.weight = [0.0, 0.0, 0.0, 0.0, 10/1.5, 0.0]
		# self.weight = [1.0, 1.0, 1.0, 1.0, 1.0]

	def output_features(self, x):
		output = {}
		for name, module in self.vgg_layers._modules.items():
			# print("vgg_layers name:",name,module)
			x = module(x)
			if name in self.layer_name_mapping:
				output[self.layer_name_mapping[name]] = x
		# print(output.keys())
		return list(output.values())

	def forward(self, output, gt):
		l1_loss = nn.L1Loss()  # l1_loss
		loss = []
		output_features = self.output_features(output)
		gt_features = self.output_features(gt)
		for iter, (dehaze_feature, gt_feature, loss_weight) in enumerate(
				zip(output_features, gt_features, self.weight)):
			loss.append(F.mse_loss(dehaze_feature, gt_feature) * loss_weight)  # F.mse_loss
		return sum(loss), output_features  # /len(loss)


####SNR
def define_S(im):
	# 对每个图像分别计算
	std = torch.std(im, dim=(2, 3), keepdim=True)  # 按通道计算标准差
	mean = torch.mean(im, dim=(2, 3), keepdim=True)  # 按通道计算均值
	snr = 10 * torch.log10(mean / std)  # 计算信噪比，使用对数转换为 dB
	return snr.squeeze(-1).squeeze(-1)  # 输出大小为 (6, 1)


####sparsity
def define_S1(image_tensor):
	b, _, _, _ = image_tensor.shape
	non_zero_elements = torch.count_nonzero(image_tensor, dim=(2, 3))  # .cuda()  # 计算每个图像的非零元素
	total_elements_per_image = torch.tensor([image_tensor[i].numel() for i in range(image_tensor.size(0))]).reshape(b,
																													1)  # .cuda()
	sparsity = non_zero_elements / total_elements_per_image  # 计算稀疏度
	return sparsity.squeeze(-1)  # 压缩维度，输出大小为 (6, 1)


####background_snr
from scipy.ndimage import label

def define_background_snr(im_n):
	mode_value, mode_count = torch.mode(im_n.view(-1))
	original_image = im_n[0, 0].numpy()
	tensor = (im_n > 0).float()  # 二值化处理
	d = torch.count_nonzero(tensor)  # 非零元素个数
	non_zero_area = tensor[0, 0].numpy() > 0  # 转化为bool型
	structure = np.ones((3, 3), dtype=int)  # 8连通
	labeled_array, num_features = label(non_zero_area, structure)  # 检查连通情况，labeled_array为被标记的tensor，num_features为连通区域个数
	# 0——计算连通区域的均值比方差
	sizes = np.bincount(labeled_array.ravel())  # 计算每个连通区域的大小
	largest_region_label = np.argmax(sizes[1:]) + 1  # 找到最大区域的标签（忽略标签0）
	largest_region = (labeled_array == largest_region_label)  # 获取最大区域的掩码
	if np.any(largest_region):  # 确保最大区域存在
		m_region_values_0 = original_image[largest_region]  # 最大连通区域元素集合ndarray
		m_region_values = original_image * largest_region  # 除最大连通区域都置零的tensor
		region_values = original_image - m_region_values  # 最大连通区域置零的tensor
		a = original_image.size  # 原图元素个数
		b = np.count_nonzero(region_values)  # 非最大连通区域的非零值个数
		c = a - m_region_values_0.size  # 除了最大连通区域之外的元素个数
		mean = np.mean(region_values[region_values > 0])  # *a/ (a-m_region_values_0) # 均值
		variance = np.var(region_values[region_values > 0]) + 0.00000001  # 方差
	else:
		mean = 0
		variance = 0
	out = mean / variance
	# 检查无效值
	if torch.isinf(torch.tensor(out)) or torch.isnan(torch.tensor(out)):
		out = torch.tensor(0.0)
	return out


import warnings
from skimage.filters import threshold_otsu
from skimage import filters
from skimage.morphology import remove_small_objects


def signal_snr(im):
	original_image = im.squeeze()  # 将形状变为 (1088, 1600)
	image = im.squeeze().numpy()  # 变为形状 (1088, 1600)
	thresh_sauvola = filters.threshold_sauvola(image, window_size=25, k=0.5)  # 0.3
	tensor = (im < torch.tensor(thresh_sauvola)).float()  # 二值化处理
	non_zero_area = tensor[0, 0].numpy() > 0  # 转化为bool型
	structure = np.ones((3, 3), dtype=int)  # 8连通
	labeled_array, num_features = label(non_zero_area, structure)  # 检查连通情况，labeled_array为被标记的tensor，num_features为连通区域个数
	sizes = np.bincount(labeled_array.ravel())
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		cleaned_image = remove_small_objects(labeled_array, min_size=200)
	final_binary_image = cleaned_image > 0
	binary_tensor = torch.tensor(final_binary_image).float()  # 转换为 float 类型 tensor
	signal_tensor = original_image * binary_tensor
	masked_values = (original_image.numpy())[final_binary_image == 1]
	mean_value = masked_values.mean() if masked_values.size > 0 else 0
	variance_value = masked_values.var(ddof=0) if masked_values.size > 0 else 0  # 使用 ddof=0 计算总体方差
	std_deviation = np.sqrt(variance_value) if masked_values.size > 0 else 0  # 避免空数组计算标准差
	mean_tensor = torch.tensor(mean_value).float()
	var_tensor = torch.tensor(variance_value).float()
	std_tensor = torch.tensor(std_deviation).float()
	out = 10 * torch.log10((mean_tensor * mean_tensor) / std_tensor)  # 信号
	# 检查无效值
	if torch.isinf(out) or torch.isnan(out):
		out = torch.tensor(0.0)
	return out


def background_snr(im):
	original_image = im.squeeze()  # 将形状变为 (1088, 1600)
	image = im.squeeze().numpy()  # 变为形状 (1088, 1600)
	thresh_sauvola = filters.threshold_sauvola(image, window_size=25, k=0.5)  # 0.3
	tensor = (im < torch.tensor(thresh_sauvola)).float()  # 二值化处理
	non_zero_area = tensor[0, 0].numpy() > 0  # 转化为bool型
	structure = np.ones((3, 3), dtype=int)  # 8连通
	labeled_array, num_features = label(non_zero_area, structure)  # 检查连通情况，labeled_array为被标记的tensor，num_features为连通区域个数
	sizes = np.bincount(labeled_array.ravel())
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", UserWarning)
		cleaned_image = remove_small_objects(labeled_array, min_size=200)
	final_binary_image = cleaned_image > 0
	# 背景区域
	background_values = (original_image.numpy())[final_binary_image == 0]  # 选择 mask 为 0 的区域
	background_mean = background_values.mean() if background_values.size > 0 else 0.00000001  # 避免空数组计算均值
	background_variance = background_values.var(
		ddof=0) if background_values.size > 0 else 0.00000001  # 使用 ddof=0 计算总体方差
	background_std_deviation = np.sqrt(background_variance) if background_values.size > 0 else 0  # 避免空数组计算标准差
	background_mean_tensor = torch.tensor(background_mean).float()
	background_var_tensor = torch.tensor(background_variance).float()
	background_std_tensor = torch.tensor(background_std_deviation).float()
	out = -torch.log10((background_mean_tensor * background_mean_tensor) / background_std_tensor)  # 背景
	# 检查无效值
	if torch.isinf(out) or torch.isnan(out):
		out = torch.tensor(0.0)
	return out
