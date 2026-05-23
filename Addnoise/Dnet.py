import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(self, in_nc, nf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = True
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(nf * nf_mult_prev, nf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(nf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(nf * nf_mult_prev, nf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(nf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(nf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()]
            # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input):
        return self.model(input)
# class NLayerDiscriminator(nn.Module):
#     """Defines a PatchGAN discriminator"""
#     def __init__(self, in_nc, nf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             nf (int)        -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator, self).__init__()
#         use_bias = True
#         kw = 4
#         padw = 1
#         sequence = [nn.Conv2d(in_nc, nf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv2d(nf * nf_mult_prev, nf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(nf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv2d(nf * nf_mult_prev, nf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(nf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         # 计算展平后的特征图大小
#         self.model = nn.Sequential(*sequence)
#         self._initialize_conv_layers()
#
#         # 计算展平后的特征图大小
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, in_nc, 1088, 1600)
#             self.flattened_size = self._get_flattened_size(dummy_input)
#
#         # 定义全连接层
#         self.fc = nn.Linear(self.flattened_size, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def _initialize_conv_layers(self):
#         """初始化卷积层的参数"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#
#     def _get_flattened_size(self, x):
#         """计算展平后的特征图大小"""
#         with torch.no_grad():
#             x = self.model(x)
#             return int(np.prod(x.size()))
#
#     def forward(self, input):
#         x = self.model(input)
#         x = x.view(x.size(0), -1)  # 展平
#         x = self.fc(x)
#         #x = self.sigmoid(x)
#         return x
# torch.cuda.set_device(1)
# #device = torch.device("cuda:1")  # 指定gpu1为主GPU
# torch.backends.cudnn.benchmark = True  # CUDNN optimization
# # 初始化模型
# model = NLayerDiscriminator(in_nc=1).cuda()
#
# # 使用 DataParallel 并行化模型
# model = nn.DataParallel(model, device_ids=[0], output_device=0)
# # 示例输入张量
# x = torch.randn(4, 1, 1088, 1600)
#
# # 前向传播
# x = model(x)
# print(output.shape)  # 输出形状
# print(kernel.shape)  # 内核形状
