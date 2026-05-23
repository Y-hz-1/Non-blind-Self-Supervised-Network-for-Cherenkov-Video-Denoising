import functools
import glob
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.optim as optim
from torch.utils.checkpoint import checkpoint

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).

    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks_0520':
        net = ResnetGenerator_0520(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_9blocks_0514':
        net = ResnetGenerator_0514(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'resnet_9blocks_f':
        net = ResnetGenerator_f(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
       X -------------------identity----------------------
       |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=None, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool) -- if this module is the outermost module
            innermost (bool) -- if this module is the innermost module
            norm_layer -- normalization layer (default: None)
            use_dropout (bool) -- if use dropout layers
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        use_bias = True  # Assuming use_bias for Conv2d layers

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)

        if outermost:
            # upconv = [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1)]
            upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(inner_nc * 2,outer_nc,kernel_size=3, stride=1, padding=1, bias=use_bias)]
            down = [downconv]
            up = [uprelu]+upconv+[nn.Sigmoid()]#
            model = down + [submodule] + up
        elif innermost:
            # upconv = [nn.ConvTranspose2d(inner_nc, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)]
            upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)]
            down = [downrelu, downconv]
            up = [uprelu] + upconv
            model = down + up
        else:
            # upconv = [nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                             kernel_size=4, stride=2,
            #                             padding=1, bias=use_bias)]
            upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(inner_nc * 2, outer_nc, kernel_size = 3, stride = 1, padding = 1, bias = use_bias)]
            down = [downrelu, downconv]
            up = [uprelu]+upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            a = self.model(x)
            return torch.cat([x, a], 1)

class ReFastDVDnet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_downs=7, ngf=8, norm_layer=None, use_dropout=False):
        super(ReFastDVDnet, self).__init__()

        # Create the U-Net structure without normalization layers
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=None, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=None, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=None)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=None)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=None)

        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             outermost=True, norm_layer=None)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, input):
        return self.model(input)

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


class ResnetGenerator_i(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_i, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_1 = []
        model0_2 = []
        model0_3 = []
        model0_4 = []
        model1 = []
        model2 = []
        model3 = []
        model4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        # mult = 2 ** 3
        # model0_4 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        #             norm_layer(ngf * mult * 2),
        #             nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(style_dim=1, num_features=ngf * mult)
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(style_dim=1, num_features=ngf * mult *2)
        model2 = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            norm_layer(int(ngf * mult / 2)),
            nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(style_dim=1, num_features=ngf * mult * 2)
        model3 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        mult = 2 ** (n_downsampling - 3)
        # self.adin4 = AdaIN(style_dim=2, num_features=ngf * mult * 2)
        # model4 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #           nn.Conv2d(ngf * mult * 2, int(ngf * mult / 2),
        #                     kernel_size=1, stride=1, bias=use_bias),
        #           norm_layer(int(ngf * mult / 2)),
        #           nn.LeakyReLU()]
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

    def forward(self, input,s):
        """Standard forward"""
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out = self.adin1(out3,s)
        out = self.model1(out)
        out2 = self.cbam3(out2)
        out = torch.cat((out2,out),dim=1)
        out = self.adin2(out,s)
        out = self.model2(out)
        out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.adin3(out, s)
        out = self.model3(out) #(3,64,512,512)
        out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out
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
class ResnetGenerator_s(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_s, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model1 = []
        model2 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(style_dim=2, num_features=ngf * mult)
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(style_dim=2, num_features=ngf * mult)
        model2 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=3, stride=1, padding=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        model2 += [nn.ReflectionPad2d(3)]
        model2 += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model2 += [nn.Sigmoid()]
        self.model = nn.Sequential(*model)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)

    def forward(self, input,s):
        """Standard forward"""
        out = self.model(input)
        out = self.adin1(out,s)
        out = self.model1(out)
        out = self.adin2(out,s)
        out = self.model2(out)
        return out


class ResnetGenerator_f(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_f, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        self.adin0 = AdaIN_fc(style_dim=1)
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN_out(num_features=int(ngf * mult / 2))
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias)]
        self.act1 = nn.LeakyReLU()
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN_out(num_features=int(ngf * mult / 2))
        model2 = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias)]
        self.act2 = nn.LeakyReLU()
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN_out(num_features=int(ngf * mult / 2))
        model3 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias)]
        self.act3 = nn.LeakyReLU()
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

    def forward(self, input,s):
        """Standard forward"""
        h = self.adin0(s)
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out = self.model1(out3)
        out = self.adin1(out,h)
        out = self.act1(out)
        out2 = self.cbam3(out2)
        out = torch.cat((out2,out),dim=1)
        out = self.model2(out)
        out = self.adin2(out,h)
        out = self.act2(out)
        out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.model3(out) #(3,64,512,512)
        out = self.adin3(out, h)
        out = self.act3(out)
        out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out
class ResnetGenerator_0514(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_0514, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(1, ngf * mult)   #512
        model1 = [#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(1, ngf * mult)  #256
        model2 = [
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            # norm_layer(int(ngf * mult / 2)),
            # nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(1, ngf * mult)  #128
        model3 = [#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        self.adin4 = AdaIN(1, int(ngf * mult/2))
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.adin = AdaIN_fc(1)   #0513加

        self.actv = nn.LeakyReLU(0.2)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        #self.model_np = nn.Sequential(*model_np)

    def forward(self, input, S):
        """Standard forward"""
        S = self.adin(S)
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out3 = self.conv(out3)
        out3 = self.adin1(out3,S)
        out3 = self.actv(out3)
        out = self.model1(out3)
        #out2 = self.cbam3(out2)
        out = self.adin2(out,S)
        out = self.actv(out)
        out = torch.cat((out2,out),dim=1)
        out = self.model2(out)
        out = self.adin3(out,S)
        out = self.actv(out)
        #out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.model3(out) #(3,64,512,512)
        out = self.adin4(out,S)
        out = self.actv(out)
        #out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out
class ResnetGenerator_0520(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_0520, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(1, ngf * mult)   #512
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  #nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(1, ngf * mult)  #256
        model2 = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            # norm_layer(int(ngf * mult / 2)),
            # nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(1, ngf * mult)  #128
        model3 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  #nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        self.adin4 = AdaIN(1, int(ngf * mult/2))
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)

        self.adin = AdaIN_fc(1)   #0513加

        self.actv = nn.LeakyReLU(0.2)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        #self.model_np = nn.Sequential(*model_np)

    def forward(self, input, S):
        """Standard forward"""
        S = self.adin(S)
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out3 = self.conv(out3)
        out3 = self.adin1(out3,S)
        out3 = self.actv(out3)
        out = self.model1(out3)
        #out2 = self.cbam3(out2)
        out = self.adin2(out,S)
        out = self.actv(out)
        out = torch.cat((out2,out),dim=1)
        out = self.model2(out)
        out = self.adin3(out,S)
        out = self.actv(out)
        #out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.model3(out) #(3,64,512,512)
        out = self.adin4(out,S)
        out = self.actv(out)
        #out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out
class ResnetGenerator_0424(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_0424, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(1, ngf * mult)
        model1 = [#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(1, ngf * mult)
        model2 = [
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            # norm_layer(int(ngf * mult / 2)),
            # nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(1, ngf * mult)
        model3 = [#nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        self.adin4 = AdaIN(1, int(ngf * mult/2))
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.actv = nn.LeakyReLU(0.2)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        #self.model_np = nn.Sequential(*model_np)

    def forward(self, input, S):
        """Standard forward"""
        # S = S.view(S.size(0), S.size(1), 1, 1)
        # S = S.repeat(input.size(0), 1, input.size(2), input.size(3))
        #gt = torch.cat([gt, S], dim=1)
        #noise_level = self.model_np(gt)
        #input = torch.cat([input, noise_level], dim=1)
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out3 = self.conv(out3)
        out3 = self.adin1(out3,S)
        out3 = self.actv(out3)
        out = self.model1(out3)
        #out2 = self.cbam3(out2)
        out = self.adin2(out,S)
        out = self.actv(out)
        out = torch.cat((out2,out),dim=1)
        out = self.model2(out)
        out = self.adin3(out,S)
        out = self.actv(out)
        #out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.model3(out) #(3,64,512,512)
        out = self.adin4(out,S)
        out = self.actv(out)
        #out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out


class ResnetGenerator_0420(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_0420, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_1 = []
        model0_2 = []
        model0_3 = []
        model0_4 = []
        model1 = []
        model2 = []
        model3 = []
        model4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_4 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(style_dim=1, num_features=ngf * mult)
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(style_dim=1, num_features=ngf * mult)
        model2 = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            # norm_layer(int(ngf * mult / 2)),
            # nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(style_dim=1, num_features=ngf * mult)
        model3 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  # norm_layer(int(ngf * mult / 2)),
                  # nn.LeakyReLU()
                  ]
        self.adin4 = AdaIN(style_dim=1, num_features=int(ngf * mult/2))
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf*2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.actv = nn.LeakyReLU(0.2)
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

    def forward(self, input,s):
        """Standard forward"""
        out0 = self.model(input) #(3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out3 = self.model0_4(out3)
        out3 = self.adin1(out3,s)
        out3 = self.actv(out3)
        out = self.model1(out3)
        #out2 = self.cbam3(out2)
        out = self.adin2(out,s)
        out = self.actv(out)
        out = torch.cat((out2,out),dim=1)
        out = self.model2(out)
        out = self.adin3(out, s)
        out = self.actv(out)
        #out1 = self.cbam2(out1)
        out = torch.cat((out1,out),dim=1)
        out = self.model3(out) #(3,64,512,512)
        out = self.adin4(out, s)
        out = self.actv(out)
        #out0 = self.cbam1(out0)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out

class ResnetGenerator_n(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=True, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_n, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model0_1 = []
        model0_2 = []
        model0_3 = []
        model1 = []
        model2 = []
        model3 = []
        model4 = []
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU()]  #nn.LeakyReLU()nn.ReLU(True)
        n_downsampling = 3
        #for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** 0
        model0_1= [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.LeakyReLU()]
        mult = 2 ** 1
        model0_2 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** 2
        model0_3 = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.LeakyReLU()]
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model0_3 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        # for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - 0)
        self.adin1 = AdaIN(style_dim=1, num_features=ngf * mult)
        model1 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        mult = 2 ** (n_downsampling - 1)
        self.adin2 = AdaIN(style_dim=1, num_features=ngf * mult *2)
        model2 = [
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                      kernel_size=1, stride=1, bias=use_bias),
            norm_layer(int(ngf * mult / 2)),
            nn.LeakyReLU()
        ]
        mult = 2 ** (n_downsampling - 2)
        self.adin3 = AdaIN(style_dim=1, num_features=ngf * mult * 2)
        model3 = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(ngf * mult *2, int(ngf * mult / 2),
                            kernel_size=1, stride=1, bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.LeakyReLU()]
        model4 = [nn.ReflectionPad2d(3)]
        model4 += [nn.Conv2d(ngf * 2, output_nc, kernel_size=7, padding=0)]
        model4 += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.model0_1 = nn.Sequential(*model0_1)
        self.model0_2 = nn.Sequential(*model0_2)
        self.model0_3 = nn.Sequential(*model0_3)
        # self.model0_4 = nn.Sequential(*model0_4)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)

    def forward(self, input, s):
        """Standard forward"""
        out0 = self.model(input)  # (3,64,512,512)
        out1 = self.model0_1(out0)
        out2 = self.model0_2(out1)
        out3 = self.model0_3(out2)
        out = self.adin1(out3, s)
        out = self.model1(out)
        out = torch.cat((out2, out), dim=1)
        out = self.adin2(out, s)
        out = self.model2(out)
        out = torch.cat((out1, out), dim=1)
        out = self.adin3(out, s)
        out = self.model3(out)  # (3,64,512,512)
        out = torch.cat((out0, out), dim=1)
        out = self.model4(out)
        return out


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer
        self.norm1 = AdaIN(style_dim=1, num_features=6)
    def forward(self, input, metrics_vector_I):
        input = self.norm1(input, metrics_vector_I.float())
        """Standard forward"""
        out = self.model(input)
        #out = (out-out.min)/(out.max()-out.min())
        return out

# class AdaIN(nn.Module):
#     def __init__(self, style_dim, num_features):
#         super().__init__()
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         self.fc = nn.Linear(style_dim, num_features*2) #*2
#
#     def forward(self, x, s):
#         h = self.fc(s)
#         h = h.view(h.size(0), h.size(1), 1, 1)
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         a = self.norm(x)
#         b = (1 + gamma) * a + beta
#         return b
# class AdaIN(nn.Module):
#     def __init__(self, style_dim, num_features):
#         super().__init__()
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         self.fc_0 = nn.Linear(style_dim,8)
#         self.fc_1 = nn.Linear(8,16)
#         self.fc_2 = nn.Linear(16,32)
#         self.fc_3 = nn.Linear(32,64)
#         #self.fc = nn.Linear(style_dim, num_features*2) #*2
#         self.fc = nn.Linear(64, num_features*2) #*2
#         self.act = nn.LeakyReLU(0.2)
#     def forward(self, x, s):
#         h = self.fc_0(s)
#         h = self.fc_1(h)
#         h = self.fc_2(h)
#         h = self.fc_3(h)
#         h = self.fc(h)
#         h = h.view(h.size(0), h.size(1), 1, 1)
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         beta = self.act(beta)
#         a = self.norm(x)
#         b = (1 + gamma) * a + beta
#         return b
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

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv2 = nn.Conv2d(inner_nc * 2, outer_nc,
                      kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv1, upconv2, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv2 = nn.Conv2d(inner_nc, outer_nc,
                                kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv1, upconv2, upnorm]
            model = down + up
        else:
            upconv1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            upconv2 = nn.Conv2d(inner_nc * 2, outer_nc,
                                kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv1,upconv2, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        # if self.innermost:
        #     b = self.norm1(x,metrics_vector_I)
        #     b = self.model(b)
        #     return torch.cat([x, b], 1)
        if self.outermost:
            #x = self.norm1(x, metrics_vector_I.float())
            return self.model(x)
        else:   # add skip connections
            b = self.model(x)
            return torch.cat([x, b], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)
