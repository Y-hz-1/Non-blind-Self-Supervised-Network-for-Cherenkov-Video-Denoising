"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
class M_U_Net(nn.Module):
	def __init__(self, in_channel=3, out_channel=1):
		super(M_U_Net, self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(in_channel, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 90, 5, 1, 2),
			nn.BatchNorm2d(90),
			nn.ReLU(),
			nn.Conv2d(90, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU()
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 2, 1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			*(
					(
						nn.Conv2d(64, 64, 3, 1, 1),
						nn.BatchNorm2d(64),
						nn.ReLU()
					) * 2
			)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			*(
					(
						nn.Conv2d(128, 128, 3, 1, 1),
						nn.BatchNorm2d(128),
						#nn.Dropout(0.5),
						nn.ReLU(),
					) * 4
			),
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(256, 64, 3, 1, 1)

			#nn.PixelShuffle(2)
		)
		self.conv4 = nn.Sequential(
			*(
					(
						nn.Conv2d(64, 64, 3, 1, 1),
						nn.BatchNorm2d(64),
						nn.ReLU()
					) * 2
			),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(128, 32, 3, 1, 1)
			#nn.PixelShuffle(2)
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(32, 32, 5, 1, 2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, out_channel, 3, 1, 1),
			nn.Tanh()
			#nn.Sigmoid()
		)
	# Dictionary to store the features
		global features
		features = {}

		# Register hooks
		self._register_hooks()

	def _register_hooks(self):
		self.conv1.name = 'conv1'
		self.conv2.name = 'conv2'
		self.conv3.name = 'conv3'
		self.conv4.name = 'conv4'
		self.conv5.name = 'conv5'

		self.conv1.register_forward_hook(hook_fn)
		self.conv2.register_forward_hook(hook_fn)
		self.conv3.register_forward_hook(hook_fn)
		self.conv4.register_forward_hook(hook_fn)
		self.conv5.register_forward_hook(hook_fn)

	def forward(self, data, ref):
		"""
			:param data: noisy frames
			:param ref: reference frame that is the middle frame of noisy frames
			:return:
			"""
		conv1 = self.conv1(data)
		conv2 = self.conv2(conv1)
		conv3 = self.conv3(conv2)
		conv4 = self.conv4(conv3 + conv2)
		conv5 = self.conv5(conv4 + conv1)
		return conv5 + ref



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 将特征图沿宽和高维度求均值
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(b, c)

        # 通过两个全连接层生成注意力权重
        avg_out = self.fc2(F.relu(self.fc1(avg_pool))).view(b, c, 1, 1)
        max_out = self.fc2(F.relu(self.fc1(max_pool))).view(b, c, 1, 1)

        # 结合两个注意力权重
        out = torch.sigmoid(avg_out + max_out)
        return out * x  # 将注意力权重应用于输入特征图

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
			#nn.BatchNorm2d(out_ch),
			nn.InstanceNorm2d(out_ch),
			nn.LeakyReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 15
		self.convblock = nn.Sequential(
			# nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, \
			# 		  kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.ReflectionPad2d(2),
			nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, \
					  kernel_size=5, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			#nn.InstanceNorm2d(num_in_frames*self.interm_ch),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, 90,kernel_size=3, padding=1, groups=num_in_frames, bias=False),
			nn.BatchNorm2d(90),
			#nn.InstanceNorm2d(90),
			nn.LeakyReLU(inplace=True),
			nn.ReflectionPad2d(2),
			nn.Conv2d(90, out_ch, kernel_size=5, bias=False),#num_in_frames*self.interm_ch
			nn.BatchNorm2d(out_ch),
			#nn.InstanceNorm2d(out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => BN => ReLU)*2'''
	def __init__(self, in_ch, out_ch, use_maxpool=False, norm_layer=nn.BatchNorm2d):
		super(DownBlock, self).__init__()
		if use_maxpool:
			self.convblock = nn.Sequential(
				nn.MaxPool2d(2),
				norm_layer(in_ch),
				#nn.InstanceNorm2d(in_ch),
				nn.LeakyReLU(inplace=True),
				#nn.ReLU(inplace=True),#negative_slope=0.1,
				CvBlock(in_ch, out_ch)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
				nn.BatchNorm2d(out_ch),
				#nn.InstanceNorm2d(out_ch),
				nn.LeakyReLU(inplace=True),
				#nn.Dropout(0.5),
				CvBlock(out_ch, out_ch)
			)

	def forward(self, x):
		return self.convblock(x)
class UpBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch, use_interpolate=False):
		super(UpBlock, self).__init__()
		if use_interpolate:
			self.convblock = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
				#nn.InstanceNorm2d(out_ch),
				#nn.BatchNorm2d(out_ch),
				nn.LeakyReLU(inplace=True),
				#nn.Dropout(0.5),
				CvBlock(out_ch, out_ch),
			)
		else:
			self.convblock = nn.Sequential(
				CvBlock(in_ch, in_ch),
				nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=True),
				nn.PixelShuffle(2)
			)
	def forward(self, x):
		return self.convblock(x)
class OutputCvBlock(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			#nn.InstanceNorm2d(in_ch),
			nn.LeakyReLU(inplace=True),
			# nn.ReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			# nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,bias=False),
			#nn.BatchNorm2d(out_ch),
			#nn.InstanceNorm2d(out_ch),
			nn.Tanh()
			#nn.Sigmoid()
		)
	def forward(self, x):
		return self.convblock(x)
class OutputCvBlock_o(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock_o, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.LeakyReLU(inplace=True),
			#nn.ReflectionPad2d(2),
			#nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.InstanceNorm2d(out_ch),
			nn.Tanh()
			#nn.Sigmoid()
		)
	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=3,out=False):
		super(DenBlock, self).__init__()
		self.chs_lyr0_0 = 16
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128
		self.out =out
		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0_0)
		self.downc0_0 = DownBlock(in_ch=self.chs_lyr0_0, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.upc0 = UpBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr0_0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0_0, out_ch=1)
		self.outc_o = OutputCvBlock_o(in_ch=self.chs_lyr0_0, out_ch=1)
		#self.channel_attention = ChannelAttention(in_channels=128)
		self.reset_params()
		# Dictionary to store the features
		global features
		features = {}

		# Register hooks
		self._register_hooks()

	def _register_hooks(self):
		self.inc.name = 'InputCvBlock'
		self.downc0_0.name = 'DownBlock_0_0'
		self.downc0.name = 'DownBlock_0'
		self.downc1.name = 'DownBlock_1'
		self.upc2.name = 'UpBlock_2'
		self.upc1.name = 'UpBlock_1'
		self.upc1.name = 'UpBlock_0'
		self.outc.name = 'OutputCvBlock'
		self.outc_o.name = 'OutputCvBlock_o'

		self.inc.register_forward_hook(hook_fn)
		self.downc0_0.register_forward_hook(hook_fn)
		self.downc0.register_forward_hook(hook_fn)
		self.downc1.register_forward_hook(hook_fn)
		self.upc2.register_forward_hook(hook_fn)
		self.upc1.register_forward_hook(hook_fn)
		self.upc0.register_forward_hook(hook_fn)
		self.outc.register_forward_hook(hook_fn)
		self.outc_o.register_forward_hook(hook_fn)

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0_0(x0)
		x2 = self.downc0(x1)
		x3 = self.downc1(x2)
		# 应用注意力机制
		#x3 = self.channel_attention(x3)
		# Upsampling
		x3= self.upc2(x3)
		#x1 = self.upc1(x1)
		x3 = self.upc1(x3+x2)
		x1 = self.upc0(x3+x1)
		x = self.outc_o(x0 + x1)
		# Residual
		#x = (x - x.min()) / (x.max() - x.min())
		#x = in1 + x
		if self.out == False:
			# Estimation
			#x = self.outc_o(x0 + x1)
			x = in1+x
			x = (x - x.min()) / (x.max() - x.min())
			#x = (x-x.mean())/x.std()
		else:
			# Estimation
			#x = self.outc_o(x0 + x1)
			x = in1+x
			#x = (x - x.min()) / (x.max() - x.min())
		return x



class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3,out=True) #num_input_frames=3
		self.temp3 = M_U_Net()
		self.temp4 = M_U_Net()
		# Init weights
		self.reset_params()
	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)
	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4) = tuple(x[:, m:m+1, :, :] for m in range(self.num_input_frames))
		# x20 = self.temp1(x0, x1, x2)
		# x21 = self.temp1(x1, x2, x3)
		# x22 = self.temp1(x2, x3, x4)
		# x = self.temp2(x20, x21, x22)
		x20 = self.temp3(torch.cat((x0,x1,x2), dim=1),x1)
		x21 = self.temp3(torch.cat((x1, x2, x3), dim=1), x2)
		x22 = self.temp3(torch.cat((x2, x3, x4), dim=1), x3)
		# x22 = (x22 - x22.min()) / (x22.max() - x22.min())
		# x21 = (x21 - x21.min()) / (x21.max() - x21.min())
		# x20 = (x20 - x20.min()) / (x20.max() - x20.min())
		x = self.temp4(torch.cat((x20, x21, x22), dim=1), x21)
		# #Plot feature maps
		# def plot_feature_maps(features, layer_name):
		# 	if layer_name in features:
		# 		feature_maps = features[layer_name]
		# 		num_feature_maps = feature_maps.shape[1]
		# 		plt.figure(figsize=(15, 15))
		# 		for i in range(min(num_feature_maps, 64)):  # Limit to first 64 feature maps
		# 			plt.subplot(8, 8, i + 1)
		# 			plt.imshow(feature_maps[0, i].cpu().numpy(), cmap='gray')
		# 			plt.axis('off')
		# 		plt.title(layer_name)
		# 		plt.show()
		#
		# # Example usage to plot feature maps from each layer
		# for layer_name in features.keys():
		# 	plot_feature_maps(features, layer_name)
		# #x=x/2
		x = (x - x.min()) / (x.max() - x.min())
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
#Perceptual loss
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
    gpu_ids = [0,1]
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34   #34   28
    netF = VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)

    netF.eval()  # No need to train
    return netF
#perceptual loss——2
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
        self.weight = [0.0,1.0,1.0,1.0,0.0]
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