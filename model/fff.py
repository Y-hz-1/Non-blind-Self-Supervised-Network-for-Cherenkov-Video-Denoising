"""
Definition of the FastDVDnet model
"""
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F

class CvBlock(nn.Module):
	'''(Conv2d => BN => ReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, \
					  kernel_size=3, padding=1, groups=num_in_frames, bias=False),#*(1+1)
			nn.BatchNorm2d(num_in_frames*self.interm_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
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
				nn.ReLU(inplace=True),
				#nn.LeakyReLU(negative_slope=0.2,inplace=True),#negative_slope=0.1,
				CvBlock(in_ch, out_ch)
			)
		else:
			self.convblock = nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
				norm_layer(out_ch),
				#nn.InstanceNorm2d(out_ch),
				nn.ReLU(inplace=True),
				#nn.LeakyReLU(inplace=True),#negative_slope=0.1,
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
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True),
				#CvBlock(in_ch, out_ch),
			)
		else:
			self.convblock = nn.Sequential(
				CvBlock(in_ch, in_ch),
				nn.Conv2d(in_ch, out_ch*4, kernel_size=3, padding=1, bias=True),
				nn.PixelShuffle(2)
			)
	def forward(self, x):
		return self.convblock(x)

class OutputCvBlock_1(nn.Module):
	'''Conv2d => BN => ReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock_1, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(in_ch),
			nn.ReLU(inplace=True),
			#nn.Dropout(p=0.5),
			# nn.ReflectionPad2d((3,3,3,3)),
			# nn.Conv2d(in_ch, out_ch, kernel_size=7,bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			#nn.Sigmoid()
			#nn.Tanh()
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
			nn.ReLU(inplace=True),
			# nn.ReflectionPad2d((3, 3, 3, 3)),
			# nn.Conv2d(in_ch, out_ch, kernel_size=7, bias=False),
			#nn.BatchNorm2d(out_ch),
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(out_ch),
			# nn.ReLU(inplace=True),
			#nn.Tanh()
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
	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		#a=torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)
		#x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)
		# Residual
		x = in1-1
		#x = self.res0(x)
		x = (x - x.min()) / (x.max() - x.min())
		#x = (x - x.min()) / (x.max() - x.min()) if x.max() > 1.0 or x.min() < 0.0 else x
		return x

class DenBlock_1(nn.Module):
	""" Definition of the denosing block of FastDVDnet.
	Inputs of constructor:
		num_input_frames: int. number of input frames
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=3 RGB)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(DenBlock_1, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock_1(in_ch=self.chs_lyr0, out_ch=1)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block
		#a=torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)
		#x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)
		# Residual
		#x = self.res0(in1- x)
		x = in1-1
		x = (x - x.min()) / (x.max() - x.min())
		#x = (x - x.min()) / (x.max() - x.min()) if x.max() > 1.0 or x.min() < 0.0 else x
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
		self.temp2 = DenBlock_1(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		#(x0, x1, x2, x3, x4) = tuple(x[:, 3*m:3*m+3, :, :] for m in range(self.num_input_frames))  #.unsqueeze(1)
		(x0, x1, x2, x3, x4) = tuple(x[:, m:m+1, :, :] for m in range(self.num_input_frames))
		# First stage
		x20 = self.temp1(x0, x1, x2)
		x21 = self.temp1(x1, x2, x3)
		x22 = self.temp1(x2, x3, x4)
		#Second stage
		x = self.temp2(x20, x21, x22)
		#sigmoid = nn.Sigmoid()
		#x = sigmoid(x)
		#x = torch.clamp(x, 0., 1.)
		#x = (x - x.min()) / (x.max() - x.min())
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
        feature_layer = 28   #34   28
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
        self.weight = [0.0,0.5,0.0,0.0,1.0]
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