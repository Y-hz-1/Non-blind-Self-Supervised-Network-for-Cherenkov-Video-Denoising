"""
Trains a FastDVDnet model.
"""


import time
import argparse
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import pandas as pd
import torch.optim as optim
from model.models_unsupervised import UnsupervisedDVDnet
from model.discriminator import define_D
from data.dataset import ValDataset
from model.model_fast_1 import define_F,LossNetwork,hubber_loss
from model.loss import SSIM,GANLoss,CharbonnierLoss,PSNR
from data.dataloader_n import UnalignedDataset
from data.aligned_dataset import AlignedDataset
from torch.utils.data import DataLoader
from model.CBDNet_model import CBDNet
import random
import numpy as np
from Addnoise.Addnoise_model import define_G, define_D
from data.util import svd_orthogonalization, close_logger, init_logging
from train_common import resume_training_1, lr_scheduler, log_train_psnr, \
					log_train_psnr_2,validate_and_log, save_model_checkpoint

import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
# # 固定随机种子
# def set_seed(seed):
#     random.seed(seed)                        # Python random seed
#     np.random.seed(seed)                     # Numpy random seed
#     torch.manual_seed(seed)                  # PyTorch random seed (for CPU)
#     torch.cuda.manual_seed(seed)             # PyTorch random seed (for single GPU)
#     torch.cuda.manual_seed_all(seed)         # PyTorch random seed (for multi-GPU)
#     torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
#     torch.backends.cudnn.benchmark = False     # Disable CuDNN's auto-tuner
#
# # 设置随机种子，确保可复现性
# set_seed(42)


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def main(**args):
	# Load dataset
	print('> Loading datasets ...')
	class Options:
		def __init__(self):
			self.dataroot = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/clinical_sequence'
			self.gtroot = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/gt_img'
			self.batch_size = 1  # 批量大小
			self.max_dataset_size = float("inf")  # 最大数据集大小
			self.input_nc = 1  # 输入图像的通道数（例如，对于灰度图像为1）
			self.output_nc = 1  # 输出图像的通道数（例如，对于灰度图像为1）
			self.num_threads = 8  # 数据加载时的线程数量
			self.epoch_size = 4000
			self.patch_size = 512
	class Options1:
		def __init__(self):
			self.dataroot = '/home/vipuser/project/pytorch-CycleGAN-and-pix2pix-master/datasets/che_end'
			self.phase = 'train'
			self.max_dataset_size = float("inf")
			self.load_size = 286
			self.crop_size = 256
			self.input_nc = 1
			self.output_nc = 1
			self.direction = 'AtoB'

	opt = Options()
	opt1 = Options1()
	dataset_deg = AlignedDataset(opt1)
	loader_deg = DataLoader(dataset_deg, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)
	dataset = UnalignedDataset(opt)
	loader_train = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads)

	num_minibatches = len(loader_train)
	ctrl_fr_idx = (args['temp_patch_size'] - 1) // 2
	print("\t# of training samples: %d\n" % int(args['max_number_patches']))
	dataset_val = ValDataset(valsetdir=args['valset_dir'], valsetdir_gt=args['valset_gt_dir'], gray_mode=True)
	# Init loggers
	writer, logger = init_logging(args)
	torch.cuda.set_device(0)

	device = torch.device("cuda:0")  # 指定gpu1为主GPU
	torch.backends.cudnn.benchmark = True  # CUDNN optimization
	# Create model
	model = UnsupervisedDVDnet().to(device)
	model = nn.DataParallel(model, device_ids=[0], output_device=0)
	model_path = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/0812/latest_net_G.pth' #12-2 1230 0312
	model_path_D = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/0812/latest_net_D.pth' #12-2 1230 0312
	# encoder_path = '/home/fastdvdnet-master-unsupervised/fastdvdnet-master/model_encoder.pth'
	# state_encoder = torch.load(encoder_path)
	# model.load_state_dict(state_encoder)
	#reFastDVDnet = define_G(input_nc=1, output_nc=1, ngf=64, init_type='normal', netG='resnet_9blocks',norm='instance').to(device)  # resnet_6blocks unet_256
	reFastDVDnet = define_G(input_nc=1, output_nc=1, ngf=64, init_type='normal', netG='resnet_9blocks_0520', use_dropout=True, norm='batch').to(device)  # resnet_6blocks unet_256
	noise_D_Deg = define_D(input_nc=2, ndf=64, netD='basic', n_layers_D=5, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[0])
	state_dict = torch.load(model_path)
	reFastDVDnet.load_state_dict(state_dict)
	state_dict_D = torch.load(model_path_D)
	# 如果保存的模型没有 DataParallel 的 module. 前缀，但当前模型有
	from collections import OrderedDict
	new_state_dict = OrderedDict()
	for k, v in state_dict_D.items():
		if not k.startswith('module.'):  # 如果键没有 module. 前缀
			name = 'module.' + k  # 添加 module. 前缀
		else:
			name = k
		new_state_dict[name] = v
	noise_D_Deg.load_state_dict(new_state_dict)
	noise_D_Deg.eval()
	# for param in reFastDVDnet.parameters():
	# 	param.requires_grad = False
	for param in noise_D_Deg.parameters():
		param.requires_grad = False
	for param in reFastDVDnet.model.parameters():
		param.requires_grad = False
	for param in reFastDVDnet.model0_1.parameters():
		param.requires_grad = False
	for param in reFastDVDnet.model0_2.parameters():
		param.requires_grad = False
	for param in reFastDVDnet.model0_3.parameters():
		param.requires_grad = False
	#reFastDVDnet.eval()
	net_D_B = define_D(input_nc=1, ndf=64, init_type='normal', netD='basic',norm='batch').to(device)
	#loss
	criterion_GAN = GANLoss().to(device)
	criterion_1 = nn.L1Loss().to(device)
	criterion = nn.MSELoss().to(device) #reduction='sum'
	criterion_SSIM = SSIM().to(device)
	criterion_PSNR = PSNR().to(device)
	net_F = LossNetwork().to(device)
	net_F_1 = define_F().to(device)
	net_F.eval()
	# Optimizer
	optimizer_G_Deg = optim.Adam(reFastDVDnet.parameters(), lr=args['lr_2'], betas=(0.9, 0.999))
	optimizer_D_Deg = optim.Adam(noise_D_Deg.parameters(), lr=args['lr_2_D'], betas=(0.9, 0.999))
	optimizer_G = optim.Adam(model.parameters(), lr=args['lr_1'], betas=(0.9, 0.999))#, weight_decay=1e-4
	optimizer_D = optim.Adam(net_D_B.parameters(), lr=args['lr_1_D'], betas=(0.9, 0.999))
	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)
	# Resume training or start anew
	start_epoch, training_params = resume_training_1(args, model, net_D_B,optimizer_G,optimizer_D)
	#tv loss
	def total_variation(x,TVLoss_weight=100):
		# dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
		# dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
		n = x.numel()
		h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
		w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
		a = TVLoss_weight * (h_tv + w_tv) / n
		return 	a	 #dx + dy
	#dice loss
	def dice_coeff(pred, target):
		smooth = 1.
		num = pred.size(0)
		m1 = pred.view(num, -1)  # Flatten
		m2 = target.view(num, -1)  # Flatten
		intersection = (m1 * m2).sum()
		return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

	# 定义冻结状态
	def freeze_reFastDVDnet_layers():
		"""冻结 reFastDVDnet 的指定层"""
		for param in reFastDVDnet.model.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_1.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_2.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_3.parameters():
			param.requires_grad = False

	def unfreeze_reFastDVDnet_trainable():
		"""解冻 reFastDVDnet 中需要训练的部分"""
		# 解冻所有参数，然后重新冻结指定层
		for param in reFastDVDnet.parameters():
			param.requires_grad = True
		# 重新冻结指定层
		for param in reFastDVDnet.model.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_1.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_2.parameters():
			param.requires_grad = False
		for param in reFastDVDnet.model0_3.parameters():
			param.requires_grad = False
	# Training
	start_time = time.time()
	save_loss_dir = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/loss_logs"
	os.makedirs(save_loss_dir, exist_ok=True)
	loss_records = []
	global_iter = 0
	deg_log_step = 0
	for epoch in range(start_epoch, args['epochs']):
		# Set learning rate
		current_lr_D, current_lr, reset_orthog = lr_scheduler(epoch, args)
		if reset_orthog:
			training_params['no_orthog'] = True
		for param_group in optimizer_G.param_groups:
			param_group["lr_1"] = current_lr
		for param_group in optimizer_D.param_groups:
			param_group["lr_1_D"] = current_lr_D
		for param_group in optimizer_G.param_groups:
			param_group["lr_2"] = current_lr
		for param_group in optimizer_D.param_groups:
			param_group["lr_2_D"] = current_lr_D
		print('\nlearning rate %f' % current_lr)

		'''train_denoise'''
		# if epoch % 2 == 0:
		# print(f"\n>>> Start den_train at epoch {epoch}")
		set_requires_grad(reFastDVDnet, False)  # 完全冻结 reFastDVDnet
		set_requires_grad(model, True)  # 解冻主模型
		set_requires_grad(net_D_B, True)  # 解冻判别器
		for i, data in enumerate(loader_train, 0):
			model.train()
			reFastDVDnet.eval()  # 设置为评估模式
			imgn_train = data['A'].squeeze(0).view(-1,5,opt.patch_size,opt.patch_size)
			S = data['S'].squeeze(0)
			#print(S)
			gt = data['gt'].squeeze(0).to('cuda')
			m = data['m']
			N, _, H, W = imgn_train.size()
			imgn_train = imgn_train.cuda(non_blocking=True)
			#im = im.cuda(non_blocking=True)
			torch.cuda.synchronize()
			start_time = time.time()
			out_train = model(imgn_train,S.cuda())
			torch.cuda.synchronize()
			end_time = time.time()
			infer_time = end_time - start_time
			print(f"model(input_batch) inference time: {infer_time:.6f} s")
			##D_d
			set_requires_grad(net_D_B, True)
			optimizer_D.zero_grad()
			pred_d_fake = net_D_B(out_train.detach())
			loss_D_fake = criterion_GAN(pred_d_fake, False)
			pred_d_real = net_D_B(gt)
			loss_D_real = criterion_GAN(pred_d_real, True)
			loss_D = (loss_D_real + loss_D_fake) * 0.5 *10
			loss_D.backward()
			optimizer_D.step()
			##G_d
			set_requires_grad(net_D_B, False)
			optimizer_G.zero_grad()
			re_out_train = reFastDVDnet(out_train, S.cuda())
			#loss_G_5 = criterion_1(re_out_train,imgn_train[:,2,:,:].unsqueeze(1))
			loss_G_5 = criterion(re_out_train,imgn_train[:,2,:,:].unsqueeze(1))
			#loss_G_2,_ = net_F(re_out_train.repeat(1, 3, 1, 1) ,imgn_train[:,2,:,:].unsqueeze(1).repeat(1, 3, 1, 1) )
			a = net_F_1(re_out_train)
			b = net_F_1(imgn_train[:,2,:,:].unsqueeze(1))
			# a = net_F_1(out_train)
			# b = net_F_1(gt.cuda())
			loss_G_2 = criterion(a,b)
			loss_G_3 = total_variation(out_train)
			#loss_G_5 = hubber_loss(imgn_train[:,2,:,:].unsqueeze(1),re_out_train)
			loss_G_GAN = criterion_GAN(net_D_B(out_train),True)
			loss_G_5 = (1-criterion_SSIM(re_out_train,imgn_train[:,2,:,:].unsqueeze(1)))
			loss_G_4 = criterion_PSNR(re_out_train,imgn_train[:,2,:,:].unsqueeze(1))
			#loss_G = loss_G_GAN + loss_G_5*80 + loss_G_2
			#loss_G = loss_G_GAN*0.5 + loss_G_4*0.05 + loss_G_2*0.5
			loss_den = loss_G_4*0.5 + loss_G_2*10
			loss_G = loss_G_GAN + loss_den
			loss_records.append({
				"iter": global_iter,
				"epoch": epoch,
				"batch": i,
				"loss_D": loss_D.item(),
				"loss_den": loss_den.item(),
				"loss_G_GAN": loss_G_GAN.item(),
				"loss_G":loss_G.item(),
			})

			loss_G.backward()
			optimizer_G.step()
			if (training_params['step']+1) % args['save_every'] == 0:
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)
				log_train_psnr( S,
								m,
								out_train, \
								re_out_train,\
								imgn_train, \
								gt,\
								loss_D, \
								loss_G_GAN, \
								loss_den, \
								loss_G, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		'''deg_train'''
		# if epoch  == 100:
		# 	print(f"\n>>> Start den_train at epoch {epoch}")
		set_requires_grad(model, False)  # 完全冻结主模型
		set_requires_grad(net_D_B, True)  # 冻结判别器
		unfreeze_reFastDVDnet_trainable()
		for i, data in enumerate(loader_deg, 0):
			A = data['A'].squeeze(0).to('cuda')
			B = data['B'].squeeze(0).to('cuda')
			S = data['S'].squeeze(0).to('cuda')
			reFastDVDnet.train()  # 设置为训练模式
			model.eval()  # 主模型设为评估模式
			optimizer_G_Deg.zero_grad()
			fake_B = reFastDVDnet(A, S)
			# D
			set_requires_grad(noise_D_Deg, True)
			optimizer_D_Deg.zero_grad()
			fake_AB = torch.cat((A, fake_B),1)  # we use conditional GANs; we need to feed both input and output to the discriminator
			pred_fake = noise_D_Deg(fake_AB.detach())
			loss_D_fake = criterion_GAN(pred_fake, False)
			real_AB = torch.cat((A, B), 1)
			pred_real = noise_D_Deg(real_AB)
			loss_D_real = criterion_GAN(pred_real, True)
			loss_D_2 = (loss_D_fake + loss_D_real) * 0.5
			loss_D_2.backward()
			optimizer_D_Deg.step()
			#G
			set_requires_grad(noise_D_Deg, False)
			optimizer_G_Deg.zero_grad()
			fake_AB = torch.cat((A, fake_B), 1)
			pred_fake = noise_D_Deg(fake_AB)
			loss_G_GAN = criterion_GAN(pred_fake, True)
			# self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
			# Second
			fake_B_1 = net_F_1(fake_B)
			real_B_1 = net_F_1(B)
			loss_G_Lp = criterion(fake_B_1, real_B_1)  # *self.opt.lambda_L1
			# .loss_G_L1_1 = self.criterionGAN_P(self.fake_B,self.real_B)
			loss_G_L1_1 = criterion_PSNR(fake_B, B)
			loss_G_2 = loss_G_GAN * 10 + loss_G_Lp * 10 + loss_G_L1_1
			loss_records.append({
				"iter": global_iter,
				"epoch": epoch,
				"batch": i,
				"loss_D_2": loss_D_2.item(),
				"loss_G_2": loss_G_2.item(),
			})
			loss_G_2.backward()
			optimizer_G_Deg.step()
			freeze_reFastDVDnet_layers()
			deg_log_step += 1

			if deg_log_step % args['save_every'] == 0:
				print(f">>> log_train_psnr_2 执行 | "f"deg_log_step={deg_log_step}, epoch={epoch}, deg_batch={i}")
				if not training_params['no_orthog']:
					model.apply(svd_orthogonalization)
				log_train_psnr_2( S,
								m,
								fake_B, \
								A,\
								B,\
								loss_D_2, \
								loss_G_2,\
								loss_G_GAN, \
								writer, \
								epoch, \
								i, \
								num_minibatches, \
								training_params)
			# update step counter
			training_params['step'] += 1

		model.eval()
		# Validation and log images
		validate_and_log(
						model_temp=model, \
						re_model = reFastDVDnet,\
						dataset_val=dataset_val, \
						valnoisestd=args['val_noiseL'], \
						temp_psz=args['temp_patch_size'], \
						writer=writer, \
						epoch=epoch, \
						lr=current_lr, \
						logger=logger
						)
		# save model and checkpoint
		training_params['start_epoch'] = epoch + 1
		save_model_checkpoint(model, args, optimizer_G, training_params, epoch)

		df_loss = pd.DataFrame(loss_records)
		df_loss.to_csv(os.path.join(save_loss_dir, "loss_log.csv"), index=False)
	# Print elapsed time
	elapsed_time = time.time() - start_time
	print('Elapsed time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))

	# Close logger file
	close_logger(logger)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Train the denoiser")
	#Training parameters
	parser.add_argument("--batch_size", type=int, default=32, 	\
					 help="Training batch size")
	parser.add_argument("--epochs", "--e", type=int, default=60, \
					 help="Number of total training epochs")
	parser.add_argument("--resume_training", "--r", action='store_true',\
						help="resume training from a previous checkpoint")
	parser.add_argument("--milestone", nargs=2, type=int, default=[30, 50], \
						help="When to decay learning rate; should be lower than 'epochs'")
	parser.add_argument("--lr_1", type=float, default=2e-4, \
					 help="Initial learning rate") #4
	parser.add_argument("--lr_1_D", type=float, default=2e-4, \
						help="Initial learning rate")  # 上一次测试的2
	parser.add_argument("--lr_2", type=float, default=1e-4, \
					 help="Initial learning rate") #4
	parser.add_argument("--lr_2_D", type=float, default=1e-4, \
						help="Initial learning rate")  # 4
	parser.add_argument("--no_orthog", action='store_true',\
						help="Don't perform orthogonalization as regularization")
	parser.add_argument("--save_every", type=int, default=50,\
						help="Number of training steps to log psnr and perform \
						orthogonalization")
	parser.add_argument("--save_every_epochs", type=int, default=1,\
						help="Number of training epochs to save state")
	parser.add_argument("--noise_ival", nargs=2, type=int, default=[5, 55], \
					 help="Noise training interval")
	parser.add_argument("--val_noiseL", type=float, default=25, \
						help='noise level used on validation set')
	# Preprocessing parameters
	parser.add_argument("--patch_size", "--p", type=int, default=256, help="Patch size")
	parser.add_argument("--temp_patch_size", "--tp", type=int, default=5, help="Temporal patch size")
	parser.add_argument("--max_number_patches", "--m", type=int, default=32, \
						help="Maximum number of patches")
	# Dirs
	parser.add_argument("--log_dir", type=str, default="logs", \
					 help='path of log files')
	parser.add_argument("--trainset_dir", type=str, default='/home/fastdvdnet-master-unsupervised/fastdvdnet-master/clinical_sequence', \
					 help='path of trainset')
	#/home/yuhuizhen_1/cherenkov/fastdvdnet-master-unsupervised/fastdvdnet-master/val_seq
	parser.add_argument("--valset_dir", type=str, default='/home/fastdvdnet-master-unsupervised/fastdvdnet-master/val_seq', \
						 help='path of validation set')
	parser.add_argument("--valset_gt_dir", type=str,
						default='/root/fast_dvd/fastdvdnet-master/val_seq', \
						help='path of validation set')
	argspar = parser.parse_args()

	# Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noise_ival[0] /= 255.
	argspar.noise_ival[1] /= 255.

	print("\n### Training FastDVDnet denoiser model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	main(**vars(argspar))
