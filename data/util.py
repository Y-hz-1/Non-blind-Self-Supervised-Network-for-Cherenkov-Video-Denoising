"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc
"""
import os
import subprocess
import glob
import logging
from random import choices # requires Python >= 3.6
import numpy as np
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2
import torch
from model import model_fast_1
import torch.nn.functional as F
from data.dataloader_n import outliers_rmv_circle
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tensorboardX import SummaryWriter

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff') # Supported image types

def normalize_augment(datain, ctrl_fr_idx):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	def transform(sample):
		# define transformations
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
								 std=(5/255.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		# define transformations and their frequency, then pick one.
		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
		transf = choices(aug_list, w_aug)

		# transform all images in array
		return transf[0](sample)

	img_train = datain.permute(0, 1, 4, 2, 3)
	# convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
	img_train = img_train.view(img_train.size()[0], -1, \
							   img_train.size()[-2], img_train.size()[-1]) / 255.

	#augment
	img_train = transform(img_train)

	# extract ground truth (central frame)
	#gt_train = img_train[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
	gt_train = img_train[:, ctrl_fr_idx, :, :].unsqueeze(1)
	return img_train, gt_train

def init_logging(argdict):
	"""Initilizes the logging and the SummaryWriter modules
	"""
	if not os.path.exists(argdict['log_dir']):
		os.makedirs(argdict['log_dir'])
	writer = SummaryWriter(argdict['log_dir'])
	logger = init_logger(argdict['log_dir'], argdict)
	return writer, logger

def get_imagenames(seq_dir, pattern=None):
	""" Get ordered list of filenames
	"""
	files = []
	for typ in IMAGETYPES:
		files.extend(glob.glob(os.path.join(seq_dir, typ)))

	# filter filenames
	if not pattern is None:
		ffiltered = []
		ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
		files = ffiltered
		del ffiltered

	# sort filenames alphabetically
	files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
	return files

def open_sequence_1(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=1000):
	r""" Opens a sequence of images and expands it to even sizes if necesary
	Args:
		fpath: string, path to image sequence
		gray_mode: boolean, True indicating if images is to be open are in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
		max_num_fr: maximum number of frames to load
	Returns:
		seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	# Get ordered list of filenames
	files = get_imagenames(seq_dir)
	seq_list = []
	S_list = []
	print("\tOpen sequence in folder: ", seq_dir)
	for fpath in files[0:max_num_fr]:
		img,S, expanded_h, expanded_w = open_image(fpath,\
												   gray_mode=gray_mode,\
												   expand_if_needed=expand_if_needed,\
												   expand_axis0=False)
		seq_list.append(img)
		S_list.append(S)
	seq = torch.stack(seq_list, dim=0)
	seq_S = torch.stack(S_list, dim=0)#.squeeze(1)
	return seq,seq_S, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
	r""" Opens an image and expands it if necesary
	Args:
		fpath: string, path of image file
		gray_mode: boolean, True indicating if image is to be open
			in grayscale mode
		expand_if_needed: if True, the spatial dimensions will be expanded if
			size is odd
		expand_axis0: if True, output will have a fourth dimension
	Returns:
		img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
			if expand_axis0=False, the output will have a shape CxHxW.
			The image gets normalized gets normalized to the range [0, 1].
		expanded_h: True if original dim H was odd and image got expanded in this dimension.
		expanded_w: True if original dim W was odd and image got expanded in this dimension.
	"""
	if not gray_mode:
		# Open image as a CxHxW torch.Tensor
		img = cv2.imread(fpath)
		# from HxWxC to CxHxW, RGB image
		img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
	else:
		# from HxWxC to  CxHxW grayscale image (C=1)
		img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
		img = np.expand_dims(img, axis=2)

	if expand_axis0:
		img = np.expand_dims(img, 0)
	# Handle odd sizes
	expanded_h = False
	expanded_w = False
	sh_im = img.shape
	if expand_if_needed:
		if sh_im[-2]%2 == 1:
			expanded_h = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
			else:
				img = np.concatenate((img, \
					img[:, -1, :][:, np.newaxis, :]), axis=1)


		if sh_im[-1]%2 == 1:
			expanded_w = True
			if expand_axis0:
				img = np.concatenate((img, \
					img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
			else:
				img = np.concatenate((img, \
					img[:, :, -1][:, :, np.newaxis]), axis=2)

	if normalize_data:
		img,S = normalize(img,5)
	return img ,S , expanded_h, expanded_w

def batch_psnr(img, imclean, data_range):
	r"""
	Computes the PSNR along the batch dimension (not pixel-wise)

	Args:
		img: a `torch.Tensor` containing the restored image
		imclean: a `torch.Tensor` containing the reference image
		data_range: The data range of the input image (distance between
			minimum and maximum possible values). By default, this is estimated
			from the image data-type.
	"""
	img_cpu = img.data.cpu().numpy()#.astype(np.float32)
	imgclean = imclean.data.cpu().numpy()#.astype(np.float32)
	psnr = 0
	for i in range(img_cpu.shape[0]):
		psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], \
					   data_range=data_range)
	return psnr/img_cpu.shape[0]

def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
	r"""Converts a torch.autograd.Variable to an OpenCV image

	Args:
		invar: a torch.autograd.Variable
		conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
	Returns:
		a HxWxC uint8 image
	"""
	assert torch.max(invar) <= 1.0

	size4 = len(invar.size()) == 4
	if size4:
		nchannels = invar.size()[1]
	else:
		nchannels = invar.size()[0]

	if nchannels == 1:
		if size4:
			res = invar.data.cpu().numpy()[0, 0, :]
		else:
			res = invar.data.cpu().numpy()[0, :]
		#res = (res*65535.).clip(0, 65535).astype(np.uint16)
		res = (res * 255.).clip(0, 255).astype(np.uint16)  #65535
	elif nchannels == 3:
		if size4:
			res = invar.data.cpu().numpy()[0]
		else:
			res = invar.data.cpu().numpy()
		res = res.transpose(1, 2, 0)
		res = (res*255.).clip(0, 255).astype(np.uint8)
		if conv_rgb_to_bgr:
			res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
	else:
		raise Exception('Number of color channels not supported')
	return res

def get_git_revision_short_hash():
	r"""Returns the current Git commit.
	"""
	return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def init_logger(log_dir, argdict):
	r"""Initializes a logging.Logger to save all the running parameters to a
	log file

	Args:
		log_dir: path in which to save log.txt
		argdict: dictionary of parameters to be logged
	"""
	from os.path import join

	logger = logging.getLogger(__name__)
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(log_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	try:
		logger.info("Commit: {}".format(get_git_revision_short_hash()))
	except Exception as e:
		logger.error("Couldn't get commit number: {}".format(e))
	logger.info("Arguments: ")
	for k in argdict.keys():
		logger.info("\t{}: {}".format(k, argdict[k]))

	return logger

def init_logger_test(result_dir):
	r"""Initializes a logging.Logger in order to log the results after testing
	a model

	Args:
		result_dir: path to the folder with the denoising results
	"""
	from os.path import join

	logger = logging.getLogger('testlog')
	logger.setLevel(level=logging.INFO)
	fh = logging.FileHandler(join(result_dir, 'log.txt'), mode='w+')
	formatter = logging.Formatter('%(asctime)s - %(message)s')
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	return logger

def close_logger(logger):
	'''Closes the logger instance
	'''
	x = list(logger.handlers)
	for i in x:
		logger.removeHandler(i)
		i.flush()
		i.close()

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
trans_totensor = transforms.ToTensor()
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

    return cropped_img
import random
def random_crop_with_coords(img, crop_size=512):
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

    return cropped_img, top, left
def random_crop_with_coords_1(img, crop_size):
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
    cropped_img = cropped_img.unsqueeze(0)
    cropped_img = F.interpolate(cropped_img, size=(512, 512), mode='bilinear', align_corners=False)
    return cropped_img.squeeze(0), top, left
def normalize(data,n):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	data = data.astype(np.float32)
	trans_normalize = transforms.Normalize((0.5,), (0.5,))
	#
	data = data.squeeze(-1)
	data = torch.from_numpy(data).unsqueeze(0).unsqueeze(1)
	data = outliers_rmv_circle(data,10,10,16)

	#data = outliers_rmv_circle(data,10,75,16)
	#data = outliers_rmv_circle(data, 5, 10, 16)
	#平场矫正
	data = data[0,0].unsqueeze(-1).numpy()
	ff_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/flat_cor/AVG_trc411flatfield.tif"
	df_path = "/home/fastdvdnet-master-unsupervised/fastdvdnet-master/flat_cor/AVG_trc411darkfield.tif"
	ff = cv2.imread(ff_path, cv2.IMREAD_UNCHANGED)
	ff = np.expand_dims(ff, axis=2)
	df = cv2.imread(df_path, cv2.IMREAD_UNCHANGED)
	df = np.expand_dims(df, axis=2)
	denominator = ff - df
	#de = denominator[:,:,0]
	a = (denominator.sum().item())/(1088*1600)
	#a = np.median(denominator)
	data = a*((data - df) / denominator)
	data = np.round(data.clip(0,65535))
	data = data.astype(np.uint16)
	data = data.squeeze(-1)
	data = torch.from_numpy(data).unsqueeze(0).unsqueeze(1)

	# m = data[0,0]
	# plt.imshow(m, cmap='gray', vmin=0, vmax=255)
	# plt.axis('off')
	# plt.title("Grayscale Visualization")
	# plt.show()
	# data,_,_ = random_crop_with_coords_1(data,1088)
	# data = data[0]
	# n = 5
	S = torch.full((1, 1),n, dtype=torch.float32)  #80
	# data = (data/512).clamp(0, 1.)
	# data = trans_normalize(data)
	data = data.squeeze(0)
	return data,S

def svd_orthogonalization(lyr):
	r"""Applies regularization to the training by performing the
	orthogonalization technique described in the paper "An Analysis and Implementation of
	the FFDNet Image Denoising Method." Tassano et al. (2019).
	For each Conv layer in the model, the method replaces the matrix whose columns
	are the filters of the layer by new filters which are orthogonal to each other.
	This is achieved by setting the singular values of a SVD decomposition to 1.

	This function is to be called by the torch.nn.Module.apply() method,
	which applies svd_orthogonalization() to every layer of the model.
	"""
	classname = lyr.__class__.__name__
	if classname.find('Conv') != -1:
		weights = lyr.weight.data.clone()
		c_out, c_in, f1, f2 = weights.size()
		dtype = lyr.weight.data.type()

		# Reshape filters to columns
		# From (c_out, c_in, f1, f2)  to (f1*f2*c_in, c_out)
		weights = weights.permute(2, 3, 1, 0).contiguous().view(f1*f2*c_in, c_out)

		try:
			# SVD decomposition and orthogonalization
			mat_u, _, mat_v = torch.svd(weights)
			weights = torch.mm(mat_u, mat_v.t())

			lyr.weight.data = weights.view(f1, f2, c_in, c_out).permute(3, 2, 0, 1).contiguous().type(dtype)
		except:
			pass
	else:
		pass

def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary


	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = v

	return new_state_dict
