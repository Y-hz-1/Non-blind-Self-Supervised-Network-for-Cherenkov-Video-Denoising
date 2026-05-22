"""
Dataset related functions
"""
import os
import glob
import torch
from torch.utils.data.dataset import Dataset
from data.util import open_sequence_1
import matplotlib.pyplot as plt

NUMFRXSEQ_VAL = 15	# number of frames of each sequence to include in validation dataset
VALSEQPATT = '*' # pattern for name of validation sequence

class ValDataset(Dataset):
	"""Validation dataset. Loads all the images in the dataset folder on memory.
	"""
	def __init__(self, valsetdir=None, valsetdir_gt=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
		self.gray_mode = gray_mode

		# Look for subdirs with individual sequences
		seqs_dirs = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		seqs_dirs_gt = sorted(glob.glob(os.path.join(valsetdir, VALSEQPATT)))
		# open individual sequences and append them to the sequence list
		sequences = []
		sequences_gt = []
		s = []
		s_gt = []
		for i in range(0,len(seqs_dirs)):
			seq_dir = seqs_dirs[i]
			seq_dir_gt = seqs_dirs_gt[i]
			seq,S, _, _ = open_sequence_1(seq_dir, gray_mode, expand_if_needed=False, \
									  max_num_fr=num_input_frames)
			seq_gt,S_gt, _, _ = open_sequence_1(seq_dir_gt, gray_mode, expand_if_needed=False, \
			 						  max_num_fr=num_input_frames)
			# for seq_dir in seqs_dirs:
			# 	seq, _, _ = open_sequence(seq_dir, gray_mode, expand_if_needed=False, \
			# 					 max_num_fr=num_input_frames)
			# seq is [num_frames, C, H, W]
			sequences.append(seq)
			sequences_gt.append(seq_gt)
			s.append(S)
			s_gt.append(S_gt)
		self.sequences = sequences
		self.sequences_gt = sequences_gt
		self.s = s
		self.s_gt = s_gt

	def __getitem__(self, index):
		return torch.from_numpy(self.sequences[index])

	def __len__(self):
		return len(self.sequences)
