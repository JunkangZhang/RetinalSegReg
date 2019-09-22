import torch
import torch.nn as nn
import os

def init_weights(m):
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
		print(m)
		nn.init.xavier_normal(m.weight, gain=1)
		if m.bias is not None:
			nn.init.constant(m.bias, 0)
	elif type(m) in [None]:
		pass

def save_checkpoint(state_dict, file):
	model_dir = os.path.dirname(file)
	# make dir if needed (should be non-empty)
	if model_dir!='' and not os.path.exists(model_dir):
		os.makedirs(model_dir)
	torch.save(state_dict, file)

def normalize_batch(batch):
	'''https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py'''
	# normalize using imagenet mean and std
	mean = batch.new_tensor([0.485, 0.456, 0.406], device=batch.get_device()).repeat(batch.size()[0], 1, 1, 1).permute(0,3,1,2)
	std = batch.new_tensor([0.229, 0.224, 0.225], device=batch.get_device()).repeat(batch.size()[0], 1, 1, 1).permute(0,3,1,2)
	return (batch - mean) / std

def gram_matrix(y):
	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

def get(x):
	return x.cpu().detach().numpy()
