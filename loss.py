import torch
import torch.nn as nn

from utils_pytorch import gram_matrix

class TVLoss_sq(nn.Module):
	def __init__(self):
		super(TVLoss_sq, self).__init__()

	def forward(self, x):
		diff_x = torch.mean(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2))
		diff_y = torch.mean(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2))
		return diff_x + diff_y

class TVLoss_L1(nn.Module):
	def __init__(self):
		super(TVLoss_L1, self).__init__()

	def forward(self, x):
		diff_x = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
		diff_y = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
		return diff_x + diff_y

class StyleLoss(nn.Module):
	def __init__(self):
		super(StyleLoss, self).__init__()
		self.mse = nn.MSELoss()

	def forward(self, features, g_ref):
		loss = 0
		for feat, g_r in zip(features, g_ref):
			g = gram_matrix(feat)
			loss += self.mse(g, g_r)
		return loss

class MSELossMask(nn.Module):
	def __init__(self):
		super(MSELossMask, self).__init__()
		# self.mse = nn.MSELoss()

	def forward(self, res, gt, mask=None):
		if mask is not None:
			if type(mask) is list:
				mask_ = mask[0]
				for i in range(1, len(mask)):
					mask_ = mask_ * mask[i]
				mask = mask_
			res = res*mask
			gt = gt*mask
		# loss = self.mse(res, gt)
		loss = torch.mean((res-gt)**2)
		return loss
