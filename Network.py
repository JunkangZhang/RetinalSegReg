import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from utils_pytorch import init_weights

class vgg16_features(nn.Module):
	def __init__(self, cuda_id=0, trainable=False):
		super(vgg16_features, self).__init__()
		self.vgg16 = models.vgg16(pretrained=True)

		if trainable==False:
			for param in self.vgg16.parameters():
				param.requires_grad = False

		features = list(self.vgg16.features)[:23]
		self.vgg16_features = nn.ModuleList(features).eval()

		self.vgg16_features.cuda(cuda_id)

	def forward(self, x, verbose=False):
		res = []
		for i, layer in enumerate(self.vgg16_features):
			x = layer(x)
			if i in [3, 8, 15, 22]:
				res.append(x)
		return res

class DRIU_novgg_siamese_feat(nn.Module):
	def __init__(self, with_relu=True, num_layers=1, num_filters=64, cuda_id=0, upsampling='deconv'):
		super(DRIU_novgg_siamese_feat, self).__init__()
		# self.features = vgg16_features()
		self.upsampling = upsampling
		num_filters_div = num_filters//4

		self.conv1 = nn.Sequential(
			nn.Conv2d(64, num_filters_div, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(),
		)
		self.conv1.apply(init_weights)
		self.conv1.cuda(cuda_id)

		conv2_list = [nn.Conv2d(128, num_filters_div, kernel_size=3, stride=1, padding=1, bias=True),
		              nn.ReLU()]
		if upsampling=='deconv':
			conv2_list.append( nn.ConvTranspose2d(num_filters_div, num_filters_div, kernel_size=4, stride=2, padding=1) )
			conv2_list.append( nn.ReLU() )
		self.conv2 = nn.Sequential(*conv2_list)
		self.conv2.apply(init_weights)
		self.conv2.cuda(cuda_id)

		conv3_list = [nn.Conv2d(256, num_filters_div, kernel_size=3, stride=1, padding=1, bias=True),
		              nn.ReLU()]
		if upsampling=='deconv':
			conv3_list.append( nn.ConvTranspose2d(num_filters_div, num_filters_div, kernel_size=8, stride=4, padding=2) )
			conv3_list.append( nn.ReLU() )
		self.conv3 = nn.Sequential(*conv3_list)
		self.conv3.apply(init_weights)
		self.conv3.cuda(cuda_id)

		conv4_list = [nn.Conv2d(512, num_filters_div, kernel_size=3, stride=1, padding=1, bias=True),
		              nn.ReLU()]
		if upsampling == 'deconv':
			conv4_list.append( nn.ConvTranspose2d(num_filters_div, num_filters_div, kernel_size=16, stride=8, padding=4) )
			conv4_list.append( nn.ReLU() )
		self.conv4 = nn.Sequential(*conv4_list)
		self.conv4.apply(init_weights)
		self.conv4.cuda(cuda_id)

		conv_feat_list = []
		for i in range(num_layers):
			conv_feat_list.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=True))
			if i<num_layers-1:
				conv_feat_list.append(nn.ReLU())
		if with_relu == True:
			conv_feat_list.append(nn.ReLU())

		self.conv_feat = nn.Sequential(*conv_feat_list)
		self.conv_feat.apply(init_weights)
		self.conv_feat.cuda(cuda_id)

	def forward(self, features):
		x1 = self.conv1(features[0])
		x2 = self.conv2(features[1])
		x3 = self.conv3(features[2])
		x4 = self.conv4(features[3])
		if self.upsampling != 'deconv':
			x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
			x3 = F.interpolate(x3, scale_factor=4, mode='bilinear')
			x4 = F.interpolate(x4, scale_factor=8, mode='bilinear')
		x_cat = torch.cat((x1, x2, x3, x4), dim=1)
		feat = self.conv_feat(x_cat)
		return feat

class DRIU_novgg_siamese_seg(nn.Module):
	def __init__(self, input_channels=64, output_channels=1, with_relu=False, with_sigmoid=True, cuda_id=0):
		super(DRIU_novgg_siamese_seg, self).__init__()

		conv_pred_list = []
		if with_relu == True:
			conv_pred_list.append(nn.ReLU())

		conv_pred_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
		if with_sigmoid == True:
			conv_pred_list.append(nn.Sigmoid())

		self.conv_pred=nn.Sequential(*conv_pred_list)
		self.conv_pred.apply(init_weights)
		self.conv_pred.cuda(cuda_id)

	def forward(self, feat):
		prob = self.conv_pred(feat)
		return prob

class STN_Flow_relative(nn.Module):
	def __init__(self, size=None, cuda_id=0):
		'''
		:param size: (H, W) or (y, x)
		:param cuda_id:
		'''
		super(STN_Flow_relative, self).__init__()
		self.size = size
		self.cuda_device = torch.device('cuda:%d'%cuda_id)
		if size is not None:
			self.create_base()

	def create_base(self):
		xv, yv = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
		self.base_x = torch.cuda.FloatTensor(xv[np.newaxis, np.newaxis, :, :], device=self.cuda_device)
		self.base_y = torch.cuda.FloatTensor(yv[np.newaxis, np.newaxis, :, :], device=self.cuda_device)

	def forward(self, flow, x):
		_, _, h, w = flow.size()
		if self.size is None or (self.size[0]!=h or self.size[1]!=w) or flow.device!=self.cuda_device:
			self.size = (h, w)
			self.cuda_device = flow.device
			self.create_base()

		flow_x = (flow[:, 0, :,:] + self.base_x) / (self.size[1]-1) * 2 - 1
		flow_y = (flow[:, 1, :,:] + self.base_y) / (self.size[0]-1) * 2 - 1
		grid = torch.cat((flow_x, flow_y), dim=1)
		grid = grid.permute(0, 2, 3, 1)
		x = F.grid_sample(x, grid)
		return x
