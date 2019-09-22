import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_pytorch import init_weights

class UNet_ContractingBlock(nn.Module):
	def __init__(self, in_channels, out_channels, max_channels=65536, downsampling=None, cuda_id=0):
		'''
		:param in_channels:
		:param out_channels:
		:param downsampling: None, 'pooling', 'conv'
		:param cuda_id:
		'''
		super(UNet_ContractingBlock, self).__init__()
		self.downsampling = downsampling
		self.conv = nn.Sequential(
			nn.Conv2d(min(in_channels, max_channels), min(out_channels, max_channels), kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(),
			nn.Conv2d(min(out_channels, max_channels), min(out_channels, max_channels), kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU()
		)
		self.conv.apply(init_weights)
		self.conv.cuda(cuda_id)
		if self.downsampling is not None:
			if self.downsampling.lower().startswith('p') or self.downsampling==True:
				self.down_conv = nn.MaxPool2d(kernel_size=2, stride=2)
			else:
				self.down_conv = nn.Sequential(
					nn.Conv2d(min(out_channels, max_channels), min(out_channels*2, max_channels), kernel_size=4, stride=2, padding=1, bias=True),
					nn.ReLU(),
				)
			self.down_conv.apply(init_weights)
			self.down_conv.cuda(cuda_id)

	def forward(self, x):
		x = self.conv(x)
		if self.downsampling is not None:
			x_d = self.down_conv(x)
		else:
			x_d = x
		return x_d, x

class UNet_UpsamplingBlock(nn.Module):
	def __init__(self, in_channels, out_channels, max_channels=65536, deconv_ksize=2, deconv_stride=2, deconv_pad=0, in_flow=0, cuda_id=0, with_relu=True):
		super(UNet_UpsamplingBlock, self).__init__()
		# self.convtransposed = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
		self.conv_transposed = nn.Sequential(
			nn.ConvTranspose2d(min(in_channels, max_channels), min(out_channels, max_channels),
			                   kernel_size=deconv_ksize, stride=deconv_stride, padding=deconv_pad),
			nn.ReLU()
		)
		self.conv_transposed.apply(init_weights)
		self.conv_transposed.cuda(cuda_id)

		conv_seq = [
			nn.Conv2d(min(out_channels, max_channels)*2+in_flow, min(out_channels, max_channels), kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(),
			nn.Conv2d(min(out_channels, max_channels), min(out_channels, max_channels), kernel_size=3, stride=1, padding=1, bias=True),
			# nn.ReLU()
		]
		if with_relu==True:
			conv_seq.append(nn.ReLU())
		self.conv = nn.Sequential(*conv_seq)
		self.conv.apply(init_weights)
		self.conv.cuda(cuda_id)

	def forward(self, x, x_u):
		x = self.conv_transposed(x)

		if type(x_u) is not list:
			x_u = [x_u]

		x = torch.cat( tuple([x]+x_u), dim=1)
		x = self.conv(x)
		return x

class UNetFlow(nn.Module):
	def __init__(self, input_channels=2, output_channels=2, down_scales=5, output_scale=2, num_filters_base=64, max_filters=512,
	             downsampling='conv', sigmoid=False, cuda_id=0, filter_ch='2x'):
		super(UNetFlow, self).__init__()

		self.scales = down_scales
		self.output_scale = output_scale
		self.contracting_blocks = nn.ModuleList()
		self.upsampling_blocks = nn.ModuleList()
		in_channels = input_channels
		out_channels = num_filters_base

		for i in range(self.scales):
			self.contracting_blocks.append(UNet_ContractingBlock(in_channels, out_channels, max_channels=max_filters, downsampling=downsampling, cuda_id=cuda_id))
			if filter_ch=='2x':
				if downsampling == 'conv':
					in_channels = out_channels*2
				else:
					in_channels = out_channels
				out_channels *= 2
			else: # 'fixed'
				in_channels = out_channels

		self.lowest_block = UNet_ContractingBlock(in_channels, out_channels, max_channels=max_filters, downsampling=None, cuda_id=cuda_id)

		for i in range(self.scales-self.output_scale):
			if filter_ch == '2x':
				in_channels = out_channels
				out_channels = out_channels//2
				in_flow = 0
			else: # 'fixed'
				in_flow = num_filters_base
			self.upsampling_blocks.append(UNet_UpsamplingBlock(in_channels, out_channels, max_channels=max_filters, in_flow=in_flow,
			                                              deconv_ksize=4, deconv_stride=2, deconv_pad=1, cuda_id=cuda_id))

		in_channels = min(out_channels, max_filters) # num_filters_base
		out_channels = output_channels
		if sigmoid == True:
			self.conv_output = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
				nn.Sigmoid()
			)
		else:
			self.conv_output = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
			)
		self.conv_output.apply(init_weights)
		self.conv_output.cuda(cuda_id)

	def forward(self, x):
		contracting_res = []
		for i in range(self.scales):
			x, x_u = self.contracting_blocks[i](x)
			contracting_res.append(x_u)

		x, x_u = self.lowest_block(x)

		for i in range(self.scales-self.output_scale):
			x = self.upsampling_blocks[i](x, contracting_res[-i-1])

		x = self.conv_output(x)
		return x
