import torch
import torch.nn as nn
import numpy as np
import cv2
import os, time, psutil, shutil

from utils import *
from utils_image import *
from utils_pytorch import *
import Network, UNet
import loss
import pytorch_ssim
from globalsetting import globalsetting

class options():
	def __init__(self):
		gs = globalsetting()
		self.cuda_id = [0,0]
		self.save_route = os.path.join(gs.ckpt_path, 'temp')

		self.flow_feat_scales = 5 # total scales of features
		self.flow_scale_times = 2 # output scale for flow map, 0 for non-scale, [0, flow_feat_scales-1]
		self.flow_base_filters = 128
		self.flow_max_filters = 128
		self.flow_downsample = 'pool'

		self.learing_rate = 1e-3
		self.rand_offset = 5
		self.totalvar = 2e-5 # 2e-3 for L2sq
		self.ssim = 1e-5
		self.photometric = 1e-3

		self.smoothness_type = 'L1' # 'L2sq'
		self.rot_inv_type = 'rot' # 'rot': rotation of 180 degree / 'neg': negative image

		self.debug = False # if True, save training results every epoch
		self.interval_test = 100
		self.interval_save_ckpt = 100
		self.total_epoch = int(5e4)

		self.checkpoint_recovery = None

		self.grid_size = 40 # int(np.ceil(768./5))

		self.style_target = os.path.join(gs.data_path, 'HRF/manual1/12_h.tif')
		self.dataset_path = os.path.join(gs.data_path, 'Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients')
		self.csv_path = os.path.join(gs.proj_path, 'FFAPCFIDP_affine.csv')
		self.save_codes = ''
		self.rand_offset_folder = None # not for training

		self.mode = 'train' # 'eval' on specified data / 'test' on outside data
		self.save_im = False # save image during 'train' or 'eval'

class RetSegReg():
	def __init__(self, opt=options()):
		self.opt = opt
		if type(opt.cuda_id) is not list:
			opt.cuda_id = [opt.cuda_id, opt.cuda_id]
		self.cuda_id = opt.cuda_id
		self.cuda_device = [torch.device('cuda:%d'%id) for id in opt.cuda_id]
		self.process = psutil.Process(os.getpid())

		if opt.save_route is not None and len(opt.save_route) > 0:
			self.save_route = opt.save_route
			os.makedirs(self.save_route, exist_ok=True)

		if opt.mode.lower()[:2] in ['tr', 'ev']:
			self.LoadData()
		else:
			self.SIZE = None

		self.InitializeNetwork()
		if opt.mode.lower()[:2] in ['tr']:
			if self.opt.save_codes is not None and len(self.opt.save_codes) > 0:
				self.save_codes()
			self.InitializeTraining()
			self.GetStyleReference()
			self.loss_all_dict = {'train_ax': [], 'test_ax': []}

		self.epoch_st = 0
		self.epoch = -1
		if opt.checkpoint_recovery is not None:
			self.LoadCheckpoint(opt.checkpoint_recovery)

	def save_codes(self):
		code_dir = os.path.join(self.save_route, 'code')
		os.makedirs(code_dir, exist_ok=True)
		sp = [x.strip() for x in self.opt.save_codes.split(',')]
		for name in sp:
			if name.endswith('/'):
				self.save_files(name, os.path.join(code_dir, name))
			else:
				shutil.copyfile(name, os.path.join(code_dir, name))

	def save_files(self, src, dst):
		os.makedirs(dst, exist_ok=True)
		files = os.listdir(src)
		for f in files:
			if os.path.isdir(f):
				self.save_files(os.path.join(src,f), os.path.join(dst,f))
			else:
				shutil.copyfile(os.path.join(src,f), os.path.join(dst,f))


	def LoadData(self):
		self.ds = FFAPCFIDP_icip(dataset_path=self.opt.dataset_path, csv_path=self.opt.csv_path)
		self.SIZE = (self.ds.src_train.shape[1], self.ds.src_train.shape[2])

	def InitializeNetwork(self):
		self.vgg = Network.vgg16_features(cuda_id=self.cuda_id[0])
		self.stn = Network.STN_Flow_relative(size=self.SIZE, cuda_id=self.cuda_id[0])
		self.stn_destroy = Network.STN_Flow_relative(size=self.SIZE, cuda_id=self.cuda_id[0])

		self.models = {}
		self.models['seg_feat_src'] = Network.DRIU_novgg_siamese_feat(with_relu=False, cuda_id=self.cuda_id[0])
		self.models['seg_feat_tgt'] = Network.DRIU_novgg_siamese_feat(with_relu=False, cuda_id=self.cuda_id[0])
		self.models['seg_pred'] = Network.DRIU_novgg_siamese_seg(with_relu=True, with_sigmoid=True, cuda_id=self.cuda_id[0])
		self.models['flow'] = UNet.UNetFlow(down_scales=self.opt.flow_feat_scales, output_scale=self.opt.flow_scale_times,
		                                    num_filters_base=self.opt.flow_base_filters, max_filters=self.opt.flow_max_filters,
		                                    input_channels=128, output_channels=2,
		                                    downsampling=self.opt.flow_downsample, cuda_id=self.cuda_id[1])

		for k in self.models.keys():
			print(self.models[k])
		for k in self.models.keys():
			num_params = 0
			for param in self.models[k].parameters():
				num_params += param.numel()
			print('[Network %s] Total number of parameters : %.3f M' % (k, num_params / 1e6))

	def InitializeTraining(self):
		self.loss_funcs = {}
		self.loss_funcs['style_gram'] = loss.StyleLoss()
		self.loss_funcs['cmp_src_tgt'] = loss.MSELossMask()
		if self.opt.smoothness_type.lower() == 'l1':
			self.loss_funcs['flow_smooth'] = loss.TVLoss_L1()
		else:
			self.loss_funcs['flow_smooth'] = loss.TVLoss_sq()
		self.loss_funcs['cmp_self'] = loss.MSELossMask()
		self.loss_funcs['ssim'] = pytorch_ssim.SSIM()

		model_params = []
		for k in self.models.keys():
			model_params += self.models[k].parameters()
		self.optimizer = torch.optim.Adam(model_params, lr=self.opt.learing_rate)

	def GetStyleReference(self):
		ref_im = np.float32(cv2.imread(self.opt.style_target, cv2.IMREAD_GRAYSCALE))/255.
		ref_im = cv2.resize(ref_im, dsize=(0,0), fx=576./ref_im.shape[0], fy=576./ref_im.shape[0])
		ref_im = ref_im[:, (ref_im.shape[1]-768)//2:(ref_im.shape[1]-768)//2+768]
		cv2.imwrite(os.path.join(self.save_route, 'style_reference.png'), np.uint8(ref_im * 255))

		ref_batch = torch.cuda.FloatTensor(ndarray2tensor(ref_im), device=self.cuda_device[0]).repeat((1, 3, 1, 1))
		features_ref = self.vgg(normalize_batch(ref_batch))
		self.g_ref = [gram_matrix(feat) for feat in features_ref]

	def LoadCheckpoint(self, checkpoint_recovery):
		if checkpoint_recovery is not None:
			print('checkpoint:', checkpoint_recovery)
			checkpoint = torch.load(checkpoint_recovery, map_location='cuda:%d' % self.cuda_id[0])
			for k in self.models.keys():
				self.models[k].load_state_dict(checkpoint['model_' + k])
			print('network parameters loaded ')
			self.epoch_st = checkpoint['epoch']

			if self.opt.mode.lower().startswith('tr'):
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				for k in checkpoint.keys():
					if k.startswith('train') or k.startswith('test'):
						self.loss_all_dict[k] = np.array(checkpoint[k]).tolist()
				np.random.set_state(checkpoint['np_state'])
				print('training state loaded ')

	def Save(self, phase=None):
		self.GetResults()
		self.SaveResults(phase=phase)

	def GetResults(self, additional_keys=None):
		self.save_dict = {}
		save_keys = ['src_seg', 'tgt_seg', 'src_seg_t', 'src_t', 'src', 'tgt']
		if additional_keys is not None:
			save_keys += additional_keys
		for k in save_keys:
			tensor_np = get(self.tensors[k])
			tensor_np = np.squeeze(np.transpose(tensor_np, axes=(0, 2, 3, 1)))
			tensor_np[tensor_np > 1] = 1
			tensor_np[tensor_np < 0] = 0
			self.save_dict[k] = tensor_np

		self.save_dict['comb'] = Combine2Channels(self.save_dict['src_seg'], self.save_dict['tgt_seg'])
		self.save_dict['comb_t'] = Combine2Channels(self.save_dict['src_seg_t'], self.save_dict['tgt_seg'])
		self.save_dict['grid'] = DualIm2Grid(self.save_dict['src'], self.save_dict['tgt'], (self.opt.grid_size,self.opt.grid_size))
		self.save_dict['grid_t'] = DualIm2Grid(self.save_dict['src_t'], self.save_dict['tgt'], (self.opt.grid_size,self.opt.grid_size))

		self.save_flows = {}
		for k in ['flow', 'flow_r']:
			self.save_flows[k] = {}
			flow_np = get(self.tensors[k])
			self.save_flows[k]['u'] = flow_np[0, 0, :, :]
			self.save_flows[k]['v'] = flow_np[0, 1, :, :]

			max_flow = np.ceil(np.max( (self.save_flows[k]['u']**2 + self.save_flows[k]['v']**2)**0.5 ))
			flow_vis = Flow2Image(self.save_flows[k]['u'], self.save_flows[k]['v'], max_dis=max_flow)
			if k.endswith('_r'):
				cv2.putText(flow_vis, str(int(max_flow)), (10, flow_vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
				            fontScale=2, color=(0, 0, 0), thickness=2)
			self.save_dict[k] = flow_vis

	def SaveResults(self, phase=None):
		if phase is None:
			phase = self.phase
		save_ckpt_res = self.opt.mode.lower().startswith('tr') and (self.epoch + 1) % self.opt.interval_save_ckpt == 0
		if save_ckpt_res==True:
			im_dir = os.path.join(self.save_route, '%08d' % (self.epoch + 1))
			os.makedirs(im_dir, exist_ok=True)

		for k in self.save_dict.keys():
			fname = phase[:2]+'_%02d_' % (self.it) + k + '.png'
			cv2.imwrite(os.path.join(self.save_route, fname), np.uint8(self.save_dict[k] * 255))
			if save_ckpt_res==True:
				shutil.copyfile(os.path.join(self.save_route, fname), os.path.join(im_dir, fname))
		# for k in self.save_flows.keys():
		# 	fname = phase[:2] + '_%02d_'% (self.it) + k+'.mat'
		# 	sio.savemat(os.path.join(self.save_route, fname), self.save_flows[k])
		# 	if save_ckpt_res==True:
		# 		shutil.copyfile(os.path.join(self.save_route, fname), os.path.join(im_dir, fname))

	def _Segment(self, modal='src'):
		features_1 = self.vgg(normalize_batch(self.tensors[modal]))
		feat_f_1 = self.models['seg_feat_'+modal](features_1)
		seg_1 = self.models['seg_pred'](feat_f_1)

		if self.opt.rot_inv_type.startswith('rot'):
			im_inv = torch.flip(self.tensors[modal], dims=[2,3])
		else:
			im_inv = 1 - self.tensors[modal]
		features_2 = self.vgg(normalize_batch( im_inv ))
		feat_f_2 = self.models['seg_feat_'+modal](features_2)
		seg_2 = self.models['seg_pred'](feat_f_2)
		if self.opt.rot_inv_type.startswith('rot'):
			seg_2 = torch.flip(seg_2, dims=[2,3])
			feat_f_2 = torch.flip(feat_f_2, dims=[2,3])

		im_seg = ( seg_1 + seg_2 ) / 2 * self.tensors[modal+'_msk']
		feat_f_c = (feat_f_1+feat_f_2) / 2
		return im_seg, feat_f_c, seg_1, seg_2

	def Forward(self):
		'''  '''
		''' random flow for src and tgt '''
		if self.phase.lower().startswith('tr'):
			SIZE = self.SIZE
			randscale = self.opt.rand_offset

			flow_r_src = np.array([cv2.resize(np.random.normal(scale=randscale, size=(3,4)), dsize=(SIZE[1],SIZE[0])),
			                       cv2.resize(np.random.normal(scale=randscale, size=(3,4)), dsize=(SIZE[1],SIZE[0]))])
			flow_r_src = torch.cuda.FloatTensor(flow_r_src[np.newaxis], device=self.cuda_device[0])
			self.tensors['src'] = self.stn_destroy(flow_r_src, self.tensors['src'])
			self.tensors['src_msk'] = self.stn_destroy(flow_r_src, self.tensors['src_msk'])

			flow_r_tgt = np.array([cv2.resize(np.random.normal(scale=randscale, size=(3,4)), dsize=(SIZE[1],SIZE[0])),
			                       cv2.resize(np.random.normal(scale=randscale, size=(3,4)), dsize=(SIZE[1],SIZE[0]))])
			flow_r_tgt = torch.cuda.FloatTensor(flow_r_tgt[np.newaxis], device=self.cuda_device[0])
			self.tensors['tgt'] = self.stn_destroy(flow_r_tgt, self.tensors['tgt'])
			self.tensors['tgt_msk'] = self.stn_destroy(flow_r_tgt, self.tensors['tgt_msk'])

		''' seg '''
		self.tensors['src_seg'], feat_f_src_c, self.interm['seg_src_1'], self.interm['seg_src_2'] = self._Segment(modal='src')
		self.tensors['tgt_seg'], feat_f_tgt_c, self.interm['seg_tgt_1'], self.interm['seg_tgt_2'] = self._Segment(modal='tgt')

		''' flow '''
		feat_f = torch.cat(( feat_f_src_c, feat_f_tgt_c ), dim=1)
		self.tensors['flow'] = self.models['flow'](feat_f.to(self.cuda_device[1])).to(self.cuda_device[0])
		if self.opt.flow_scale_times>0:
			self.tensors['flow_r'] = nn.functional.interpolate(self.tensors['flow'], scale_factor=2**self.opt.flow_scale_times, mode='bilinear')
		else:
			self.tensors['flow_r'] = self.tensors['flow']

		''' transformation '''
		self.tensors['src_seg_t'] = self.stn(self.tensors['flow_r'], self.tensors['src_seg'])
		self.tensors['src_msk_t'] = self.stn(self.tensors['flow_r'], self.tensors['src_msk'])

		self.tensors['src_t'] = self.stn(self.tensors['flow_r'], self.tensors['src'])

	def ComputeLoss(self):
		''' losses '''
		''' self compare '''
		self.losses['cmp_self_src'] = self.loss_funcs['cmp_self'](self.interm['seg_src_1'], self.interm['seg_src_2'])
		self.losses['cmp_self_tgt'] = self.loss_funcs['cmp_self'](self.interm['seg_tgt_1'], self.interm['seg_tgt_2'])

		''' style loss '''
		features_seg_src = self.vgg(normalize_batch(self.tensors['src_seg'].repeat(1, 3, 1, 1)))
		features_seg_tgt = self.vgg(normalize_batch(self.tensors['tgt_seg'].repeat(1, 3, 1, 1)))
		self.losses['style_gram_src'] = self.loss_funcs['style_gram'](features_seg_src, self.g_ref)
		self.losses['style_gram_tgt'] = self.loss_funcs['style_gram'](features_seg_tgt, self.g_ref)

		''' flow '''
		self.losses['cmp_src_tgt'] = self.loss_funcs['cmp_src_tgt'](self.tensors['src_seg_t'], self.tensors['tgt_seg'],
		                                                            [self.tensors['src_msk_t'], self.tensors['tgt_msk']])

		self.losses['flow_smooth'] = self.loss_funcs['flow_smooth'](self.tensors['flow'])

		self.losses['total'] = (self.losses['style_gram_src'] + self.losses['style_gram_tgt']) * 1 + \
		                       self.losses['cmp_src_tgt'] * self.opt.photometric + \
		                       (self.losses['cmp_self_src'] + self.losses['cmp_self_tgt']) * 1e-3 + \
		                       self.losses['flow_smooth'] * self.opt.totalvar # losses['feat_norm'] * 1e-4

		if self.opt.ssim > 0:
			self.losses['ssim'] = 1 - self.loss_funcs['ssim'](self.tensors['src_t'] * self.tensors['src_msk_t'] * self.tensors['tgt_msk'],
			                                                  self.tensors['tgt'] *  self.tensors['src_msk_t'] * self.tensors['tgt_msk'])
			self.losses['total'] += self.losses['ssim']*self.opt.ssim

	def SetStatus(self, status='train'):
		self.phase = status
		for k in self.models.keys():
			if status.startswith('tr'):
				self.models[k].train()
			else:
				self.models[k].eval()

	def Train(self):
		t0 = time.time()

		for self.epoch in range(self.epoch_st, self.opt.total_epoch):
			train_loss_dict = {}
			self.SetStatus('train')
			for self.it in range(len(self.ds.src_train)):
				self.tensors, self.interm, self.losses = {}, {}, {}
				self.optimizer.zero_grad()
				im_train_list = PatchRandMirror([np.copy(self.ds.src_train[self.it]), np.copy(self.ds.tgt_train[self.it]),
				                                 np.copy(self.ds.src_train_msk[self.it]), np.copy(self.ds.tgt_train_msk[self.it])])
				self.tensors['src'] = torch.cuda.FloatTensor( ndarray2tensor(im_train_list[0].copy()), device=self.cuda_device[0] )
				self.tensors['tgt'] = torch.cuda.FloatTensor( ndarray2tensor(im_train_list[1].copy()), device=self.cuda_device[0] )
				self.tensors['src_msk'] = torch.cuda.FloatTensor( ndarray2tensor(im_train_list[2].copy()), device=self.cuda_device[0] )
				self.tensors['tgt_msk'] = torch.cuda.FloatTensor( ndarray2tensor(im_train_list[3].copy()), device=self.cuda_device[0] )

				self.Forward()
				self.ComputeLoss()
				self.losses['total'].backward()
				self.optimizer.step()
				if (self.opt.save_im and (self.epoch+1)%self.opt.interval_test==0) or self.opt.debug:
					self.Save()

				for k, v in self.losses.items():
					if k not in train_loss_dict:
						train_loss_dict[k] = []
					train_loss_dict[k].append(get(v))
			self.loss_all_dict[self.phase+'_ax'].append(self.epoch+1)
			for k, v in train_loss_dict.items():
				if self.phase+'loss_'+k not in self.loss_all_dict:
					self.loss_all_dict[self.phase+'loss_'+k] = []
				self.loss_all_dict[self.phase + 'loss_' + k].append(v)
			print(self.epoch, '%.02f'%(time.time()-t0), ' sec', end=' -- ram ')
			resident_size = self.process.memory_info().rss
			rss_g = resident_size //(1024**3)
			rss_m = (resident_size % (1024**3)) //(1024**2)
			rss_k = (resident_size % (1024**2)) //(1024**1)
			rss_b = resident_size % 1024
			print(rss_g, 'G', rss_m, 'M', rss_k, 'K', rss_b)

			self.test_loss_dict = {}
			if (self.epoch+1)%self.opt.interval_test == 0:
				self.TestFlow(phase='test')

				self.loss_all_dict[self.phase+'_ax'].append(self.epoch + 1)
				for k, v in self.test_loss_dict.items():
					if self.phase + 'loss_' + k not in self.loss_all_dict:
						self.loss_all_dict[self.phase + 'loss_' + k] = []
					self.loss_all_dict[self.phase + 'loss_' + k].append(v)
				sio.savemat(os.path.join(self.save_route, 'stat.mat'), mdict=self.loss_all_dict)
				shutil.copyfile(os.path.join(self.save_route, 'stat.mat'), os.path.join(self.save_route, 'stat_bk.mat'))

			if (self.epoch+1)%self.opt.interval_test == 0 or self.opt.debug:
				print('epoch %d' % (self.epoch + 1))
				for loss_dict, phase_l in zip([train_loss_dict, self.test_loss_dict], ['train loss', 'test loss']):
					for k, v in loss_dict.items():
						print(phase_l, k, np.mean(v))
					print('----------')
				print('%.02f'%(time.time()-t0), ' sec')
				print('\n')

			if (self.epoch+1)%self.opt.interval_save_ckpt == 0:
				checkpoint_name = os.path.join(self.save_route, 'ckpt_%d.pth.tar'%(self.epoch+1))
				checkpoint_dict = {}
				for k, v in self.loss_all_dict.items():
					checkpoint_dict[k] = v
				checkpoint_dict['epoch'] = self.epoch + 1
				checkpoint_dict['optimizer'] = self.optimizer.state_dict()
				for k, v in self.models.items():
					checkpoint_dict['model_'+k] = v.state_dict()
				checkpoint_dict['np_state'] = np.random.get_state()
				save_checkpoint(checkpoint_dict, checkpoint_name)

	def TestFlow(self, phase='test', rand_offset_folder=None):
		if rand_offset_folder is None:
			rand_offset_folder = self.opt.rand_offset_folder

		if phase.startswith('tr'):
			src = self.ds.src_train
			tgt = self.ds.tgt_train
			src_msk = self.ds.src_train_msk
			tgt_msk = self.ds.tgt_train_msk
		else:
			src = self.ds.src_test
			tgt = self.ds.tgt_test
			src_msk = self.ds.src_test_msk
			tgt_msk = self.ds.tgt_test_msk

		self.flows_test = []
		self.SetStatus('test')
		for self.it in range(len(src)):
			self.tensors, self.interm, self.losses = {}, {}, {}
			if self.opt.mode.lower().startswith('tr'):
				self.optimizer.zero_grad()
			if rand_offset_folder is not None:
				src_u, src_v, tgt_u, tgt_v = \
					[cv2.resize(sio.loadmat(os.path.join(rand_offset_folder, phase[:2]+'_rand_flow_%02d.mat'%(self.it)))[k].astype(np.float32),
					            dsize=(self.SIZE[1], self.SIZE[0]))
					 for k in ['src_u', 'src_v', 'tgt_u', 'tgt_v']]
				src_, src_msk_ = RemapCPU([src[self.it].copy(), src_msk[self.it].copy()], {'u': src_u, 'v': src_v}, interpolation=cv2.INTER_CUBIC)
				tgt_, tgt_msk_ = RemapCPU([tgt[self.it].copy(), tgt_msk[self.it].copy()], {'u': tgt_u, 'v': tgt_v}, interpolation=cv2.INTER_CUBIC)
			else:
				src_, src_msk_ = src[self.it].copy(), src_msk[self.it].copy()
				tgt_, tgt_msk_ = tgt[self.it].copy(), tgt_msk[self.it].copy()

			self.tensors['src'] = torch.cuda.FloatTensor(ndarray2tensor(src_), device=self.cuda_device[0])
			self.tensors['tgt'] = torch.cuda.FloatTensor(ndarray2tensor(tgt_), device=self.cuda_device[0])
			self.tensors['src_msk'] = torch.cuda.FloatTensor(ndarray2tensor(src_msk_), device=self.cuda_device[0])
			self.tensors['tgt_msk'] = torch.cuda.FloatTensor(ndarray2tensor(tgt_msk_), device=self.cuda_device[0])
			with torch.no_grad():
				self.Forward()

			if self.opt.save_im:
				self.Save(phase=phase)

			if self.opt.mode=='eval':
				self.flows_test.append(np.squeeze(get(self.tensors['flow_r'])))

			for k, v in self.losses.items():
				if k not in self.test_loss_dict:
					self.test_loss_dict[k] = []
				self.test_loss_dict[k].append(get(v))

		if self.opt.mode=='eval':
			sio.savemat(os.path.join(self.save_route, phase[:2] + '_flows.mat'), {'flows': self.flows_test})
			# if self.opt.mode.startswith('tr'):
			# 	shutil.copyfile(os.path.join(self.save_route, phase[:2] + '_flows.mat'),
			# 	                os.path.join(self.save_route, '%08d'%(self.epoch+1), phase[:2]+'_flows_ep%d.mat'%(self.epoch+1)))

	def Pred_Seg(self, im, msk, modal='src'):
		self.SetStatus('test')
		self.tensors = {}
		self.tensors[modal] = torch.cuda.FloatTensor(ndarray2tensor(im), device=self.cuda_device[0])
		self.tensors[modal+'_msk'] = torch.cuda.FloatTensor(ndarray2tensor(msk), device=self.cuda_device[0])
		with torch.no_grad():
			seg, _, _, _ = self._Segment(modal)
		seg_np = np.squeeze(get(seg))
		return seg_np

	def Pred_Flow(self, src, src_msk, tgt, tgt_msk):
		self.SetStatus('test')
		self.tensors = {}
		self.tensors['src'] = torch.cuda.FloatTensor(ndarray2tensor(src), device=self.cuda_device[0])
		self.tensors['tgt'] = torch.cuda.FloatTensor(ndarray2tensor(tgt), device=self.cuda_device[0])
		self.tensors['src_msk'] = torch.cuda.FloatTensor(ndarray2tensor(src_msk), device=self.cuda_device[0])
		self.tensors['tgt_msk'] = torch.cuda.FloatTensor(ndarray2tensor(tgt_msk), device=self.cuda_device[0])
		with torch.no_grad():
			self.Forward()
		src_seg = np.squeeze(get(self.tensors['src_seg']))
		tgt_seg = np.squeeze(get(self.tensors['tgt_seg']))
		flow_r = np.squeeze(get(self.tensors['flow_r']))
		return src_seg, tgt_seg, flow_r
