import numpy as np
import cv2, time, os
import scipy.io as sio
from utils_image import Combine2Channels, RemapCPU
from dice_s import DiceCoefficient
from globalsetting import globalsetting

if __name__ == '__main__':

	gs = globalsetting()
	method = 'cnn' # 'mind' #
	interpolation = cv2.INTER_CUBIC  # cv2.INTER_LINEAR #
	bcosfire_folder = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset_bcosfire')

	save_seg = False
	save_comb = False
	save_dice = False
	save_smooth = False

	suffix = ''
	if method.lower()=='mind':
		ckpt_path = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset_phase-mind')
		epoches = [-1]
	else:
		name = 'icip_reported'
		epoches = [3000]
		ckpt_path = os.path.join(gs.ckpt_path, 'Prediction_' + name + '/%d')

	# [49, 34] or [29.17]
	# src_ths = list(range(20,81)) # 68
	# tgt_ths = list(range(20,81)) # 43
	src_th = 53 # 36
	tgt_th = 68 # 48
	sfx = '-s%02d-t%02d' % (src_th, tgt_th)

	t0 = time.time()
	for phase in ['te', 'tr']:
		if phase.startswith('tr'):
			suff = 'train'
			tot = 60
		else:
			suff = 'test'
			tot = 58

		for epoch in epoches:

			if method.lower()=='mind':
				ckpt_folder = ckpt_path
				flows = sio.loadmat( os.path.join(ckpt_folder, '%s_flow.mat'%phase) )
				flow_u = flows['ust'].astype(np.float32)
				flow_v = flows['vst'].astype(np.float32)
			else:
				ckpt_folder = ckpt_path % epoch
				flows = sio.loadmat( os.path.join(ckpt_folder, '%s_flows%s.mat'%(phase, suffix)) )['flows'].astype(np.float32)
				flow_u = flows[:, 0, :, :]
				flow_v = flows[:, 1, :, :]

			if save_seg or save_comb:
				save_im_folder = ckpt_folder + '_bcosf' + sfx
				os.makedirs(save_im_folder, exist_ok=True)

			dice_o = []
			dice_t = []
			for i in range(tot):
				''' bcosfire '''
				src_seg = sio.loadmat( os.path.join(bcosfire_folder, '%s_%02d_src_bcosfire.mat'%(phase, i)) )['resp'].astype(np.float32)
				src_msk = sio.loadmat( os.path.join(bcosfire_folder, '%s_%02d_src_bcosfire.mat'%(phase, i)) )['mask'].astype(np.float32)
				src_msk = src_msk > 0.5

				tgt_seg = sio.loadmat( os.path.join(bcosfire_folder, '%s_%02d_tgt_bcosfire.mat'%(phase, i)) )['resp'].astype(np.float32)
				tgt_msk = sio.loadmat( os.path.join(bcosfire_folder, '%s_%02d_tgt_bcosfire.mat'%(phase, i)) )['mask'].astype(np.float32)
				tgt_msk = tgt_msk > 0.5

				''' dice before warpping '''
				src_ = src_seg >= src_th
				src_[src_msk == False] = 0
				src_[tgt_msk == False] = 0

				tgt_ = tgt_seg >= tgt_th
				tgt_[tgt_msk == False] = 0
				tgt_[src_msk == False] = 0

				dice_o.append( DiceCoefficient(src_, tgt_) )

				if save_seg:
					cv2.imwrite(os.path.join(save_im_folder, '%s_%02d_src_bcosf.png'%(phase, i)), np.uint8(src_*255.))
					cv2.imwrite(os.path.join(save_im_folder, '%s_%02d_tgt_bcosf.png'%(phase, i)), np.uint8(tgt_*255.))
				if save_comb:
					cv2.imwrite(os.path.join(save_im_folder, '%s_%02d_comb_bcosf.png'%(phase, i)), np.uint8(Combine2Channels(src_, tgt_)*255.))

				''' dice after warpping '''
				''' remap response map '''
				# src_seg, src_msk = RemapCPU([src_seg, src_msk], {'u': flow_u[i], 'v': flow_v[i]}, interpolation=interpolation)
				# src_ = src_seg >= src_th
				''' remap binary map '''
				src_seg, src_msk = RemapCPU([src_seg >= src_th, src_msk], {'u': flow_u[i], 'v': flow_v[i]}, interpolation=interpolation)
				src_ = src_seg
				src_[src_msk == False] = 0
				src_[tgt_msk == False] = 0

				tgt_ = tgt_seg >= tgt_th
				tgt_[tgt_msk == False] = 0
				tgt_[src_msk == False] = 0

				dice_t.append( DiceCoefficient(src_, tgt_) )

				if save_comb:
					cv2.imwrite(os.path.join(save_im_folder, '%s_%02d_comb_t_bcosf.png'%(phase, i)), np.uint8(Combine2Channels(src_, tgt_)*255.))

			dice_o_avg = np.mean(dice_o)
			dice_t_avg = np.mean(dice_t)
			print('\nphase:', phase, '/ epoch:', epoch)
			print('before warping ->', dice_o)
			print('average ->', dice_o_avg)
			print('after warping ->', dice_t)
			print('average ->', dice_t_avg)
			if save_dice==True:
				sio.savemat( os.path.join(ckpt_folder, '%s_flows%s_dice_bcosf'%(phase, suffix)+sfx+'.mat'), {'dice':dice_t, 'dice_o':dice_o})

			if save_smooth:
				smooth_L1 = np.mean(np.abs(flow_u[:,1:,:] - flow_u[:,:-1,:]), axis=(1,2)) + np.mean(np.abs(flow_v[:,:,1:] - flow_v[:,:,:-1]), axis=(1,2))
				sio.savemat( os.path.join(ckpt_folder, '%s_flows%s_smoothL1.mat'%(phase, suffix)), {'smooth':smooth_L1})
				smooth_L2sq = np.mean((flow_u[:,1:,:] - flow_u[:,:-1,:])**2, axis=(1,2)) + np.mean((flow_v[:,:,1:] - flow_v[:,:,:-1])**2, axis=(1,2))
				sio.savemat( os.path.join(ckpt_folder, '%s_flows%s_smoothL2sq.mat'%(phase, suffix)), {'smooth':smooth_L2sq})


