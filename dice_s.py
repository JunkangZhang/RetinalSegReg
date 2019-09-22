import numpy as np
import cv2
import time, os
from skimage.filters import frangi
import scipy.io as sio
from utils_image import RemapCPU, DualIm2Grid, ColorCLAHE, FFAPCFIDP_icip, MskShrink
from globalsetting import globalsetting

def IoU(im1, im2, th=0.5):
	if im1.dtype is not np.bool:
		im1 = im1.astype(np.float32) >= th
	if im2.dtype is not np.bool:
		im2 = im2.astype(np.float32) >= th
	union = np.logical_or(im1, im2).astype(np.float32)
	intersection = np.logical_and(im1, im2).astype(np.float32)
	iou = np.sum(intersection) / np.sum(union)
	return iou

def SoftDice(im1, im2):
	dsc = 2*np.sum(np.minimum(im1, im2)) / (np.sum(im1) + np.sum(im2))
	return dsc

def DiceCoefficient(im1, im2, th=0.5):
	'''
	Equivently F1 --> 2*TP / (2*TP + FP + FN)
	:param im1: a single channel image
	:param im2: a single channel image, same size as im1
	:param th:
	:return:
	'''
	if im1.dtype is not np.bool:
		im1 = im1.astype(np.float32) >= th
	if im2.dtype is not np.bool:
		im2 = im2.astype(np.float32) >= th
	intersection = np.logical_and(im1, im2).astype(np.float32)
	dsc = 2*np.sum(intersection) / (np.sum(im1.astype(np.float32)) + np.sum(im2.astype(np.float32))+1e-12)
	return dsc

def PrecisionRecall(pred, gt, th=0.5):
	TP, FP, TN, FN = DiagnoseBinary(pred, gt, th)
	Precision = TP / (TP+FP)
	Recall = TP / (TP+FN)
	F1 = 2*Precision*Recall / (Precision+Recall)
	return Precision, Recall, F1

def DiagnoseBinary(pred, gt, th=0.5):
	pred = pred.astype(np.float32) >= th
	gt = gt.astype(np.float32) >= th
	TP = np.sum( np.logical_and(pred, gt).astype(np.float32) ) / np.sum(gt)
	FP = np.sum( np.logical_and(pred, 1-gt).astype(np.float32) ) / np.sum(1-gt)
	FN = 1 - TP
	TN = 1 - FP
	return TP, FP, TN, FN

def FrangiWarp4Eval(im, msk_frangi=None, msk_output=None, shrink=0, neg=False,
                   scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15, black_ridges=True):
	'''
	:param im:
	:param msk_frangi: set outliers to black (0) in case of negative image (neg==True) for the frangi algorithm
	:param msk_output: remove unwanted frangi results on the boundaries
	:param shrink: if msk_output is not None, shrink is not used
	:param neg:
	:param scale_range:
	:param scale_step:
	:param beta1:
	:param beta2:
	:param black_ridges:
	:return:
	'''
	if msk_frangi is None:
		msk_frangi = np.ones(im.shape[:2], dtype=im.dtype)

	if msk_output is None:
		msk_output = np.ones(im.shape[:2], dtype=im.dtype)
		if shrink > 0:
			se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink * 2 + 1, shrink * 2 + 1))
			msk_output = cv2.erode(msk_output, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)

	if im.ndim == 3:
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im = ColorCLAHE(cv2.GaussianBlur(im, ksize=(5,5), sigmaX=0.5))
	if neg:
		im = 1 - im
	im[msk_frangi<1] = 0
	res = frangi(im, scale_range, scale_step, beta1, beta2, black_ridges)
	res[msk_output<1] = 0

	res = res / np.max(res)

	return res

def SoftDiceFrangi(src, tgt, src_msk_full=None, tgt_msk_full=None, src_msk_shrink=None, tgt_msk_shrink=None):
	if src_msk_shrink is None:
		src_msk_shrink = np.ones(src.shape[:2], dtype=src.dtype)
	if tgt_msk_shrink is None:
		tgt_msk_shrink = np.ones(tgt.shape[:2], dtype=tgt.dtype)
	src_f = FrangiWarp4Eval(src, msk_frangi=src_msk_full, msk_output=src_msk_shrink)
	src_f[tgt_msk_shrink<1] = 0
	tgt_f = FrangiWarp4Eval(tgt, msk_frangi=tgt_msk_full, msk_output=tgt_msk_shrink)
	tgt_f[src_msk_shrink<1] = 0
	# cv2.imshow('a', src)
	# cv2.imshow('b', tgt)
	# cv2.imshow('c', src_f)
	# cv2.imshow('d', tgt_f)
	# cv2.waitKey()
	dice = SoftDice(src_f, tgt_f)
	return dice

if __name__ == '__main__':
	gs = globalsetting()
	SIZE = (576, 768)
	dataset_path = os.path.join(gs.data_path, 'Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients')
	csv_path = os.path.join(gs.proj_path, 'FFAPCFIDP_affine.csv')

	method = 'cnn' # 'mind' #
	interpolation = cv2.INTER_CUBIC  # cv2.INTER_LINEAR #
	rand_offset_folder = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset')

	# save_seg = False
	save_grid = False
	grid_size = int(np.ceil(768. / 5))
	compute_dice = True
	save_dice = compute_dice and False

	suffix = ''
	if method.lower() == 'mind':
		ckpt_path = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset_phase-mind')
		epoches = [-1]
	else:
		name = 'icip_reported'
		epoches = [3000]
		ckpt_path = os.path.join(gs.ckpt_path, 'Prediction_' + name + '/%d')

	ds = FFAPCFIDP_icip(dataset_path=dataset_path, csv_path=csv_path)

	for phase in ['te', 'tr']:
		if phase.startswith('tr'):
			suff = 'train'
		else:
			suff = 'test'

		if phase.startswith('tr'):
			src_list = ds.src_train
			tgt_list = ds.tgt_train
			src_msk_list = ds.src_train_msk
			tgt_msk_list = ds.tgt_train_msk
		else:
			src_list = ds.src_test
			tgt_list = ds.tgt_test
			src_msk_list = ds.src_test_msk
			tgt_msk_list = ds.tgt_test_msk

		src_msk_shrink_list = MskShrink(src_msk_list, shrink=10)
		tgt_msk_shrink_list = MskShrink(tgt_msk_list, shrink=10)

		t0 = time.time()
		for epoch in epoches:  # range(500,5001,500): # [3000]: # [4500]: #

			if method.lower() == 'mind':
				ckpt_path_c = ckpt_path
				flows = sio.loadmat(os.path.join(ckpt_path_c, '%s_flow.mat' % phase))
				flow_u = flows['ust'].astype(np.float32)
				flow_v = flows['vst'].astype(np.float32)
			else:
				ckpt_path_c = ckpt_path % epoch
				flows = sio.loadmat(os.path.join(ckpt_path_c, '%s_flows%s.mat'%(phase, suffix)))['flows'].astype(np.float32)
				flow_u = flows[:, 0, :, :]
				flow_v = flows[:, 1, :, :]

			dice_o = []
			dice_t = []
			for i in range(src_list.shape[0]):
				''' warp the input images '''
				src_u, src_v, tgt_u, tgt_v = \
					[cv2.resize(sio.loadmat(os.path.join(rand_offset_folder, '%s_rand_flow_%02d.mat'%(phase, i)))[k].astype(np.float32),
					            dsize=(SIZE[1], SIZE[0]))
					 for k in ['src_u', 'src_v', 'tgt_u', 'tgt_v']]

				src_o = src_list[i].copy()
				src_msk = src_msk_list[i].copy() > 0.5
				src_msk_s = src_msk_shrink_list[i].copy() > 0.5

				src_o, src_msk, src_msk_s = RemapCPU([src_o, src_msk, src_msk_s], {'u': src_u, 'v': src_v}, interpolation=interpolation)

				tgt_o = tgt_list[i].copy()
				tgt_msk = tgt_msk_list[i].copy() > 0.5
				tgt_msk_s = tgt_msk_shrink_list[i].copy() > 0.5

				tgt_o = 1 - tgt_o
				tgt_o[tgt_msk < 1] = 0

				tgt_o, tgt_msk, tgt_msk_s = RemapCPU([tgt_o, tgt_msk, tgt_msk_s], {'u': tgt_u, 'v': tgt_v}, interpolation=interpolation)

				''' dice before warpping '''
				# if compute_dice:
				# 	dice_o.append(SoftDiceFrangi(src_o, tgt_o, src_msk_full=src_msk, tgt_msk_full=tgt_msk, src_msk_shrink=src_msk_s, tgt_msk_shrink=tgt_msk_s))

				if save_grid:
					comb = DualIm2Grid(src_o, tgt_o, (grid_size, grid_size))
					comb[comb > 1] = 1
					comb[comb < 0] = 0
					os.makedirs(os.path.join(ckpt_path_c + '_grid'), exist_ok=True)
					cv2.imwrite(os.path.join(ckpt_path_c + '_grid', '%s_%02d_grid.png' % (phase, i)),
					            np.uint8(comb * 255))

				''' dice after warpping '''
				src_o_t, src_msk_t, src_msk_s_t = RemapCPU([src_o, src_msk, src_msk_s], {'u': flow_u[i], 'v': flow_v[i]},
				                                           interpolation=interpolation)

				if compute_dice:
					dice_t.append(SoftDiceFrangi(src_o_t, tgt_o, src_msk_full=src_msk_t, tgt_msk_full=tgt_msk,
					                             src_msk_shrink=src_msk_s_t, tgt_msk_shrink=tgt_msk_s))

				if save_grid:
					comb_t = DualIm2Grid(src_o_t, tgt_o, (grid_size, grid_size))
					comb_t[comb_t > 1] = 1
					comb_t[comb_t < 0] = 0
					os.makedirs(os.path.join(ckpt_path_c + '_grid'), exist_ok=True)
					cv2.imwrite(os.path.join(ckpt_path_c + '_grid', '%s_%02d_grid_t.png' % (phase, i)), np.uint8(comb_t*255))
					comb_t = DualIm2Grid(tgt_o, src_o_t, (grid_size, grid_size))
					comb_t[comb_t > 1] = 1
					comb_t[comb_t < 0] = 0
					cv2.imwrite(os.path.join(ckpt_path_c + '_grid', '%s_%02d_grid_t2.png' % (phase, i)), np.uint8(comb_t*255))

				print(i, end=' ')
				if (i + 1) % 10 == 0:
					print(' ')

			dice_o_avg = np.mean(dice_o)
			dice_t_avg = np.mean(dice_t)
			print('\nphase:', phase, '/ epoch:', epoch)
			print('before warping ->', dice_o)
			print('average ->', dice_o_avg)
			print('after warping ->', dice_t)
			print('average ->', dice_t_avg)

			if save_dice:
				sio.savemat(os.path.join(ckpt_path_c, '%s_flows%s_dice.mat' % (phase, suffix)), {'dice': dice_t})
			print('\n')
