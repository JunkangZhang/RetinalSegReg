import numpy as np
import cv2
import os
import scipy.io as sio
from utils_image import RemapCPU, ReadFFAPCFIDP_2
from globalsetting import globalsetting

np.random.seed(2018)

interpolation = cv2.INTER_CUBIC

gs = globalsetting()
save_route = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset')
os.makedirs(save_route, exist_ok=True)

SIZE = (576, 768)
randscale = 5

def rand_3x4():
	# return cv2.resize(np.random.normal(scale=randscale, size=(3, 4)), dsize=(SIZE[1], SIZE[0]))
	return np.random.normal(scale=randscale, size=(3, 4))

''' data '''
gs = globalsetting()
dataset_path = os.path.join(gs.data_path, 'Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients')
csv_path = os.path.join(gs.proj_path, 'FFAPCFIDP_affine.csv')
src_list, tgt_list, M_list, src_msk_list, tgt_msk_list, src_t_list, tgt_t_list, src_t_msk_list, tgt_t_msk_list = \
	ReadFFAPCFIDP_2(dataset_path=dataset_path, csv_path=csv_path, width=768, mask_shrink=0)

def SplitFFAPCFIDP(st=0):
	src_train = np.concatenate((src_list[st::2], src_t_list[st::2]), axis=0)
	tgt_train = np.concatenate((tgt_t_list[st::2], tgt_list[st::2]), axis=0)
	src_train_msk = np.concatenate((src_msk_list[st::2], src_t_msk_list[st::2]), axis=0)
	tgt_train_msk = np.concatenate((tgt_t_msk_list[st::2], tgt_msk_list[st::2]), axis=0)
	return src_train, tgt_train, src_train_msk, tgt_train_msk

data = {'tr':{}, 'te':{}}
data['tr']['src'], data['tr']['tgt'], data['tr']['src_msk'], data['tr']['tgt_msk'] = SplitFFAPCFIDP(st=0)
data['te']['src'], data['te']['tgt'], data['te']['src_msk'], data['te']['tgt_msk'] = SplitFFAPCFIDP(st=1)

for phase in ['tr', 'te']:
	num = data[phase]['src'].shape[0]

	for i in range(num):
		rand_flow = {}
		rand_flow['src_u'] = rand_3x4()
		rand_flow['src_v'] = rand_3x4()

		rand_flow['tgt_u'] = rand_3x4()
		rand_flow['tgt_v'] = rand_3x4()

		sio.savemat(os.path.join(save_route, phase + '_rand_flow_%02d.mat' % (i)), rand_flow)

		for k1 in ['src', 'tgt']:
			for k2 in ['_u', '_v']:
				rand_flow[k1+k2] = cv2.resize(rand_flow[k1+k2], dsize=(SIZE[1], SIZE[0]))

		src_ = RemapCPU([data[phase]['src'][i]], {'u': rand_flow['src_u'], 'v': rand_flow['src_v']}, interpolation=interpolation)[0]
		tgt_ = RemapCPU([data[phase]['tgt'][i]], {'u': rand_flow['tgt_u'], 'v': rand_flow['tgt_v']}, interpolation=interpolation)[0]

		sio.savemat(os.path.join(save_route, phase + '_%02d_src.mat' % (i)), {'im':src_})
		sio.savemat(os.path.join(save_route, phase + '_%02d_tgt.mat' % (i)), {'im':tgt_})

		src_[src_ > 1] = 1
		src_[src_ < 0] = 0
		cv2.imwrite(os.path.join(save_route, phase + '_%02d_src.png'%(i)), np.uint8(src_*255))
		tgt_[tgt_ > 1] = 1
		tgt_[tgt_ < 0] = 0
		cv2.imwrite(os.path.join(save_route, phase + '_%02d_tgt.png'%(i)), np.uint8(tgt_*255))
