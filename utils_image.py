import numpy as np
import cv2
import os
# import h5py

''' image read '''

def CropSquare(im):
	transform = False
	ss = im.shape
	if ss[0] < ss[1]:
		im = np.swapaxes(im, 0, 1)
		ss = im.shape
		transform = True
	diff = ss[0]-ss[1]
	im = im[diff//2:diff//2+ss[1], ...]
	if transform:
		im = np.swapaxes(im, 0, 1)
	return im

def ExpandSquare(im, fill=0):
	transform = False
	ss = im.shape
	if ss[0] > ss[1]:
		im = np.swapaxes(im, 0, 1)
		ss = im.shape
		transform = True
	diff = ss[1]-ss[0]
	if len(ss)==2:
		imn = np.zeros((ss[1],ss[1]), dtype=im.dtype) #+ fill
	else:
		imn = np.zeros((ss[1], ss[1], 3), dtype=im.dtype)  # + fill
	imn[diff//2:diff//2+ss[0], ...] = im
	if transform:
		imn = np.swapaxes(imn, 0, 1)
	return imn

def FFAPCFIDP_Expand(im, width=720, height=576):
	ss = im.shape
	if height <= ss[0] and width <= ss[1]:
		return im
	if len(ss)==3:
		nim = np.zeros((height, width, ss[2]), dtype=im.dtype)
		nim[(height-ss[0])//2:(height-ss[0])//2+ss[0], (width-ss[1])//2:(width-ss[1])//2+ss[1], :] = im
	else:
		nim = np.zeros((height, width), dtype=im.dtype)
		nim[(height-ss[0])//2:(height-ss[0])//2+ss[0], (width-ss[1])//2:(width-ss[1])//2+ss[1]] = im
	return nim

def ReadFFAPCFIDP_2(dataset_path, csv_path, width=720, height=576, mask_shrink=0):
	mask = np.zeros((576, 720), dtype=np.float32)
	cv2.circle(mask, (720 // 2, 576 // 2), radius=720 // 2 - 1, color=1, thickness=-1)
	# mask = cv2.resize(ExpandSquare(mask), (768, 768)) == 1
	# mask = mask.astype(np.float32)
	# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
	# mask = cv2.erode(mask, se)
	Mt_f = np.array([[1,0,(width-720)/2], [0,1,0], [0,0,1]])
	mask = FFAPCFIDP_Expand(mask, width=width, height=height)

	src_list, tgt_list, M_list, src_msk_list, tgt_msk_list = [], [], [], [], []
	src_t_list, tgt_t_list, src_t_msk_list, tgt_t_msk_list = [], [], [], []
	fp = open(csv_path)
	for line in fp.readlines():
		sp = line.split(',')
		src = cv2.imread(os.path.join(dataset_path, sp[0]))
		src = FFAPCFIDP_Expand(src, width=width, height=height)
		src_list.append(src)
		tgt = cv2.imread(os.path.join(dataset_path, sp[1]))
		tgt = FFAPCFIDP_Expand(tgt, width=width, height=height)
		tgt_list.append(tgt)
		M = np.array([float(sp[_+2]) for _ in range(6)]).reshape((2,3))
		M_f = np.concatenate((M, [[0,0,1]]), axis=0)
		M_new = Mt_f.dot(M_f.dot(np.linalg.inv(Mt_f)))
		M = M_new[:2,:]
		M_list.append(M)
		src_msk_list.append(mask)
		tgt_msk_list.append(mask)

		src_t = cv2.warpAffine(src.copy(), M, (tgt.shape[1], tgt.shape[0]))
		src_t_list.append(src_t)
		src_t_msk_list.append(cv2.warpAffine(mask.copy(), M, (tgt.shape[1], tgt.shape[0])))
		M_t = np.eye(3)
		M_t[0:2] = M
		M_t = np.linalg.inv(M_t)
		tgt_t = cv2.warpAffine(tgt.copy(), M_t[0:2], (src.shape[1], src.shape[0]))
		tgt_t_list.append(tgt_t)
		tgt_t_msk_list.append(cv2.warpAffine(mask.copy(), M_t[0:2], (src.shape[1], src.shape[0])))

	# 	cv2.imshow('a', src)
	# 	cv2.imshow('b', tgt)
	# 	cv2.imshow('c', mask)
	# 	cv2.imshow('d', src_t)
	# 	cv2.imshow('e', tgt_t)
	# 	cv2.imshow('f', src_t_msk_list[-1])
	# 	cv2.imshow('g', tgt_t_msk_list[-1])
	# 	cv2.waitKey()
	# cv2.destroyAllWindows()
	fp.close()

	if mask_shrink > 0:
		mask_shrink *= 2+1
		se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_shrink, mask_shrink))
		for i in range(len(src_msk_list)):
			src_msk_list[i] = cv2.erode(src_msk_list[i], se, borderValue=0)
			tgt_msk_list[i] = cv2.erode(tgt_msk_list[i], se, borderValue=0)
			src_t_msk_list[i] = cv2.erode(src_t_msk_list[i], se, borderValue=0)
			tgt_t_msk_list[i] = cv2.erode(tgt_t_msk_list[i], se, borderValue=0)


	src_list = np.array(src_list).astype(np.float32) / 255.
	tgt_list = np.array(tgt_list).astype(np.float32) / 255.
	M_list = np.array(M_list)
	src_msk_list = np.array(src_msk_list)
	tgt_msk_list = np.array(tgt_msk_list)
	src_t_list = np.array(src_t_list).astype(np.float32) / 255.
	tgt_t_list = np.array(tgt_t_list).astype(np.float32) / 255.
	src_t_msk_list = np.array(src_t_msk_list)
	tgt_t_msk_list = np.array(tgt_t_msk_list)

	return src_list, tgt_list, M_list, src_msk_list, tgt_msk_list, \
	       src_t_list, tgt_t_list, src_t_msk_list, tgt_t_msk_list

class FFAPCFIDP_icip():
	def __init__(self, dataset_path, csv_path):
		src_list, tgt_list, M_list, src_msk_list, tgt_msk_list, src_t_list, tgt_t_list, src_t_msk_list, tgt_t_msk_list = \
			ReadFFAPCFIDP_2(dataset_path=dataset_path, csv_path=csv_path, width=768, mask_shrink=0)

		def SplitFFAPCFIDP(st=0):
			src_train = np.concatenate((src_list[st::2], src_t_list[st::2]), axis=0)
			tgt_train = np.concatenate((tgt_t_list[st::2], tgt_list[st::2]), axis=0)
			src_train_msk = np.concatenate((src_msk_list[st::2], src_t_msk_list[st::2]), axis=0)
			tgt_train_msk = np.concatenate((tgt_t_msk_list[st::2], tgt_msk_list[st::2]), axis=0)
			return src_train, tgt_train, src_train_msk, tgt_train_msk

		self.src_train, self.tgt_train, self.src_train_msk, self.tgt_train_msk = SplitFFAPCFIDP(st=0)
		self.src_test, self.tgt_test, self.src_test_msk, self.tgt_test_msk = SplitFFAPCFIDP(st=1)

''' augmentation '''

def PatchRandMirror(im_list):
	dic = type(im_list) is dict
	bracket = type(im_list) is list
	if dic==False:
		if bracket==False:
			im_list = [im_list]
		im_list_ = {}
		length = len(im_list)
		for i in range(length):
			im_list_[i] = im_list[i]
		im_list = im_list_

	swap1 = np.random.uniform() < 0.5
	swap2 = np.random.uniform() < 0.5
	for k in im_list.keys():
		im = im_list[k]
		if swap1==True:
			im = im[::-1]
		if swap2 == True:
			im = im[:,::-1]
		im_list[k] = im.copy()

	if dic==False:
		if bracket==False:
			im_list = im_list[0]
		else:
			im_list_ = []
			for k in range(length):
				im_list_.append(im_list[k])
			im_list = im_list_
	return im_list

''' visualization '''
def Flow2Image(u, v, max_dis=20, dtype=np.float32):
	im = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
	mag = (u**2 + v**2)**0.5
	
	ang = np.arccos(u/mag) / np.pi * 180
	ang[v<0] *= -1
	ang = (ang+360) % 360
	im[:,:,0] = np.floor(ang/2).astype(np.uint8)
	
	mag_t = mag
	mag_t[mag_t>max_dis] = max_dis
	im[:,:,1] = np.round((mag_t / max_dis)*255).astype(np.uint8)

	im[:,:,2] = 255

	rgb = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
	if dtype == np.float32:
		rgb = rgb.astype(np.float32) / 255.
	return rgb

def DualIm2Grid(im1, im2, grid=(50,50)):
	'''
	:param im1:
	:param im2:
	:param grid: (H,W)
	:return:
	'''
	shape = im1.shape[0:2]
	if im1.ndim==2:
		im1 = np.repeat(im1[:,:,np.newaxis], 3, axis=2)
	if im2.ndim==2:
		im2 = np.repeat(im2[:,:,np.newaxis], 3, axis=2)
	nim = np.zeros(im1.shape, dtype=im1.dtype)
	for r in range( np.ceil(shape[0]/grid[0]).astype(np.int32) ):
		for c in range(np.ceil(shape[1]/grid[1]).astype(np.int32)):
			switch = (r+c)%2
			re = min((r+1)*grid[0], shape[0])
			ce = min((c+1)*grid[1], shape[1])
			if switch==0:
				nim[r*grid[0]:re, c*grid[1]:ce, :] = im1[r*grid[0]:re, c*grid[1]:ce, :]
			else:
				nim[r*grid[0]:re, c*grid[1]:ce, :] = im2[r*grid[0]:re, c*grid[1]:ce, :]
	return nim

def Combine2Channels(im1, im2):
	comb = np.zeros((im1.shape[0], im1.shape[1], 3))
	comb[:, :, 2] = im1
	comb[:, :, 1] = im2
	return comb

def RemapCPU(im_list, flow, interpolation=cv2.INTER_CUBIC):
	'''
	:param im_list:
	:param flow:
	:return:
	'''
	mesh_u_o, mesh_v_o = np.meshgrid(np.arange(flow['u'].shape[1]), np.arange(flow['u'].shape[0]))
	mesh_u = mesh_u_o.astype(np.float32) + flow['u']
	mesh_v = mesh_v_o.astype(np.float32) + flow['v']
	im_t_list = []
	for i in range(len(im_list)):
		im = im_list[i].copy()
		if im_list[i].dtype==np.bool:
			im = im.astype(np.float32)
		im_t = cv2.remap(im, mesh_u.astype(np.float32), mesh_v.astype(np.float32),
		                 interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
		if im_list[i].dtype==np.bool:
			im_t = im_t>0.5
		im_t_list.append(im_t)

	return im_t_list

''' processing '''
def ColorCLAHE(im):
	'''
	:param im: [h,w,3], range [0,1]
	:return:
	'''
	if im.ndim==2:
		im_ = im
	else:
		im_ = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE()
	imn_ = clahe.apply(np.uint8(im_*255))
	imn_ = imn_.astype(np.float32) / 255
	if im.ndim==2:
		im = imn_
	else:
		ratio = imn_ / (im_+1e-10)
		im = im*ratio[:,:,np.newaxis]
		im[im>1] = 1
		im[im<0] = 0
	return im

def MskShrink(arr, shrink=0, shape=cv2.MORPH_ELLIPSE):
	arrn = np.zeros(arr.shape, dtype=arr.dtype)
	se = cv2.getStructuringElement(shape, (shrink * 2 + 1, shrink * 2 + 1))
	for i in range(arr.shape[0]):
		arrn[i] = cv2.erode(arr[i], se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
	return arrn

if __name__ == '__main__':
	pass
