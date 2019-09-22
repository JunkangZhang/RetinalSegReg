import numpy as np

def ndarray2tensor(im):
	if im.ndim==2:
		return np.expand_dims(np.expand_dims(im, axis=0), axis=0)
	else:
		return np.expand_dims(np.transpose(im, axes=(2, 0, 1)), axis=0)
