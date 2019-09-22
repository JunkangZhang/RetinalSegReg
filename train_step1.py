import torch
import numpy as np
import os

from RetSegReg import options, RetSegReg
from globalsetting import globalsetting

torch.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)
os.environ['OMP_NUM_THREADS'] = '1'

gs = globalsetting()
opt = options()
opt.cuda_id = 0
opt.save_route = os.path.join(gs.ckpt_path, 'icip_train')

opt.flow_feat_scales = 5 # max downsampled times of features
opt.flow_scale_times = 2 # output scale for flow map, {0, 1, ..., flow_feat_scales}
opt.flow_base_filters = 128
opt.flow_max_filters = 128
opt.flow_downsample = 'pool'

opt.learing_rate = 1e-3
opt.rand_offset = 5
opt.totalvar = 2e-5 # 5e-5
opt.ssim = 1e-5

opt.smoothness_type = 'L1'

opt.debug = False # if true, save results on training set every epoch
opt.interval_test = 500
opt.interval_save_ckpt = 100
opt.total_epoch = 500
opt.save_im = True

opt.checkpoint_recovery = None
# opt.checkpoint_recovery = os.path.join(gs.proj_path, 'ckpt/'
#                           'icip_train_step1/'
#                           'ckpt_500.pth.tar')

opt.style_target = os.path.join(gs.data_path, 'HRF/manual1/12_h.tif')
opt.dataset_path = os.path.join(gs.data_path, 'FFAPCFIDP')
opt.csv_path = os.path.join(gs.proj_path, 'FFAPCFIDP_affine.csv')

model = RetSegReg(opt)
model.Train()
