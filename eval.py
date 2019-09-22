import torch
import torch.nn as nn
import numpy as np
import os, time

from RetSegReg import options, RetSegReg
from globalsetting import globalsetting

torch.manual_seed(2019)
torch.cuda.manual_seed_all(2019)
np.random.seed(2019)
os.environ['OMP_NUM_THREADS'] = '1'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

gs = globalsetting()
opt = options()
opt.cuda_id = 0
opt.save_route = ''

opt.flow_feat_scales = 5
opt.flow_scale_times = 2
opt.flow_base_filters = 128
opt.flow_max_filters = 128
opt.flow_downsample = 'pool'

opt.rand_offset_folder = os.path.join(gs.ckpt_path, 'FFAPCFIDP_random_offset')  # not for training

opt.mode = 'eval'
opt.save_im = False

opt.dataset_path = os.path.join(gs.data_path, 'Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients')
opt.csv_path = os.path.join(gs.proj_path, 'FFAPCFIDP_affine.csv')

# name = 'icip_train'
name = 'icip_reported'

model = RetSegReg(opt)
for ep in [3000]:
    model.save_route = os.path.join(gs.ckpt_path, 'Prediction_' + name + '/%d'%ep)
    os.makedirs(model.save_route, exist_ok=True)
    checkpoint_recovery = os.path.join(gs.ckpt_path, name + '/ckpt_%d.pth.tar' % ep)
    model.LoadCheckpoint(checkpoint_recovery)
    model.TestFlow('train')
    model.TestFlow('test')
    print(ep)
