# Joint Vessel Segmentation and Deformable Registration <br> on Multi-Modal Retinal Images based on Style Transfer

**Joint Vessel Segmentation and Deformable Registration on Multi-Modal Retinal Images based on Style Transfer** <br>
Junkang Zhang, 
Cheolhong An, 
[Ji Dai](https://jidai-code.github.io/), 
Manuel Amador, 
Dirk-Uwe Bartsch, 
Shyamanga Borooah, 
William R. Freeman, 
and [Truong Q. Nguyen](https://jacobsschool.ucsd.edu/faculty/faculty_bios/index.sfe?fmp_recid=48). <br>
IEEE International Conference on Image Processing (ICIP), 2019. <br>

\[[Paper \(IEEEXplore\)](https://ieeexplore.ieee.org/document/8802932)\] &ensp; 
\[[Supplementary pdf](https://github.com/JunkangZhang/RetinalSegReg/blob/master/ICIP2019_supplementary.pdf)\] &ensp; 
\[[Slides (to be uploaded)]()\] <br>


## Contents
- [Setups](#setups)
- [Evaluation](#evaluation)
- [Training](#training)
- [Possible issues & solutions](#solutions)
- [Citation](#citation)
- [Reference](#reference)

## 1. Setups <a name="setups"></a>
### 1.1 Environments
**Basic requirements**: <br>
python3 (Anaconda) <br>
pytorch (Version 0.4.1 was used for this project. Version 1.1.0 should also work) <br>
[pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) ([instructions](https://github.com/JunkangZhang/RetinalSegReg/blob/master/pytorch_ssim/readme.md)) <br>

**Requirements (optional)**: <br>
scikit-image (for Dice_s) <br>
MATLAB (for Dice, Phase+MIND) <br>

**Hardwares**: <br>
Our platform: Intel Core i7-7700K, nVidia GTX 1080Ti, Windows 10 64-bit, Python 3.6 (Anaconda), Matlab 2017b <br>

To use the model reported in the paper, ~8GB (training) or ~4GB (testing) GPU memory is needed. Another model used in `train.py` will take more memory. 

### 1.2 Datasets
#### (1) Multi-Modal Retinal Dataset
Download 2 files from [Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients](https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1/fundus-fluorescein-angiogram-photographs--colour-fundus-images-of-diabetic-patients) [1]. Unzip them into `./retina/FFAPCFIDP/`. <br>
**NOTE #1**: In `./retina/FFAPCFIDP/normal/`, switch the file names of `10-10.jpg` and `11-11.jpg` manually. <br>
**NOTE #2**: The pair of `30-30.jpg` & `30.jpg` in `./retina/FFAPCFIDP/normal/` is not used due to wrong image. <br>
**NOTE #3**: It seems the folders' names has been changed to upper cases in the latest version. Please change them to lower cases on Linux-based systems. <br>

For this dataset, we use pairs with odd numbers for training and even numbers for testing. 

#### (2) Coarse alignment for images pairs
For each pair of images, we obtain the coarse alignment as an affine transformation matrix (2\*3) which is based on 3 pairs of manually labeled corresponding points. The matrices are stored in [`FFAPCFIDP_affine.csv`](https://github.com/JunkangZhang/RetinalSegReg/blob/master/FFAPCFIDP_affine.csv).  <br>

#### (3) Retinal Vessel Segmentation Dataset (for training only)
We used [HRF](https://www5.cs.fau.de/research/data/fundus-images/) [2]. Download and unzip the Segmentation Dataset into  `./retina/HRF/`.  <br>
Only one segmentation map `./retina/HRF/manual1/12_h.tif` is needed as a style target. Other files or binary/probability maps from other datasets also work. <br>

## 2. Evaluation <a name="evaluation"></a>
### 2.1 Getting results on a trained model
(1) Run `randflow.py` to generate random flow maps to simulate larger misalignment between the input images pairs. The generated folder `./ckpt/FFAPCFIDP_random_offset/` will take ~1.3GB on the disk.  <br>
(2) Download the pretrained model \([Google Drive](https://drive.google.com/file/d/1iNS-2war7jGdS-i5twadZZ14LXUWR0Rw/view?usp=sharing)\). Place it into `./ckpt/icip_reported/`.  <br>
If Google Drive cannot be accessed, please try [Baidu Wangpan](https://pan.baidu.com/s/1vA6alBhSppZFhdRu00UpGA) (access code: ryg6). <br>
(3) Run `eval.py` to predict registration fields on the image pairs. The generated folder `./ckpt/Prediction_icip_reported/` will take ~400MB with `opt.save_im=False`.  <br>

To use another checkpoint, modify line #38 & 41 before running `eval.py`. <br>

### 2.2 Computing Soft Dice
(1) Run `dice_s.py`.

### 2.3 Computing Dice
(1) Download matlab codes of [B-cosfire](https://www.mathworks.com/matlabcentral/fileexchange/49172-trainable-cosfire-filters-for-curvilinear-structure-delineation-in-images) [6] and extract them into `./matlab/`. <br>
(2) Run `matlab/bcosf_get.m` to obtain segmentation responses. It took ~9 minutes on our platform. The generated folder `./ckpt/FFAPCFIDP_random_offset_bcosfire/` will take ~600MB on the disk. <br>
(Optional) If system memory is not enough, switch `parfor` to `for` on line #46 (and it will take more time). <br>
(3) Run `dice.py`. 

### 2.4 Getting results for Phase+MIND (optional)
(1) Download matlab codes of [monogenic_signal_matlab](https://github.com/CPBridge/monogenic_signal_matlab) [3,4] and place them into `./matlab/`. <br>
(2) Download matlab codes of [MIND](http://www.ibme.ox.ac.uk/research/biomedia/julia-schnabel/files/gn-mind2d.zip/view) [5] and extract them into `./matlab/`. <br>
Then compile `pointsor.cpp` (enter the codes' folder and run `mex pointsor.cpp`). <br>
(Optional) In `./matlab/GN-MIND2d/deformableReg2Dmind.m`, comment lines # 18, 60 & 61 to disable plotting. <br>
(3) Run `./matlab/getmind2.m`. It took ~23 minutes on our platform. The results will be stored in `./ckpt/FFAPCFIDP_random_offset_phase-mind/`. <br>
(Optional) If system memory runs out, switch `parfor` to `for` on line #22 (and it will be much slower). <br>
(4) Do **2.2** & **2.3** to get measurements. Modify the codes as `method = 'mind'` on line #11 of `dice.py` and line #115 of `dice_s.py`. 


## 3. Training <a name="training"></a>
Run `train_2steps.bat` to get the model in the paper. Or run `train_step1.py` and `train_step2.py` sequentially by hand. <br>

One can try squared L2 smoothness loss by running `train.py`. <br>


## Possible issues & solutions <a name="solutions"></a>
We observed a drastic memory increase taken by the python process during training on a Ubuntu workstation with pytorch 0.4.0/0.4.1. An update to pytorch 1.1.0 solved the problem. 


## Citation <a name="citation"></a>
```
@inproceedings{Zhang:2019:ICIP:Retinal,
  author={Junkang Zhang and Cheolhong An and Ji Dai and Manuel Amador and Dirk-Uwe Bartsch and Shyamanga Borooah and William R. Freeman and Truong Q. Nguyen},
  booktitle={2019 IEEE International Conference on Image Processing (ICIP)},
  title={Joint Vessel Segmentation and Deformable Registration on Multi-Modal Retinal Images Based on Style Transfer},
  year={2019},
  pages={839-843}
}
```


## Reference <a name="reference"></a>
[1] Shirin Hajeb Mohammad Alipour, Hossein Rabbani, and Mohammad Reza Akhlaghi, “Diabetic retinopathy grading by digital curvelet transform,” Computational and mathematical methods in medicine, vol. 2012, pp. 1607–1614, 2012. <br>
[2] Attila Budai, R¨udiger Bock, Andreas Maier, Joachim Hornegger, and Georg Michelson, “Robust vessel segmentation in fundus images,” International journal of biomedical imaging, vol. 2013, 2013. <br>
[3] Michael Felsberg and Gerald Sommer, “The monogenic signal,” IEEE Transactions on Signal Processing, vol. 49, no. 12, pp. 3136–3144, 2001. <br>
[4] Christopher P. Bridge, “Introduction To The Monogenic Signal,” CoRR, vol. abs/1703.09199, 2017. <br>
[5] Mattias P. Heinrich, Mark Jenkinson, Manav Bhushan, Tahreema Matin, Fergus V. Gleeson, Sir Michael Brady, and Julia A. Schnabel, “Mind: Modality independent neighbourhood descriptor for multi-modal deformable registration,” Medical Image Analysis, vol. 16, no. 7, pp. 1423 – 1435, 2012. <br>
[6] George Azzopardi, Nicola Strisciuglio, Mario Vento, and Nicolai Petkov, “Trainable cosfire filters for vessel delineation with application to retinal images,” Medical Image Analysis, vol. 19, no. 1, pp. 46 – 57, 2015. <br>
