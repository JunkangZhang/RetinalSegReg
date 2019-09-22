# Multi-Modal Retinal Image Registration via Style Transfer

**Joint Vessel Segmentation and Deformable Registration on Multi-Modal Retinal Images based on Style Transfer** <br>
\[[Paper \(IEEEXplore\)](https://ieeexplore.ieee.org/document/8802932)\] <br>
Junkang Zhang, 
Cheolhong An, 
[Ji Dai](https://jidai-code.github.io/), 
Manuel Amador, 
Dirk-Uwe Bartsch, 
Shyamanga Borooah, 
William R. Freeman, 
and [Truong Q. Nguyen](https://jacobsschool.ucsd.edu/faculty/faculty_bios/index.sfe?fmp_recid=48). <br>
Accepted by IEEE International Conference on Image Processing (ICIP), 2019. 


## 1. Basic Setups
### 1.1 Environments
python3 (Anaconda) <br>
pytorch (Version 0.4.1 was used for this project. Version 1.1.0 should also work) <br>
[pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) ([instructions](https://github.com/JunkangZhang/RetinalSegReg/blob/master/pytorch_ssim/readme.md)) <br>

### 1.2 Datasets
#### (1) Multi-Modal Retinal Dataset
Download 2 files from [Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients](https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1/fundus-fluorescein-angiogram-photographs--colour-fundus-images-of-diabetic-patients). Unzip them into `./retina/FFAPCFIDP/`. <br>
**NOTE #1**: In `./retina/FFAPCFIDP/normal/`, switch the file names of `10-10.jpg` and `11-11.jpg` manually. <br>
**NOTE #2**: The pair of `30-30.jpg` & `30.jpg` in `./retina/FFAPCFIDP/normal/` is not used due to wrong image. <br>
**NOTE #3**: It seems the folders' names has been changed to upper cases in the latest version. Please change them to lower cases on Linux-based systems. <br>

In this dataset, pairs with odd numbers are used for training, and ones with even numbers are for evaluation. 

#### (2) Coarse alignment for images pairs
For each pair of images, we obtain the coarse alignment as an affine transformation matrix (2\*3) which is based on 3 pairs of manually labeled corresponding points. The matrices are stored in [`FFAPCFIDP_affine.csv`](https://github.com/JunkangZhang/RetinalSegReg/blob/master/FFAPCFIDP_affine.csv).  <br>

#### (3) Retinal Vessel Segmentation Dataset (for training only)
We used [HRF](https://www5.cs.fau.de/research/data/fundus-images/). Download and unzip the Segmentation Dataset into  `./retina/HRF/`.  <br>
Only one segmentation map is used as a style target. Binary/probability maps from other datasets also work. <br>

## 2. Evaluation on a pretrained model
### 2.1 Getting results on test set
(1) Run `andflow.py` to generate random flow masp to simulate larger misalignment between the input images pairs. The generated folder `./ckpt/FFAPCFIDP_random_offset` will take ~1.3GB on the disk.  <br>
(2) Download the pretrained model ([Google Drive](https://drive.google.com/file/d/1iNS-2war7jGdS-i5twadZZ14LXUWR0Rw/view?usp=sharing)). Place it into `./ckpt/icip_reported/`.  <br>
(3) Run `eval.py` to obtain registration fields on the image pairs. The generated folder `./ckpt/Prediction_icip_reported/` will take ~400MB with `opt.save_im=False`.  <br>

### 2.2 Computing Soft Dice
Requirement: scikit-image <br>
Run `dice_s.py`.

### 2.3 Computing Dice
Requirements: Matlab

Obtain matlab codes for [B-cosfire](https://www.mathworks.com/matlabcentral/fileexchange/49172-trainable-cosfire-filters-for-curvilinear-structure-delineation-in-images)

Run bcosf_get.m. If your platform has insufficient memory, switch parfor to for on line #46 (and it will be much slower). With parfor on, it took ~9 minutes on a platform with Intel Core i7-7700K and Matlab 2017b. The folder ./ckpt/FFAPCFIDP_random_offset_bcosfire will take ~600MB on the disk. 


## 3. Training
To be added. 

## Reference
To be added
