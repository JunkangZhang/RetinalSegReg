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


## Basic Setups
### Environments
python3 (Anaconda) <br>
pytorch (Version 0.4.1 was used for this project. Version 1.1.0 should also work) <br>
[pytorch_ssim](https://github.com/Po-Hsun-Su/pytorch-ssim) ([instructions](https://github.com/JunkangZhang/RetinalSegReg/blob/master/pytorch_ssim/readme.md)) <br>

### Datasets
#### Multi-Modal Retinal Dataset
Download 2 files from [Fundus Fluorescein Angiogram Photographs & Colour Fundus Images of Diabetic Patients](https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1/fundus-fluorescein-angiogram-photographs--colour-fundus-images-of-diabetic-patients). Unzip them into `./retina/FFAPCFIDP/`. <br>
**NOTE #1**: In `./retina/FFAPCFIDP/normal/`, switch the file names of `10-10.jpg` and `11-11.jpg` manually. <br>
**NOTE #2**: The pair of `30-30.jpg` & `30.jpg` in `./retina/FFAPCFIDP/normal/` is not used due to wrong image. <br>
**NOTE #3**: It seems the folders' names has been changed to upper cases in the latest version. Please change them to lower cases on Linux-based systems. <br>

#### Coarse alignment for images pairs
For each pair of images, we obtain the coarse alignment as an affine transformation matrix (2\*3) which is based on 3 pairs of manually labeled corresponding points. The matrices are stored in [`FFAPCFIDP_affine.csv`](https://github.com/JunkangZhang/RetinalSegReg/blob/master/FFAPCFIDP_affine.csv).  <br>

#### Retinal Vessel Segmentation Dataset (for training only)
We used [HRF](https://www5.cs.fau.de/research/data/fundus-images/). Download and unzip the Segmentation Dataset into  `./retina/HRF/`.  <br>

## Evaluation on a pretrained model
To be added. 

## Training
To be added. 
