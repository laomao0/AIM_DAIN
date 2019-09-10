# DAIN (For AIM Challenge)

Wang Shen,
Wenbo Bao,
Guangtao Zhai,
Li Chen,
and Zhiyong Gao.


### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)
1. [Downloading Results](#downloading-results)
1. [Test Results Generation](#test-results-generation)
1. [Training New Models](#training-new-models) 

### Introduction
We propose the **D**epth-**A**ware video frame **IN**terpolation (**DAIN**) model to explicitly detect the occlusion by exploring the depth cue.
We develop a depth-aware flow projection layer to synthesize intermediate flows that preferably sample closer objects than farther ones.
Our method achieves state-of-the-art performance on the Middlebury dataset.

    
### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 18.04 LTS)
- Python (We test with Python = 3.6.8 in Anaconda3 = 4.5.11)
- Cuda & Cudnn (We test with Cuda = 10.0 and Cudnn = 7.5.0)
- PyTorch (The customized depth-aware flow projection and other layers require ATen API in PyTorch = 1.0.0)
- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 7.4.0 and nvcc = 10.0 compilers)
- NVIDIA GPU (We use RTX-2080 Ti with compute = 7.5, but we support compute_50/52/60/61/75 devices, should you have devices with higher compute capability, please revise [this](https://github.com/baowenbo/DAIN/blob/master/my_package/DepthFlowProjection/setup.py))

### Installation
Download repository:

    $ git clone https://github.com/laomao0/AIM_Challenge.git

Before building Pytorch extensions, be sure you have `pytorch >= 1.0.0`:
    
    $ python -c "import torch; print(torch.__version__)"
    
Generate our PyTorch extensions:
    
    $ cd AIM_Challenge
    $ cd my_package 
    $ ./build.sh

Generate the Correlation package required by [PWCNet](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch/external_packages/correlation-pytorch-master):
    
    $ cd ../PWCNet/correlation_package_pytorch1_0
    $ ./build.sh

### Testing Pre-trained Models
Make model weights dir and Middlebury dataset dir:

    $ cd AIM_Challenge
    $ mkdir model_weights
    $ mkdir test_weight
    
Download [pretrained models](), 

    $ cd test_weight

put best.pth into AIM_Challenge/model_weights/test_weight dir.
    

Download [AIM-Challenge dataset](wwww.vision.ee.ethz.ch/aim19/):
    
Make sure the dataset dirs follows the below structure.

         
    root -----train/train_15fps/000,...,239/00000000.png,...
         -----train/train_30fps
         -----train/train_60fps
         -----val/val_15fps
         -----val/val_30fps
         -----val/val_60fps
         -----test/test_15fps

### Downloading Results

Our test results 60fps (without 15fps inputs) can be downloaded by [link]().
    
    
### Test Results Generation

Modify the testset path of AIM_Challenge/src/test_AIM.py to be your path.

    AIM_Other_DATA = "/DATA/wangshen_data/AIM_challenge/test/test_15fps"
    AIM_Other_RESULT = "/DATA/wangshen_data/AIM_challenge/test/test_15fps_result"
    AIM_Other_RESULT_UPLOAD = "/DATA/wangshen_data/AIM_challenge/test/test_15fps_upload"

AIM_Other_RESULT is the path of full 60fps results without 15fps inputs.
AIM_Other_RESULT_UPLOAD path is the 30fps results upload to [CodaLab](AIM_Other_RESULT_UPLOAD).

The script for generating the 60fps testing results.

    $ cd AIM_Challenge
    $ cd src
    $ ./run_15_to_60fps_testset.bash 0
    
where 0 is the GPU index. For 4 GPUs, it can select from {0,1,2,3}.

### Training New Models
    
Run the training script:

    $ cd AIM_Challenge
    $ cd src
    $ ./train.bash 0 weight_name
    
where 0 is the GPU index, weight_name is the path, i.e. AIM_Challenge/model_weights/weight_name, to
save the new trained weights.



### Contact
[Wang Shen](mailto:wangshen834@gmail.com) [Wenbo Bao](mailto:bwb0813@gmail.com); 

### License
See [MIT License](https://github.com/baowenbo/DAIN/blob/master/LICENSE)
