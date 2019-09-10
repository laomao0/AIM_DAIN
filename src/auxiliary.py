
import sys
import os

import sys
import  threading
import torch
from torch.autograd import Variable
from lr_scheduler import *
from torch.autograd import gradcheck


import numpy

MODEL_PATH = "./../model_weights/weights_MS.pth"
auxiliary_path = '../iter-loss_MS.txt'


GANVideo_new_FineTexture = "/home/wenbobao/bwb/GANVideo_new_fineTexture/"
GANVideo_new = "/home/wenbobao/bwb/GANVideo_new/"
MB_new =  "/home/wenbobao/bwb/MiddleBurySet_Sample/"
MB_keepall_new =  "/home/wenbobao/bwb/MiddleBurySet_Sample_keepall/"
Vimeo_90K_interp = "/tmp4/wenbobao_data/vimeo_triplet" 
MB_raw  = "/tmp4/wenbobao_data/MiddleBurySet_rawSamples"

netName = 'MultiScaleStructure_occ_filt_flo'#dvf'#flow2filter'# occ_filt_flo'#_AdaptiveWeight'
print("Using " + netName + " network")
datasetName = 'Vimeo_90K_interp'#'GANVideo_FineTex' ##,'GANVideo_FineTex'] #'MiddleBury_raw'  #'MiddleBury_keepall' # # 'GANVideo' # 'MiddleBury_keepall'
datasetPath = Vimeo_90K_interp #GANVideo_new_FineTexture #,GANVideo_new_FineTexture]  #MB_raw # MB_keepall_new # GANVideo_new_FineTexture #GANVideo_new ##MB_keepall_new
split = 97 #  99
#val_max_samples = 1024 # too much is a waste of time
if type(datasetName) == list:
    for ii in datasetName:
        print("Using " + ii + " dataset")
else:
    print("Using " + datasetName + " dataset ")


numEpoch = 150

BATCH_SIZE = 1#16 # 8#4#4 # 16 # 8 #12 #4 #6#1 #32 # 64 #32
workers = int(round(1.6 * BATCH_SIZE) )#24 #
print(str(workers) + " workers for parallel loading of data")
# EPOCH = int( 51300/BATCH_SIZE) #3000
# VALIDATION_MAX_BATCHES = 32
# validation_batches =  int( 3782/BATCH_SIZE) #VALIDATION_MAX_BATCHES
# print("val batchsize is :" + str(BATCH_SIZE))
# print("val batch num is :" + str(validation_batches))
 
batch_size = BATCH_SIZE
imagesize = (3, 256,  448)  #(3,128,128) # (3,64,64) # (3,128,128) #
# kernellength = 51  #33#
filter_size = 4#2#4 #6
print("filter size is "+ str(filter_size) +"( ^2 = " +str(filter_size *filter_size) + ")") 
offset_scale = 20
#print("offset is scaled by " + str(offset_scale))

outputsize =  (3, 256, 448)  #(3,128,128) #  (3,128,128) # (3,32,32) # 
learning_rate = 0# 0.002 #* round(BATCH_SIZE/10.0,3) # 0#0.00002 # 0.001 #0.0005 # 0.0001
rectify_lr = 0.001 #0
save_which = 1 # save the rectified results.
flownets_lr_coe = 0.01# 0.02# 0.01#0.02# 0 #0.02 #0.01 # 0 # 0.001 # 0.01 # 0.001 # fine tune on the flownets
occlusion_lr_coe = 1.0 #0#1.0 # 0#0.5# 0.0#0.5# 0# 0.1
filter_lr_coe = 1.0#0#1.0#0 #1.0#0.0#1.0# 0.0 #
print("relative flow lr is : " + str(flownets_lr_coe))
print("relative filter lr is : " + str(filter_lr_coe))
print("relative occlusion lr is : " + str(occlusion_lr_coe))

use_negPSNR = False
if use_negPSNR:
    print("Using negPSNR instead of L1 loss ")
alpha  =  [0,1.0] #[1.0,0.0] #[0,1.0] #[2.0/3.0, 1.0/3.0] #[1.0/3.0, 2.0/3.0]  # [1.0,0.0]
lambda1 = [0]#[0.02] # [0]#[0.001] # [0]#[0.0004] #[0.001] # [0.02] #0.05 seems works good # 0.0001 not good, too small,
lambda2 = [0]#[0.005] #[0.001]#[0.001] #[0]#[0.001] # [0.005]
lambda3 = [0]#[0.001] #[0]#[0.001] #[0]#[0.001]
epsilon = 1e-6
weight_decay = 0# 1e-4 # weight decay is good for small dataset like Middle Bury to avoid overfitting
# reduce on plateou
patience = 5
factor = 0.2

FINE_TUNING = True
SAVED_MODEL = "../model_weights/weights_MS_uid_19239_epoch3.pth"#25422_epoch28.pth"#best.pth"#748899_epoch7.pth"#15581_epoch63.pth"#_2367_epoch0.pth"#72054_epoch6.pth"#66014_epoch63.pth"#67757_epoch30.pth"#5182_best.pth"#40410_best.pth"#6197_best.pth"#34270_epoch28.pth"#6197_best.pth"#34270_best.pth"
#28476_best.pth"#42430_best.pth"#68061_best.pth"#20821_best.pth"#25620_best.pth" #25620_best.pth"#83389_best.pth" #81091_epoch83.pth"#52185_best.pth"#68323_best.pth"# "../model_weights/weights_MS_iter15999.pth" # MODEL_PATH[:-4]+"_best"+MODEL_PATH[-4:]+".bk"

TEST_MODEL ="../model_weights/weights_MS_uid_12661_bestPSNR.pth"
#19239_epoch3.pth"#17317_bestPSNR.pth"#25422_best.pth"#748899_best.pth"#2054_bestMB.pth"#72054_epoch6.pth"#34270_epoch28.pth"#372054_bestMB.pth"#6197_bestMB.pth"#4270_bestMB.pth"
#28476_best.pth"#42430_best.pth"#25620_best.pth"#17592_epoch18.pth"
#model_weights/weights_MS_uid_88618_best.pth"#21108_best.pth" #95046_best.pth"#94897_best.pth" #78767_best.pth"#weights_MS_uid_56792_best.pth"#"../model_weights/weights_MS_uid_80721_best.pth" # "../model_weights/weights_MS_uid_28824_best.pth"#MODEL_PATH[:-4]+"_best"+MODEL_PATH[-4:]
PLOT = False

def select_processor():
    if torch.has_cudnn:
        use_cuda = True
        dtype = torch.cuda.FloatTensor
    else:
        use_cuda = False
        dtype = torch.FloatTensor
    #use_cuda = False
    #dtype = torch.FloatTensor
   
    return use_cuda,dtype

