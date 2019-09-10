#!/usr/bin/env bash

device=$1 #the first arg is the device num.
echo Using CUDA device $device

arg="
    --netName DAIN
    --pretrained test_weight
    --test_uid test_15_to_60fps_results
    --val_fps 60
    --time_step 0.25
    --flowmethod 2
    --batch_size 1
    --use_cudnn 1
    --flowproj_threshhold 0.0784313725490196
    --lr 0.0
    --rectify_lr 0.001
    --save_which 1
    --alpha 0.0 1.0 --lambda1 0.0 --lambda2 0.0 --lambda3 0.0
    --weight_decay 0.0
    --patience 5
    --factor 0.2
"

CUDA_VISIBLE_DEVICES=$device python test_AIM.py $arg






