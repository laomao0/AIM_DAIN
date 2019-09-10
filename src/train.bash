#!/usr/bin/env bash
device=$1
uid=$2
echo Using CUDA device $device
echo Using UID  $uid


# todo continue train
CUDA_VISIBLE_DEVICES=$device python main_AIM_2019.py --uid  $uid  --pretrained test_weight \
    --netName DAIN \
    --batch_size 1 --save_which 1 \
    --ctx_lr_coe 1.0 --depth_lr_coe 0.001 \
    --flowproj_threshhold 0.0784313725490196 \
    --time_step 0.25 \
    --flowmethod 2 \
    --datasetName AIM_Challenge \
    --datasetPath  /DATA/wangshen_data/AIM_challenge \
    --task interp  --single_output 1 \
    --numEpoch 50 \
    --lr 0.00002000 \
    --rectify_lr 0.00002000 \
    --N_iter 1 \
    --flow_lr_coe 0.001  --occ_lr_coe 0.0 --filter_lr_coe 1.0 \
    --alpha 0.0 1.0 --lambda1 0.0000 --lambda2 0.0 --lambda3 0.0000 --lambda4 0.0 --weight_decay 0.0 \
    --patience 4 --factor 0.2 \
    --vis_env  $uid

echo ""
echo ""

