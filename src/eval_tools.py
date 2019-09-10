import sys
import os
import numpy as np

import imageio
from myssim import compare_ssim as ssim

#SCALE = 8 
SCALE = 1

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def _open_img(img_p):
    F = np.array(imageio.imread(img_p)).astype(float)/255.0
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def _open_img_ssim(img_p):
    F = np.array(imageio.imread(img_p))
    h, w, c = F.shape
    F = F[:h-h%SCALE, :w-w%SCALE, :]
    boundarypixels = SCALE 
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels,:]
    return F

def compute_psnr(ref_im, res_im):
    return output_psnr_mse(_open_img(ref_im), _open_img(res_im))

def compute_mssim(ref_im, res_im):
    ref_img = _open_img_ssim(ref_im)
    res_img = _open_img_ssim(res_im)
    channels = []
    for i in range(3):
        channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
        gaussian_weights=True, use_sample_covariance=False))
    return np.mean(channels)