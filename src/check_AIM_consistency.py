"""
    In AIM2019 challenge, the upload 30fps and 60fps results must be from the same model.
    Original 60fs : frame index                   : 0   2    4    6   8   10   12   14   16   18
    For example:                         val 30:    0        4        8        12        16

    we check the 4 8 12 ... frames of val30fps and val 60fps.

"""
import numpy
import os
from scipy.misc import imread, imsave, imshow, imresize, imsave
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

val_30_uid = "18257"
val_60_uid = "6628"

base_path = "/DATA/wangshen_data/AIM_challenge/val/val_15fps_result"
val_30_path = os.path.join(base_path, val_30_uid)
val_60_path = os.path.join(base_path, val_60_uid)


val_30_dirs = os.listdir(val_30_path)
val_60_dirs = os.listdir(val_60_path)

# find the common set
common_dirs = [folder for folder in val_30_dirs if folder in val_60_dirs]

for dir in common_dirs:

    for frame in range(4,361,8):

        first_frame_num = int(frame)
        first_frame_name = str(first_frame_num).zfill(8) + '.png'
        arguments_strFirst = os.path.join(val_30_path, dir, first_frame_name)
        arguments_strFirst_img = imread(arguments_strFirst)

        second_frame_num = int(frame)
        second_frame_name = str(second_frame_num).zfill(8) + '.png'
        arguments_strSecond = os.path.join(val_60_path, dir, second_frame_name)
        arguments_strSecond_img = imread(arguments_strSecond)

        psnr = compare_psnr(arguments_strFirst_img, arguments_strSecond_img)
        ssim = compare_ssim(arguments_strFirst_img, arguments_strSecond_img, multichannel=True)

        print(psnr, ssim)

        assert psnr >= 60
        assert ssim >= 0.99

print("Two folder is consistency")





