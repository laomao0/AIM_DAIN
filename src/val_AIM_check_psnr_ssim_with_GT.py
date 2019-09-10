import time
import os
import threading
import torch
from torch.autograd import Variable
from lr_scheduler import *
from torch.autograd import gradcheck
import sys
import getopt
import math
import numpy
import torch
# import torch.utils.serialization
# import PIL
# import PIL.Image

import AverageMeter

import random
import logging
import numpy as np
import os
from scipy.misc import imsave
import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#plt.style.use('bmh')
# from auxiliary import *
import numpy
# from SeparableConv import *
# from data_generator_GanVideo_FineTexture import *
#from networks.MultiScaleStructure_occ_filt_flo import *
import networks
from my_args import  args

from loss_function import *
from scipy.misc import imread, imsave, imshow, imresize, imsave
from AverageMeter import  *
import shutil
torch.backends.cudnn.benchmark = True  # to speed up the
# import cv2
from eval_tools import *
from PYTHON_Flow2Color.flowToColor import flowToColor
from PYTHON_Flow2Color.writeFlowFile import writeFlowFile
DO_AIMTest = False

val_fps = int(args.val_fps)  # must be 60
assert  val_fps == 60
numFrames = int(1.0 / args.time_step) - 1  # must be 3
assert numFrames == 3
time_offsets = [kk * args.time_step for kk in range(1, 1 + numFrames, 1)]

DO_AIM = True

# --------------------------------------------------------------------- #
#                        Need To Modify File Path                       #
# --------------------------------------------------------------------- #
AIM_Other_DATA = "/DATA/wangshen_data/AIM_challenge/val/val_15fps"
AIM_Other_RESULT = "/DATA/wangshen_data/AIM_challenge/val/val_"+ str(val_fps)+ "fps_results"
AIM_Other_GT = "/DATA/wangshen_data/AIM_challenge/val/val_"+ str(val_fps)+"fps"


if not os.path.exists(AIM_Other_RESULT):
    os.mkdir(AIM_Other_RESULT)


print("We check the our interpolated results using the training dataset to check psnr and ssim!")

# use_cuda, dtype = select_processor()

model = networks.__dict__[args.netName](batch=args.batch_size, channel=args.channels, width=None, height=None,
                                        scale_num=1, scale_ratio=2, temporal=False, filter_size = args.filter_size,
                                        save_which=args.save_which, flowmethod=args.flowmethod,
                                        timestep=args.time_step,
                                        FlowProjection_threshhold=args.flowproj_threshhold,
                                        offset_scale=None,cuda_available=args.use_cuda, cuda_id=None, training=False)

if args.use_cuda:
    model = model.cuda()
args.SAVED_MODEL = '../model_weights/' + args.SAVED_MODEL + "/best" + ".pth"
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We donn't load any trained weights, the model only has the pretrained Flow/Depth weights")
    print("*****************************************************************")

model = model.eval() # deploy mode


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([numpy.prod(p.size()) for p in parameters])

    return N


print("Num. of model parameters is :" + str(count_network_parameters(model)))
if hasattr(model, 'flownets'):
    print("Num. of flow model parameters is :" +
          str(count_network_parameters(model.flownets)))
if hasattr(model, 'initScaleNets_occlusion'):
    print("Num. of initScaleNets_occlusion model parameters is :" +
          str(count_network_parameters(model.initScaleNets_occlusion) +
              count_network_parameters(model.initScaleNets_occlusion1) +
              count_network_parameters(model.initScaleNets_occlusion2)))
if hasattr(model,'initScaleNets_occlusion'):
    print("Num. of initScaleNets_occlusion model parameters is :" +
          str(count_network_parameters(model.initScaleNets_occlusion) +
              count_network_parameters(model.initScaleNets_occlusion1) +
            count_network_parameters(model.initScaleNets_occlusion2)))
if hasattr(model, 'initScaleNets_filter'):
    print("Num. of initScaleNets_filter model parameters is :" +
          str(count_network_parameters(model.initScaleNets_filter) +
              count_network_parameters(model.initScaleNets_filter1) +
              count_network_parameters(model.initScaleNets_filter2)))
if hasattr(model, 'ctxNet'):
    print("Num. of ctxNet model parameters is :" +
              str(count_network_parameters(model.ctxNet)))
if hasattr(model, 'depthNet'):
    print("Num. of depthNet model parameters is :" +
          str(count_network_parameters(model.depthNet)))
if hasattr(model, 'rectifyNet'):
    print("Num. of rectifyNet model parameters is :" +
          str(count_network_parameters(model.rectifyNet)))

import  time
def val_AIM(model=model, use_cuda=args.use_cuda, save_which=args.save_which, dtype=args.dtype):
    if args.test_uid is not None:
        unique_id = args.test_uid
    else:
        unique_id =str(random.randint(0, 100000))
    print("The unique id for current testing is: " + str(unique_id))
    total_run_time = AverageMeter()
    interp_error = AverageMeter()
    if DO_AIM:
        subdir = sorted(os.listdir(AIM_Other_DATA))  # folder 0 1 2 3...
        gen_dir = os.path.join(AIM_Other_RESULT, unique_id)
        os.mkdir(gen_dir)

        print('----------------------------------')
        print('Results Dir: ', gen_dir)

        psnr_total = AverageMeter()
        ssim_total = AverageMeter()
        tot_timer = AverageMeter()
        proc_timer = AverageMeter()
        end = time.time()
        for dir in subdir:
            # prepare the image save path
            # if not dir == 'Beanbags':
            #     continue
            pstring = dir
            print(pstring)
            print(pstring, file=open(os.path.join(gen_dir, "log.txt"), "a"))
            os.mkdir(os.path.join(gen_dir, dir))
            frames_path = os.path.join(AIM_Other_DATA,dir)
            frames = sorted(os.listdir(frames_path))

            len_frames = len(frames)


            for index in range(len_frames - 1):

                if val_fps == 30:

                    first_frame_num = int(index * 4)
                    first_frame_name = str(first_frame_num).zfill(8) + '.png'

                    arguments_strFirst = os.path.join(AIM_Other_DATA, dir, first_frame_name)

                    arguments_strFirst_symlink = os.path.join(gen_dir, dir, first_frame_name)


                    second_frame_num = str(int(first_frame_num + 8))
                    second_frame_name = second_frame_num.zfill(8) + '.png'

                    arguments_strSecond = os.path.join(AIM_Other_DATA, dir, second_frame_name)

                    arguments_strSecond_symlink = os.path.join(gen_dir, dir, second_frame_name)


                    #insert softlink
                    os.symlink(arguments_strFirst, arguments_strFirst_symlink)
                    os.symlink(arguments_strSecond, arguments_strSecond_symlink)


                    middle_frame_num = int(first_frame_num + 4)

                    middle_frame_num_list = [middle_frame_num]


                    if args.time_step == 0.25: # the random middle training model, but output intermediate frames
                        time_index_list = [1]
                    elif args.time_step == 0.5: # the middle trainiing model
                        time_index_list = [0]


                elif val_fps == 60:

                    # insert first and last reference frames

                    first_frame_num = int(index * 8)
                    first_frame_name = str(first_frame_num).zfill(8) + '.png'

                    arguments_strFirst = os.path.join(AIM_Other_DATA, dir, first_frame_name)

                    arguments_strFirst_symlink = os.path.join(gen_dir, dir, first_frame_name)

                    second_frame_num = str(int(first_frame_num + 8))
                    second_frame_name = second_frame_num.zfill(8) + '.png'

                    arguments_strSecond = os.path.join(AIM_Other_DATA, dir, second_frame_name)

                    arguments_strSecond_symlink = os.path.join(gen_dir, dir, second_frame_name)

                    # insert softlink
                    if index == 0:
                        os.symlink(arguments_strFirst, arguments_strFirst_symlink)
                    os.symlink(arguments_strSecond, arguments_strSecond_symlink)

                    middle_frame_num_list = []
                    time_index_list = []
                    for index, timestep in enumerate(time_offsets):  # 0-0.25 1-0.5 2-0.75
                        middle_frame_num_list.append(int(first_frame_num + 2*(index+1)))
                        time_index_list.append(index)

                for mid_index, middle_frame_num in zip(time_index_list, middle_frame_num_list):

                    #upload_frame_name = dir + '_' + str(middle_frame_num).zfill(8) + '.png'

                    middle_frame_name = str(middle_frame_num).zfill(8) + '.png'

                    arguments_strOut = os.path.join(gen_dir, dir, middle_frame_name)

                    # arguments_strUpload = os.path.join(AIM_Other_RESULT_UPLOAD, upload_frame_name)


                    # gt_path = os.path.join(AIM_Other_GT, dir, "frame10i11.png")

                    gt_path = os.path.join(AIM_Other_GT, dir, middle_frame_name)

                    X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)
                    X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)


                    y_ = torch.FloatTensor()

                    assert (X0.size(1) == X1.size(1))
                    assert (X0.size(2) == X1.size(2))

                    intWidth = X0.size(2)
                    intHeight = X0.size(1)
                    channel = X0.size(0)
                    if not channel == 3:
                        continue

                    assert ( intWidth <= 1280)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications
                    assert ( intHeight <= 720)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications

                    if intWidth != ((intWidth >> 7) << 7):
                        intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                        intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                        intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
                    else:
                        intWidth_pad = intWidth
                        intPaddingLeft = 32
                        intPaddingRight= 32

                    if intHeight != ((intHeight >> 7) << 7):
                        intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                        intPaddingTop = int((intHeight_pad - intHeight) / 2)
                        intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
                    else:
                        intHeight_pad = intHeight
                        intPaddingTop = 32
                        intPaddingBottom = 32

                    pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                    torch.set_grad_enabled(False)
                    X0 = Variable(torch.unsqueeze(X0,0))
                    X1 = Variable(torch.unsqueeze(X1,0))
                    X0 = pader(X0)
                    X1 = pader(X1)

                    if use_cuda:
                        X0 = X0.cuda()
                        X1 = X1.cuda()
                    proc_end = time.time()

                    if args.netName == 'MultiScaleStructure_filt_flo_ctxS2D_depth_Modeling3_slomo' or \
                            args.netName == 'MultiScaleStructure_filt_flo_ctxS2D_depth_Modeling3_slomo_reviseCtx':
                        # if args.time_step == 0.5:
                        #     y_s, offset, filter, occlusion = model(torch.stack((X0, X1), dim=0), torch.tensor([mid_index]))
                        # elif args.time_step == 0.25: # 0 1 2, set output the middle frame
                        #     y_s, offset, filter, occlusion = model(torch.stack((X0, X1), dim=0), torch.tensor([mid_index]))
                        y_s, offset, filter, occlusion = model(torch.stack((X0, X1), dim=0), torch.tensor([mid_index]))
                    else:
                        # if args.time_step == 0.5:
                        #     y_s,offset,filter,occlusion = model(torch.stack((X0, X1),dim = 0))
                        # elif args.time_step == 0.25:
                        #     y_s, offset, filter, occlusion = model(torch.stack((X0, X1), dim=0), torch.tensor([1]))
                        y_s, offset, filter, occlusion = model(torch.stack((X0, X1), dim=0), torch.tensor([mid_index]))

                    #y_s,offset,filter,occlusion = model(torch.stack((X0, X1),dim = 0))
                    y_ = y_s[save_which]

                    # if index >=3:
                    proc_timer.update(time.time() -proc_end)
                    tot_timer.update(time.time() - end)
                    end  = time.time()
                    print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
                    total_run_time.update(time.time()-proc_end,1)
                    if use_cuda:
                        X0 = X0.data.cpu().numpy()
                        y_ = y_.data.cpu().numpy()
                        offset = [offset_i.data.cpu().numpy() for offset_i in offset]
                        filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
                        occlusion = [occlusion_i.data.cpu().numpy() for occlusion_i in occlusion]  if occlusion[0] is not None else None
                        X1 = X1.data.cpu().numpy()
                    else:
                        X0 = X0.data.numpy()
                        y_ = y_.data.numpy()
                        offset = [offset_i.data.numpy() for offset_i in offset]
                        filter = [filter_i.data.numpy() for filter_i in filter]
                        occlusion = [occlusion_i.data.numpy() for occlusion_i in occlusion]
                        X1 = X1.data.numpy()



                    X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
                    y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
                    offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
                    filter = [np.transpose(
                        filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                        (1, 2, 0)) for filter_i in filter]  if filter is not None else None
                    occlusion = [np.transpose(
                        occlusion_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
                        (1, 2, 0)) for occlusion_i in occlusion]  if occlusion is not None else None
                    X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))


                    imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))


                    # copy upload

                    # if (middle_frame_num - 4 ) % 32 == 0:
                    #     shutil.copy(arguments_strOut,arguments_strUpload)

                    # if occlusion is not None:
                    #     imsave(arguments_strOut[:-4] + '_occlusion1.png', occlusion[0][:,:,0])
                    #     imsave(arguments_strOut[:-4] + '_occlusion2.png', occlusion[1][:,:,0])
                    # flow1 = flowToColor(offset[0])
                    # imsave(arguments_strOut[:-4] + '_offset1.png', flow1)
                    # flow2 = flowToColor(offset[1])
                    # imsave(arguments_strOut[:-4] + '_offset2.png', flow2)
                    # writeFlowFile(offset[0],arguments_strOut[:-4] + '_offset1.flo')
                    # writeFlowFile(offset[1],arguments_strOut[:-4] + '_offset2.flo')
                    #
                    # if filter is not None:
                    #     for ii in range(filter[0].shape[2]):
                    #         imsave(arguments_strOut[:-4] + '_filter0_' + str(ii) + '.png', filter[0][:,:,ii])
                    #         imsave(arguments_strOut[:-4] + '_filter1_' + str(ii) + '.png', filter[1][:, :, ii])

                    rec_rgb = imread(arguments_strOut)
                    gt_rgb = imread(gt_path)

                    diff_rgb = 128.0 + rec_rgb - gt_rgb
                    avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))

                    interp_error.update(avg_interp_error_abs, 1)

                    mse = numpy.mean((diff_rgb - 128.0) ** 2)
                    if mse == 0:
                        return 100.0
                    PIXEL_MAX = 255.0
                    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

                    psnr = compute_psnr(gt_path, arguments_strOut)
                    ssim = compute_mssim(gt_path, arguments_strOut)

                    psnr_total.update(psnr, 1)
                    ssim_total.update(ssim, 1)

                    pstring = "interpolation error / PSNR/ SSIM : " + \
                              str(round(avg_interp_error_abs,4)) + \
                              " / " + str(round(psnr,4)) +\
                              " / " + str(round(ssim,4))
                    print(pstring)
                    print(pstring, file=open(os.path.join(gen_dir, "log.txt"), "a"))
                    pstring = "The average interpolation error / PSNR / SSIM  for all images are : "\
                              + str(round(interp_error.avg, 4)) \
                              + " / " + str(round(psnr_total.avg, 4)) \
                              + " / " + str(round(ssim_total.avg, 4))
                    print(pstring)
                    print(pstring, file=open(os.path.join(gen_dir, "log.txt"), "a"))

        # end for folders
        print("Avg psnr", psnr_total.avg)
        print("Avg ssim", ssim_total.avg)
        print("Avg ie",   interp_error.avg)

        # pstring = "runtime per image [s] : %.2f\n" % total_run_time.avg + \
        #           "CPU[1] / GPU[0] : 1 \n" + \
        #           "Extra Data [1] / No Extra Data [0] : 1" + \
        #           "Other description: Our solution based on cvpr'19 paper: Depth-Aware Video Frame Interpolation. We do not use the provided dataset to train."
        # print(pstring)
        # print(pstring, file=open(os.path.join(args.save_path, "readme.txt"), "a"))

                # plt.figure(3)
                # plt.title("GT")
                # plt.imshow(gt_rgb)
                # plt.show()
                #
                # plt.figure(4)
                # plt.title("Ii")
                # plt.imshow(rec_rgb)
                # plt.show()
                #
                # plt.figure(6)
                # plt.title("I_overlay")
                # plt.imshow(((X0 + X1) / 2.0).astype("uint8"))
                # plt.show()

                # diff_rgb = diff_rgb.astype("uint8")
                # plt.figure(5)
                # plt.title("diff")
                # plt.imshow(diff_rgb)
                # plt.show()

                # imsave(os.path.join(gen_dir, dir, "frame10i11_diff" + str('{:.4f}'.format(avg_interp_error_abs)) + ".png"),
                #        diff_rgb)
                # print()



    print("\n\n\n")

if __name__ == '__main__':
    val_AIM(model,args.use_cuda)
