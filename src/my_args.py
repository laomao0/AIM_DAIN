import os
import datetime
import argparse
import numpy
import networks
import  torch
modelnames =  networks.__all__
import datasets
datasetNames = datasets.__all__

parser = argparse.ArgumentParser(description='Flow-Directed Interpolation Kernel')

parser.add_argument('--debug',action = 'store_true', help='Enable debug mode')
parser.add_argument('--netName', type=str, default='MultiScaleStructure_occ_filt_flo',
                    choices = modelnames,help = 'model architecture: ' +
                        ' | '.join(modelnames) +
                        ' (default: MultiScaleStructure_occ_filt_flo)')

parser.add_argument('--datasetName', default='AIM_Challenge',
                    choices= datasetNames,nargs='+',
                    help='dataset type : ' +
                        ' | '.join(datasetNames) +
                        ' (default: Vimeo_90K_interp)')
parser.add_argument('--datasetPath',nargs='+', default=datasets.AIM,help = 'the path of selected datasets')
parser.add_argument('--dataset_split', type = int, default=97, help = 'Split a dataset into trainining and validation by percentage (default: 97)')
parser.add_argument('--single_output', type = int, default=97,
                    help='output single frame or not (multiple frame)')
parser.add_argument('--task', default='interp_blur',
                    choices= ['interp', 'interp_blur'],
                    help= 'specify tasks: interp or interp_blur')

parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument('--numEpoch', '-e', type = int, default=100, help= 'Number of epochs to train(default:150)')

parser.add_argument('--batch_size', '-b',type = int ,default=10, help = 'batch size (default:10)' )
parser.add_argument('--workers', '-w', type =int,default=8, help = 'parallel workers for loading training samples (default : 1.6*10 = 16)')
parser.add_argument('--channels', '-c', type=int,default=3,choices = [1,3], help ='channels of images (default:3)')
parser.add_argument('--filter_size', '-f', type=int, default=4, help = 'the size of filters used (default: 4)',
                    choices=[2,4,6, 5,51]
                    )

parser.add_argument('--test_uid', type=str, default= None, help='unique id for the training')
parser.add_argument('--lr', type =float, default= 0.002, help= 'the basic learning rate for three subnetworks (default: 0.002)')
parser.add_argument('--rectify_lr', type=float, default=0.001, help  = 'the learning rate for rectify/refine subnetworks (default: 0.001)')

parser.add_argument('--save_which', '-s', type=int, default=0, choices=[0,1], help='choose which result to save: 0 ==> interpolated, 1==> rectified')
parser.add_argument('--flowmethod',  type=int, default=0, choices=[0,1,2], help='choose which result to save: 0 ==> flownets, 1==> spynet, 2==>pwcnet')
parser.add_argument('--time_step',  type=float, default=0.5, help='choose the time steps')
parser.add_argument('--flowproj_threshhold',  type=float, default= -0.1, help='to remove the unreliable flow vectors')
parser.add_argument('--flow_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--occ_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--interp_lr_coe', type= float, default=1.0, help='relative learning rate w.r.t to basci learning rate')
parser.add_argument('--filter_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--ctx_lr_coe', type = float, default=1.0, help = 'relative learning rate w.r.t basic learning rate (default: 1.0)')
parser.add_argument('--depth_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')
parser.add_argument('--deblur_lr_coe', type = float, default=0.01, help = 'relative learning rate w.r.t basic learning rate (default: 0.01)')

parser.add_argument('--use_negPSNR', action='store_true', help ='whether to use negPSNR as loss to replace L1-norm loss (default:false)' )

parser.add_argument('--alpha', type=float,nargs='+', default=[1.0, 0.0], help= 'the ration of loss for interpolated and rectified result (default: [1.0, 0.0])')
parser.add_argument('--lambda1', type = float,nargs='+', default=[0.0], help = 'regularize the total variation of flow')
parser.add_argument('--lambda2', type = float,nargs='+', default=[0.0], help = 'regularize the sum of two occlusion maps ')
parser.add_argument('--lambda3', type = float,nargs='+', default=[0.0], help = 'regularize the symmetry of two estimated flow')
parser.add_argument('--lambda4', type= float, nargs='+', default=[0.0], help = 'regularize the consistency of flow and depth gradients')

parser.add_argument('--epsilon', type = float, default=1e-6, help = 'the epsilon for charbonier loss,etc (default: 1e-6)')
parser.add_argument('--weight_decay', type = float, default=0, help = 'the weight decay for whole network ' )
parser.add_argument('--patience', type=int, default=5, help = 'the patience of reduce on plateou')
parser.add_argument('--factor', type = float, default=0.2, help = 'the factor of reduce on plateou')

parser.add_argument('--pretrained', dest='SAVED_MODEL', default=None, help ='path to the pretrained model weights')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder' )
parser.add_argument('--use_cuda', default= True, type = bool, help='use cuda or not')
parser.add_argument('--use_cudnn',default=1,type=int, help = 'use cudnn or not')
parser.add_argument('--gpu_ids', default=0, type=int, help='gpu id')
# parser.add_argument('--nocudnn', dest='use_cudnn', default=)
parser.add_argument('--dtype', default=torch.cuda.FloatTensor, choices = [torch.cuda.FloatTensor,torch.FloatTensor],help = 'tensor data type ')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

parser.add_argument('--image_pair_dir', dest='image_pair_dir', default=None, help ='dir to the test sample')

parser.add_argument('--uid', type=str, default= None, help='unique id for the training')
parser.add_argument('--vis_env', type=str, default='train', help='unique id for the visdom')
parser.add_argument('--force', action='store_true', help='force to override the given uid')
parser.add_argument('--high_fps', action='store_true', help='input 60fps')
parser.add_argument('--val_fps', default=30, help='input 15fps, val 15->val_fps')
parser.add_argument('--N_iter', type=int, default=3, help = 'N_iter')

args = parser.parse_args()


if args.uid == None:
    unique_id = str(numpy.random.randint(0, 100000))
    print("revise the unique id to a random numer " + str(unique_id))
    args.uid = unique_id
    timestamp = datetime.datetime.now().strftime("%a-%b-%d-%H:%M")
    save_path = '../model_weights/'+ args.uid  +'-' + timestamp
else:
    save_path = '../model_weights/'+ str(args.uid)
    print(save_path)

# print("no pth here : " + save_path + "/best"+".pth")
if not os.path.exists(save_path + "/best"+".pth"):
    # print("no pth here : " + save_path + "/best" + ".pth")
    print('create dir')
    os.makedirs(save_path, exist_ok=True)
else:
    if not args.force:
        raise("please use another uid ")
    else:
        print("override this uid" + args.uid)
        for m in range(1,10):
            if not os.path.exists(save_path+"/log.txt.bk" + str(m)):
                shutil.copy(save_path+"/log.txt", save_path+"/log.txt.bk"+str(m))
                shutil.copy(save_path+"/args.txt", save_path+"/args.txt.bk"+str(m))
                break



parser.add_argument('--save_path',default=save_path,help = 'the output dir of weights')
parser.add_argument('--log', default = save_path+'/log.txt', help = 'the log file in training')
parser.add_argument('--arg', default = save_path+'/args.txt', help = 'the args used')

args = parser.parse_args()


with open(args.log, 'w') as f:
    f.close()
with open(args.arg, 'w') as f:
    print(args)
    print(args,file=f)
    f.close()
if args.use_cudnn:
    print("cudnn is used")
    torch.backends.cudnn.benchmark = True  # to speed up the
else:
    print("cudnn is not used")
    torch.backends.cudnn.benchmark = False  # to speed up the

