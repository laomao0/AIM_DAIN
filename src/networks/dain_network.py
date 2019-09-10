# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
# from torch.nn import  init
import torch.nn.init as weight_init
import torchvision
from PIL import Image, ImageOps
from itertools import  product
import torch.nn.functional as F
import torchvision.models as zoo_models
import numpy as np
import sys
import threading
from my_package.FilterInterpolation import  FilterInterpolationModule
from my_package.Interpolation import InterpolationModule

from my_package.FlowProjection import  FlowProjectionModule #,FlowFillholeModule
from my_package.InterpolationCh import  InterpolationChModule
from my_package.DepthFlowProjection import DepthFlowProjectionModule
import models

from Stack import Stack

import spynet
import PWCNet
# import Resnet_models
import S2D_models
from Resblock.BasicBlock import BasicBlock
import Resblock
import MegaDepth
import time

# from my_package.WeigtedFlowProjection import WeightedFlowProjectionModule
class DAIN(torch.nn.Module):
    def __init__(self,
                 batch=128, channel = 3,
                 width= 100, height = 100,
                 scale_num = 3, scale_ratio = 2,
                 temporal = False,
                 filter_size = 4,
                 offset_scale = 1,
                 save_which = 0,
                 flowmethod = 0,
                 timestep=0.5,
                 FlowProjection_threshhold=None,
                 cuda_available=False, cuda_id = 0,
                 training=True):

        # base class initialization
        super(DAIN, self).__init__()

        # class parameters
        self.scale_num = scale_num
        self.scale_ratio = scale_ratio
        self.temporal = temporal

        self.filter_size = filter_size
        self.cuda_available = cuda_available
        self.cuda_id = cuda_id
        self.offset_scale = offset_scale
        self.training = training
        self.timestep = timestep
        self.numFrames = int(1.0 / timestep) - 1
        #assert (timestep == 0.5) # TODO: or else the WeigtedFlowProjection should also be revised... Really Tedious work.
        print("Interpolate " + str(self.numFrames) + " frames")
        self.FlowProjection_threshhold = FlowProjection_threshhold
        # assert width == height

        self.w = []
        self.h = []
        self.ScaleNets_offset = []
        self.ScaleNets_filter = []
        self.ScaleNets_occlusion = []

        self._grid_param = None
        i = 0

        self.initScaleNets_filter,self.initScaleNets_filter1,self.initScaleNets_filter2 = \
            self.get_MonoNet5(channel if i == 0 else channel + filter_size * filter_size, filter_size * filter_size, "filter")

        self.ctxNet = S2D_models.__dict__['S2DF_3dense']()
        self.ctx_ch = 3 * 64 + 3

        # initialize model weights.

        self.rectifyNet = Resblock.__dict__['MultipleBasicBlock_4'](3 + 3 + 3 +2*1+ 2*2 +16*2+ 2 * self.ctx_ch,128)

        self._initialize_weights()


        self.flowmethod = flowmethod
        if flowmethod == 0 :
            self.flownets = models.__dict__['flownets']("models/flownets_pytorch.pth") # for estimating the flow
            self.div_flow = 20.0
        elif flowmethod == 1:
            self.flownets = spynet.Network()
            self.div_flow = 1 # No scaling is used in SPynet
        elif flowmethod == 2:
            self.flownets = PWCNet.__dict__['pwc_dc_net']("PWCNet/pwc_net.pth.tar")
            self.div_flow = 20.0

        #extract depth information
        self.depthNet=MegaDepth.__dict__['HourGlass']("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")

        self.save_which = save_which

        return

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(m)
                count+=1
                # print(count)
                weight_init.xavier_uniform(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)


    def forward(self, input, frame_index):

        """
        The input may compose 3 or 7 frames which should be consistent
        with the temporal settings.

        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        frame_index: the index of timestamp
        -----------
        """
        losses = []
        offsets= []
        filters = []
        occlusions = []

        device = torch.cuda.current_device()
        # print(device)
        # s1 = torch.cuda.Stream(device=device, priority=5)
        # s2 = torch.cuda.Stream(device=device, priority=10) #PWC-Net is slow, need to have higher priority
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        '''
            STEP 1: sequeeze the input 
        '''
        if self.training == True:
            if self.temporal== False:
                assert input.size(0) == 3
                input_0,input_1,input_2 = torch.squeeze(input,dim=0)  # input_2 middle
                input_3,input_4,input_5,input_6 = [],[],[],[]
            else:
                assert input.size(0) == 7
                input_0,input_1,input_2, input_3, input_4, input_5,input_6 = \
                    torch.squeeze(input,dim=0)
        else:
            if self.temporal == False:
                assert input.size(0) ==2
                input_0,input_2 = torch.squeeze(input,dim=0)
                input_1, input_3,input_4,input_5,input_6 = [],[],[],[],[]
            else:
                assert input.size(0) == 4
                input0,input_2,input_4,input_6 = torch.sequeeze(input,dim= 0)
                input_1,input_3,input_5,input7  =  [],[],[],[]


        '''
            STEP 2: initialize the auxiliary input either from temporal or scale predecessor
        '''
        pre_scale_offset, pre_scale_filter, pre_scale_occlusion = None, None, None
        if self.temporal:
            pre_scale_offset_c, pre_scale_filter_c, pre_scale_occlusion_c = None, None, None
            pre_scale_offset_n, pre_scale_filter_n, pre_scale_occlusion_n = None, None, None

        '''
            STEP 3: iteratively execuate the Multiscale Network 
        '''
        # from the coarser scale to the most
        for i in range(self.scale_num):

            '''
                STEP 3.1: prepare current scale inputs
            '''
            #prepare the input data of current scale
            cur_input_0 = F.avg_pool2d(input_0,pow(self.scale_ratio,self.scale_num - i - 1))
            if self.training == True:
                cur_input_1 = F.avg_pool2d(input_1,pow(self.scale_ratio,self.scale_num - i - 1))
            cur_input_2 = F.avg_pool2d(input_2,pow(self.scale_ratio,self.scale_num - i - 1))
            if self.temporal == True:
                # frame 3 is the central frame to be interpolated.
                if self.training == True:
                    cur_input_3 = F.avg_pool2d(input_3, pow(self.scale_ratio,self.scale_num - i - 1))
                cur_input_4 = F.avg_pool2d(input_4, pow(self.scale_ratio,self.scale_num - i - 1))
                if self.training== True:
                    cur_input_5 = F.avg_pool2d(input_5, pow(self.scale_ratio,self.scale_num - i - 1))
                cur_input_6 = F.avg_pool2d(input_6, pow(self.scale_ratio,self.scale_num - i - 1))

            '''
                STEP 3.2: concatenating the inputs.
            '''
            if i == 0:
                cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
                cur_filter_input = cur_offset_input # torch.cat((cur_input_0, cur_input_2), dim=1)
                # cur_occlusion_input = cur_offset_input # torch.cat((cur_input_0, cur_input_2), dim=1)

                if self.temporal==True:
                    # the central part
                    cur_offset_input_c = torch.cat((cur_input_2,cur_input_4),dim = 1)
                    cur_filter_input_c = cur_offset_input_c #torch.cat((cur_input_2,cur_input_4),dim =1)
                    # cur_occlusion_input_c = cur_offset_input_c #torch.cat((cur_input_2,cur_input_4),dim  =1)
                    # the next part
                    cur_offset_input_n = torch.cat((cur_input_4,cur_input_6),dim = 1)
                    cur_filter_input_n = cur_offset_input_n# torch.cat((cur_input_4,cur_input_6),dim = 1)
                    # cur_occlusion_input_n = cur_offset_input_n #torch.cat((cur_input_4,cur_input_6),dim = 1)
                    # # to compose a enlarged batch with the three parts
                    # cur_offset = torch.cat((cur_offset, cur_offset_c, cur_offset_n), dim=0)
                    # cur_filter = torch.cat((cur_filter, cur_filter_c,cur_filter_n), dim=0)
                    # cur_occlusion = torch.cat((cur_occlusion,cur_occlusion_c, cur_occlusion_n), dim=0)
            else:
                cur_offset_input = torch.cat((cur_input_0,cur_input_2,pre_scale_offset),dim=1)
                cur_filter_input = torch.cat((cur_input_0,cur_input_2,pre_scale_filter),dim =1)
                # cur_occlusion_input = torch.cat((cur_input_0,cur_input_2,pre_scale_occlusion),dim=1)

                if self.temporal ==True:
                    cur_offset_input_c = torch.cat((cur_input_2, cur_input_4,pre_scale_offset_c),dim=1)
                    cur_filter_input_c = torch.cat((cur_input_2,cur_input_4, pre_scale_filter_c),dim =1 )
                    # cur_occlusion_input_c = torch.cat((cur_input_2,cur_input_4,pre_scale_occlusion_c),dim = 1)

                    cur_offset_input_n = torch.cat((cur_input_4,cur_input_6,pre_scale_offset_n),dim=1)
                    cur_filter_input_n = torch.cat((cur_input_4,cur_input_6,pre_scale_filter_n),dim=1)
                    # cur_occlusion_input_n = torch.cat((cur_input_4,cur_input_6,pre_scale_occlusion_n),dim=1)

                    # # to compose a enlarged batch with the three parts
                    # cur_offset = torch.cat((cur_offset, cur_offset_c, cur_offset_n), dim=0)
                    # cur_filter = torch.cat((cur_filter, cur_filter_c,cur_filter_n), dim=0)
                    # cur_occlusion = torch.cat((cur_occlusion,cur_occlusion_c, cur_occlusion_n), dim=0)

            '''
                STEP 3.3: perform the estimation by the Three subpath Network 
            '''
            if i ==0 :

                time_offsets = [ kk * self.timestep for kk in range(1, 1+self.numFrames,1)]

                if len(time_offsets) == 1:
                    frame_index = [0]

                # always set depthNet to evaluation mode without optimizing its parameters.
                # self.depthNet = self.depthNet.eval()

                with torch.cuda.stream(s1):
                    temp  = self.depthNet(torch.cat((cur_filter_input[:, :3, ...],
                                                     cur_filter_input[:, 3:, ...]),dim=0))
                    log_depth = [temp[:cur_filter_input.size(0)], temp[cur_filter_input.size(0):]]

                    # print("depth estimation time")
                    # print(time.time() - lasttime)
                    # lasttime = time.time()

                    # log_depth = [self.depthNet(cur_filter_input[:, :3, ...]),
                    #              self.depthNet(cur_filter_input[:, 3:, ...])]
                    # combine the depth with context to
                    cur_ctx_output = [
                        torch.cat((self.ctxNet(cur_filter_input[:, :3, ...]),
                               log_depth[0].detach()), dim=1),
                        torch.cat((self.ctxNet(cur_filter_input[:, 3:, ...]),
                               log_depth[1].detach()), dim=1)
                    ]
                    # print("context extraction time")
                    # print(time.time() - lasttime)
                    # lasttime = time.time()
                    temp = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input, 'filter')
                    cur_filter_output = [self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                                     self.forward_singlePath(self.initScaleNets_filter2, temp, name=None)]

                    # print("filter estimation time")
                    # print(time.time() - lasttime)
                    # lasttime = time.time()
                    # temp = self.forward_singlePath(self.initScaleNets_occlusion,cur_occlusion_input,'occlusion')
                    # cur_occlusion_output = [self.forward_singlePath(self.initScaleNets_occlusion1,temp,name=None),
                    #                         self.forward_singlePath(self.initScaleNets_occlusion2,temp,name=None)]

                    depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

                with torch.cuda.stream(s2):
                    # use the occlusion as the depthmap outpu
                    for _ in range(1):
                        cur_offset_outputs = [
                            self.forward_flownets(self.flownets, cur_offset_input, time_offsets=time_offsets,  # F_0_t
                                                      flowmethod=self.flowmethod),
                            self.forward_flownets(self.flownets, torch.cat((cur_offset_input[:, 3:, ...],      # F_1_t
                                                                                cur_offset_input[:, 0:3, ...]), dim=1),
                                                      time_offsets=time_offsets[::-1],
                                                      flowmethod=self.flowmethod)
                        ]

                torch.cuda.synchronize() #synchronize s1 and s2

                for _ in range(1):
                    cur_offset_outputs = [
                        self.FlowProject(cur_offset_outputs[0],depth_inv[0],
                                         self.FlowProjection_threshhold,
                                         refinputs=[cur_offset_input[:,0:3,...],cur_offset_input[:,3:,...]] ),
                        self.FlowProject(cur_offset_outputs[1],depth_inv[1],
                                        self.FlowProjection_threshhold,refinputs=[ cur_offset_input[:,3:,...], cur_offset_input[:,0:3,...]])
                    ]

                    # print("flow estimation time")
                    # print(time.time() - lasttime)

                # lasttime = time.time()
                depth_inv_maxreg = [d / torch.max(d) for d in depth_inv]
                cur_occlusion_output = [
                    depth_inv_maxreg[0],depth_inv_maxreg[1]
                    # Variable(torch.cuda.FloatTensor().resize_(cur_filter_input.size(0), 1, cur_filter_input.size(2),
                    #                                           cur_filter_input.size(3)).zero_()),
                    # Variable(torch.cuda.FloatTensor().resize_(cur_filter_input.size(0), 1, cur_filter_input.size(2),
                    #                                           cur_filter_input.size(3)).zero_()),
                    # 0.5 * Variable(torch.ones(cur_filter_input.size(0),1,cur_filter_input.size(2),cur_filter_input.size(3)).type(cur_filter_input.data.type())),
                    # 0.5 * Variable(torch.ones(cur_filter_input.size(0),1,cur_filter_input.size(2),cur_filter_input.size(3)).type(cur_filter_input.data.type())),
                ]


                if self.temporal:
                    cur_offset_output_c = self.forward_singlePath(self.initScaleNets_offset,cur_offset_input_c)
                    cur_offset_output_n = self.forward_singlePath(self.initScaleNets_offset,cur_offset_input_n)

                    cur_filter_output_c = self.forward_singlePath(self.initScaleNets_filter, cur_filter_input_c)
                    cur_filter_output_n = self.forward_singlePath(self.initScaleNets_filter,cur_filter_input_n)

                    cur_occlusion_output_c = self.forward_singlePath(self.initScaleNets_occlusion,cur_occlusion_input_c)
                    cur_occlusion_output_n = self.forward_singlePath(self.initScaleNets_occlusion,cur_occlusion_input_n)
            else:
                cur_offset_output = self.forward_singlePath(self.iterScaleNets_offset, cur_offset_input)
                cur_filter_output = self.forward_singlePath(self.iterScaleNets_filter,cur_filter_input)
                cur_occlusion_output = self.forward_singlePath(self.iterScaleNets_occlusion,cur_occlusion_input)
                if self.temporal:
                    cur_offset_output_c = self.forward_singlePath(self.iterScaleNets_offset,cur_offset_input_c)
                    cur_offset_output_n = self.forward_singlePath(self.iterScaleNets_offset,cur_offset_input_n)

                    cur_filter_output_c = self.forward_singlePath(self.iterScaleNets_filter,cur_filter_input_c)
                    cur_filter_output_n = self.forward_singlePath(self.iterScaleNets_filter,cur_filter_input_n)

                    # cur_occlusion_output_c = self.forward_singlePath(self.iterScaleNets_occlusion,cur_occlusion_input_c)
                    # cur_occlusion_output_n = self.forward_singlePath(self.iterScaleNets_occlusion,cur_occlusion_input_n)

            '''
                STEP 3.4: perform the frame interpolation process 
            '''



            timeoffset = time_offsets[frame_index[0]]
            temp_0 = cur_offset_outputs[0][frame_index[0]]
            temp_1 = cur_offset_outputs[1][frame_index[0]]
            cur_offset_output = [temp_0, temp_1]
            ctx0, ctx2 = self.FilterInterpolate_ctx(cur_ctx_output[0],cur_ctx_output[1],cur_offset_output,cur_filter_output, timeoffset)

            cur_output, ref0, ref2 = self.FilterInterpolate(cur_input_0, cur_input_2, cur_offset_output,
                                                                 cur_filter_output, self.filter_size ** 2,
                                                                 timeoffset)

            cur_occlusion_output = self.Interpolate_ch(cur_occlusion_output[0], cur_occlusion_output[1],
                                                       cur_offset_output, 1)

            rectify_input = torch.cat((cur_output, ref0, ref2,
                                       cur_offset_output[0], cur_offset_output[1],
                                       cur_filter_output[0], cur_filter_output[1],
                                       ctx0, ctx2
                                       ), dim=1)

            cur_output_rectified = self.rectifyNet(rectify_input) + cur_output


            if self.temporal ==True:
                cur_output_c = self.Interpolate(cur_input_2,cur_input_4,cur_offset_output_c,cur_filter_output_c,cur_occlusion_output_c)
                cur_output_n = self.Interpolate(cur_input_4,cur_input_6,cur_offset_output_n,cur_filter_output_n,cur_occlusion_output_n)

                temp, forward = torch.split(cur_offset_output, 2, dim=1)
                forward = -forward
                backward, temp = torch.split(cur_offset_output_n,2,dim=1)
                backward = -backward

                cur_offset_sym = torch.cat((forward,backward),dim = 1)
                cur_filter_sym = cur_filter_output
                cur_occlusion_sym = cur_occlusion_output
                cur_output_sym = self.Interpolate(cur_input_2,cur_input_4,cur_offset_sym, cur_filter_sym,cur_occlusion_sym)


            '''
                STEP 3.5: for training phase, we collect the variables to be penalized.
            '''
            if self.training == True:
                losses +=[cur_output - cur_input_1]
                losses += [cur_output_rectified - cur_input_1]                
                offsets +=[cur_offset_output]
                filters += [cur_filter_output]
                occlusions += [cur_occlusion_output]
                if self.temporal == True:
                    losses+= [cur_output_c - cur_input_3]
                    losses+= [cur_output_n - cur_input_5]
                    losses+= [cur_output_c - cur_output_sym]

            '''
                STEP 3.6: prepare inputs for the next finer scale
            '''
            if self.scale_num > 1:
                ## prepare for the next finer scale's  requirements.
                pre_scale_offset = F.upsample(cur_offset_output * self.scale_ratio,         scale_factor=self.scale_ratio,mode='bilinear')
                pre_scale_filter = F.upsample(cur_filter_output,                            scale_factor=self.scale_ratio,mode='bilinear')
                pre_scale_occlusion = F.upsample(cur_offset_output,                         scale_factor=self.scale_ratio,mode='bilinear')
                if self.temporal == True:
                    pre_scale_offset_c = F.upsample(cur_offset_output_c * self.scale_ratio, scale_factor= self.scale_ratio,mode='bilinear')
                    pre_scale_filter_c = F.upsample(cur_filter_output_c,                    scale_factor=self.scale_ratio,mode='bilinear')
                    pre_scale_occlusion_c = F.upsample(cur_occlusion_output_c,              scale_factor=self.scale_ratio,mode='bilinear')

                    pre_scale_offset_n = F.upsample(cur_offset_output_n * self.scale_ratio, scale_factor= self.scale_ratio,mode='bilinear')
                    pre_scale_filter_n = F.upsample(cur_filter_output_n,                    scale_factor=self.scale_ratio, mode='bilinear')
                    pre_scale_occlusion_n = F.upsample(cur_occlusion_output_n,              scale_factor=self.scale_ratio, mode='bilinear')

        '''
            STEP 4: return the results
        '''
        if self.training == True:

            return losses, offsets,filters,occlusions
        else:
            # if in test phase, we directly return the interpolated frame
            if self.temporal == False:
                cur_outputs = [cur_output,cur_output_rectified]
                return cur_outputs,cur_offset_output,cur_filter_output,cur_occlusion_output
            else:
                return cur_output_c, cur_output_sym

    def forward_flownets(self, model, input, time_offsets = None, flowmethod = 0):

        if time_offsets == None :
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        # flowlists =  []
        # start=  time.time()
        if flowmethod == 0 or flowmethod == 2:  # spynet don't need upsample
            temp = model(input)  # this is a single direction motion results, but not a bidirectional one

            temps = [self.div_flow * temp * time_offset for time_offset in time_offsets]# single direction to bidirection should haven it.
            temps = [nn.Upsample(scale_factor=4, mode='bilinear')(temp)  for temp in temps]# nearest interpolation won't be better i think
        elif flowmethod == 1:
            temp = model(input[:,:3,:,:],input[:,3:,:,:])

            temps = [temp * time_offset for time_offset in time_offsets]
        # print("Flow Estimate and Upsample time is " +str(time.time()- start))
            # print(temp.size())
        return temps
    # @staticmethod
    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()
        # if self.temporal:
        #     stack_c = Stack()
        #     stack_n = Stack()

        k = 0
        temp = []
        # if self.temporal:
        #     temp_c = []
        #     temp_n = []
        for layers in modulelist:  # self.initScaleNets_offset:
            # TODO: we need to store the sequential results so as to add a skip connection into the whole model.
            # print(type(layers).__name__)
            # print(k)
            # if k == 27:
            #     print(k)
            #     pass
            # use the pop-pull logic, looks like a stack.
            if k == 0:
                temp = layers(input)
                # if self.temporal:
                #     temp_c = layers(input_c)
                #     temp_n = layers(input_n)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers,nn.MaxPool2d):
                    stack.push(temp)
                    # if self.temporal:
                    #     stack_c.push(temp_c)
                    #     stack_n.push(temp_n)

                temp = layers(temp)
                # if self.temporal:
                #     temp_c = layers(temp_c)
                #     temp_n = layers(temp_n)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == 'offset':
                        temp = torch.cat((temp,stack.pop()),dim=1)  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += stack.pop()  # short cut here, but optical flow should concat instead of add
                    # if self.temporal:
                    #     temp_c += stack_c.pop()
                    #     temp_n += stack_n.pop()

            k += 1
        # if self.temporal == False:
        return temp
        # else:
            # return temp, temp_c, temp_n


    def get_MonoNet5(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        # model += self.conv_relu(32,32,(3,3),(1,1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(16, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        # model += self.conv_relu(32, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        model += self.conv_relu_maxpool(32, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        # model += self.conv_relu(64, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(64, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        # model += self.conv_relu(128, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(128, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        # model += self.conv_relu(256, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))# THE OUTPUT No.1
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        # model += self.conv_relu(512+512 if name == "offset" else 512, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # block 7
        model += self.conv_relu_unpool(256, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        # model += self.conv_relu(256+256 if name == "offset" else 256, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
        # model += self.conv_relu(128+128 if name == "offset" else 128, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
        # model += self.conv_relu(64+64 if name == "offset" else 64, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))

        # block 10
        model += self.conv_relu_unpool(32,  16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
        # model += self.conv_relu(32+32 if name == "offset" else 32, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out,  (3, 3), (1, 1))

        if name == "offset":
            branch1 += self.sigmoid_activation() # limit to 0~1
            branch2 += self.sigmoid_activation()
            pass
        elif name == "filter":
            # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
            # since i use a non-normalized distance weighted , then the learned filter is also non-normalized.
            # We only have to make it a positive value.
            # model += self.softmax_activation()
            # model += self.relu_activation()
            # model = self.binary_activation() # we shouldn't have used the relu because each participated pixel should have a weight larget than zeros
            pass

        elif name == "occlusion":
            # we need to get a binary occlusion map for both reference frames
            # model += self.binary_activation()
            # model  += self.softmax_activation()
            pass # we leave all the three branched no special layer.
        return  (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    def get_RectifyNet3(self, channel_in, channel_out):
        model = []
        # model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
        model.append(self.conv_relu(channel_in, 64, (3, 3), (1, 1)))
        model.append(BasicBlock(64, 64, dilation=1))
        model.append(BasicBlock(64, 64, dilation=1))
        model.append(BasicBlock(64, 64, dilation=1))
        model.append(BasicBlock(64, 64, dilation=1))
        model.append(nn.Sequential(*[nn.Conv2d(64, channel_out, (3, 3), 1, (1, 1))]))
        return nn.ModuleList(model)

    def get_RectifyNet2(self, channel_in, channel_out):
        model = []
        #model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
        model += self.conv_relu(channel_in, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += self.conv_relu(64, 64, (3,3), (1,1))
        model += nn.Sequential(*[nn.Conv2d(64,channel_out,(3,3),1, (1,1))])
        return nn.ModuleList(model)
        
    def get_RectifyNet(self, channel_in, channel_out):
        model = []
        model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
        return nn.ModuleList(model)
    
    def get_MonoNet4(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32,32,(3,3),(1,1))
        model += self.conv_relu(32, 32, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(32, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu(32, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        model += self.conv_relu_maxpool(64, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu(64, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(128, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu(128, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(256, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu(256, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))# THE OUTPUT No.1
        model += self.conv_relu_maxpool(512, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(512, 512, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        model += self.conv_relu(512+512 if name == "offset" else 512, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # block 7
        model += self.conv_relu_unpool(256, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        model += self.conv_relu(256+256 if name == "offset" else 256, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # block 8
        model += self.conv_relu_unpool(128, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
        model += self.conv_relu(128+128 if name == "offset" else 128, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))

        # block 9
        model += self.conv_relu_unpool(64, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
        model += self.conv_relu(64+64 if name == "offset" else 64, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))

        # block 10
        model += self.conv_relu_unpool(32,  32, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
        model += self.conv_relu(32+32  if name == "offset" else 32, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))

        # output our final purpose
        model += self.conv_relu_conv(16, channel_out * 2, (3, 3), (1, 1))

        if name == "offset":
            model += self.sigmoid_activation() # limit to 0~1
            # pass
        elif name == "filter":
            # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
            # since i use a non-normalized distance weighted , then the learned filter is also non-normalized.
            # We only have to make it a positive value.
            # model += self.softmax_activation()
            # model += self.relu_activation()
            # model = self.binary_activation() # we shouldn't have used the relu because each participated pixel should have a weight larget than zeros
            pass

        elif name == "occlusion":
            # we need to get a binary occlusion map for both reference frames
            # model += self.binary_activation()
            # model  += self.softmax_activation()
            pass # we leave all the three branched no special layer.
        return model

    def get_MonoNet3(self, channel_in, channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32,32,(3,3),(1,1))
        model += self.conv_relu(32, 32, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(32, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
        # block2
        model += self.conv_relu(32, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        # model += self.conv_relu(64,64,(3,3),(1,1))
        model += self.conv_relu_maxpool(64, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
        # block3
        model += self.conv_relu(64, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(128, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
        # block4
        model += self.conv_relu(128, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256, 256, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(256, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
        # block5
        model += self.conv_relu(256, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))
        # model += self.conv_relu(512, 512, (3, 3), (1, 1))# THE OUTPUT No.1
        model += self.conv_relu_maxpool(512, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))
        model += self.conv_relu(512, 512, (3, 3), (1, 1))
        model += self.conv_relu(512, 512, (3, 3), (1, 1))


        # block 6
        model += self.conv_relu_unpool(512, 512, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
        model += self.conv_relu(512, 256, (3, 3), (1, 1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # model += self.conv_relu(256,256,(3,3),(1,1))
        # block 7
        model += self.conv_relu_unpool(256, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
        model += self.conv_relu(256, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # model += self.conv_relu(128, 128, (3, 3), (1, 1))
        # block 8
        model += self.conv_relu_unpool(128, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
        model += self.conv_relu(128, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))
        # model += self.conv_relu(64, 64, (3, 3), (1, 1))

        # block 9
        model += self.conv_relu_unpool(64, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
        model += self.conv_relu(64, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))
        # model += self.conv_relu(32, 32, (3, 3), (1, 1))

        # block 10
        model += self.conv_relu_unpool(32, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
        model += self.conv_relu(32, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))
        # model += self.conv_relu(16, 16, (3, 3), (1, 1))

        # output our final purpose
        model += self.conv_relu_conv(16, channel_out * 2, (3, 3), (1, 1))

        if name == "offset":
            pass
        elif name == "filter":
            # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
            model += self.softmax_activation()
        elif name == "occlusion":
            # we need to get a binary occlusion map for both reference frames
            model += self.binary_activation()
        return model

    def get_MonoNet2(self, channel_in, channel_out, name):

                '''
                Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

                :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
                :param channel_out: number of output the offset or filter or occlusion
                :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

                :return: output the network model
                '''
                model = []

                # block1
                model += self.conv_relu_bnorm(channel_in * 2, 32, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(32,32,(3,3),(1,1))
                model += self.conv_relu_bnorm(32, 32, (3, 3), (1, 1))
                model += self.conv_relu_bnorm_pool(32, 32, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.5
                # block2
                model += self.conv_relu_bnorm(32, 64, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(64,64,(3,3),(1,1))
                # model += self.conv_relu_bnorm(64,64,(3,3),(1,1))
                model += self.conv_relu_bnorm_pool(64, 64, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.4
                # block3
                model += self.conv_relu_bnorm(64, 128, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
                model += self.conv_relu_bnorm_pool(128, 128, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.3
                # block4
                model += self.conv_relu_bnorm(128, 256, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(256, 256, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(256, 256, (3, 3), (1, 1))
                model += self.conv_relu_bnorm_pool(256, 256, (3, 3), (1, 1), (2, 2))  # THE OUTPUT No.2
                # block5
                model += self.conv_relu_bnorm(256, 512, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))# THE OUTPUT No.1
                model += self.conv_relu_bnorm_pool(512, 512, (3, 3), (1, 1), (2, 2))

                # intermediate block5_5
                model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))


                # block 6
                model += self.conv_relu_bnorm_unpool(512, 512, (3, 3), (1, 1), 2)  # THE OUTPUT No.1 UP
                model += self.conv_relu_bnorm(512, 256, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(256,256,(3,3),(1,1))
                # model += self.conv_relu_bnorm(256,256,(3,3),(1,1))
                # block 7
                model += self.conv_relu_bnorm_unpool(256, 256, (3, 3), (1, 1), 2)  # THE OUTPUT No.2 UP
                model += self.conv_relu_bnorm(256, 128, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
                # block 8
                model += self.conv_relu_bnorm_unpool(128, 128, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP
                model += self.conv_relu_bnorm(128, 64, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(64, 64, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(64, 64, (3, 3), (1, 1))

                # block 9
                model += self.conv_relu_bnorm_unpool(64, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP
                model += self.conv_relu_bnorm(64, 32, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(32, 32, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(32, 32, (3, 3), (1, 1))

                # block 10
                model += self.conv_relu_bnorm_unpool(32, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP
                model += self.conv_relu_bnorm(32, 16, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(16, 16, (3, 3), (1, 1))
                # model += self.conv_relu_bnorm(16, 16, (3, 3), (1, 1))

                # output our final purpose
                model += self.conv_relu_conv(16, channel_out * 2, (3, 3), (1, 1))

                if name == "offset":
                    pass
                elif name == "filter":
                    # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
                    model += self.softmax_activation()
                elif name == "occlusion":
                    # we need to get a binary occlusion map for both reference frames
                    model += self.binary_activation()
                return model

    # @staticmethod without self as first param
    def get_MonoNet(self,channel_in,channel_out, name):

        '''
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        '''
        model = []

        # block1
        model += self.conv_relu_bnorm(channel_in *  2 ,32,(3,3),(1,1))
        model += self.conv_relu_bnorm(32,32,(3,3),(1,1))
        model += self.conv_relu_bnorm(32,32,(3,3),(1,1))
        model += self.conv_relu_bnorm_pool(32,32,(3,3),(1,1),(2,2))# THE OUTPUT No.5
        # block2
        model += self.conv_relu_bnorm(32,64,(3,3),(1,1))
        model += self.conv_relu_bnorm(64,64,(3,3),(1,1))
        model += self.conv_relu_bnorm(64,64,(3,3),(1,1))
        model += self.conv_relu_bnorm_pool(64,64,(3,3),(1,1),(2,2))# THE OUTPUT No.4
        # block3
        model += self.conv_relu_bnorm(64, 128, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_bnorm_pool(128, 128, (3, 3), (1, 1),(2,2))# THE OUTPUT No.3
        # block4
        model += self.conv_relu_bnorm(128, 256, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(256, 256, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(256, 256, (3, 3), (1, 1))
        model += self.conv_relu_bnorm_pool(256, 256, (3, 3), (1, 1),(2,2))# THE OUTPUT No.2
        # block5
        model += self.conv_relu_bnorm(256, 512, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))# THE OUTPUT No.1
        model += self.conv_relu_bnorm_pool(512, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(512, 512, (3, 3), (1, 1))


        #block 6
        model += self.conv_relu_bnorm_unpool(512, 512, (3, 3), (1, 1),2) # THE OUTPUT No.1 UP
        model += self.conv_relu_bnorm(512,256,(3,3),(1,1))
        model += self.conv_relu_bnorm(256,256,(3,3),(1,1))
        model += self.conv_relu_bnorm(256,256,(3,3),(1,1))
        #block 7
        model += self.conv_relu_bnorm_unpool(256, 256, (3, 3), (1, 1), 2)# THE OUTPUT No.2 UP
        model += self.conv_relu_bnorm(256, 128, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(128, 128, (3, 3), (1, 1))
        # block 8
        model += self.conv_relu_bnorm_unpool(128, 128, (3, 3), (1, 1), 2)# THE OUTPUT No.3 UP
        model += self.conv_relu_bnorm(128, 64, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(64, 64, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(64, 64, (3, 3), (1, 1))

        # block 9
        model += self.conv_relu_bnorm_unpool(64, 64, (3, 3), (1, 1), 2)# THE OUTPUT No.4 UP
        model += self.conv_relu_bnorm(64, 32, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(32, 32, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(32, 32, (3, 3), (1, 1))

        # block 10
        model += self.conv_relu_bnorm_unpool(32, 32, (3, 3), (1, 1), 2)# THE OUTPUT No.5 UP
        model += self.conv_relu_bnorm(32, 16, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(16, 16, (3, 3), (1, 1))
        model += self.conv_relu_bnorm(16, 16, (3, 3), (1, 1))

        #output our final purpose
        model += self.conv_relu_conv(16,channel_out * 2,(3,3),(1,1))


        if name == "offset":
            pass
        elif name== "filter":
            # TODO: Sigmoid is not used in Simon's implementation, which mean we dont acutally need to explicityly use a filter with weights summed to exactly 1.0.
            model+= self.softmax_activation()
        elif name == "occlusion":
            # we need to get a binary occlusion map for both reference frames
            model+=self.binary_activation()
        return model



    # @staticmethod
    # def Interpolate(ip0,ip1, ref0,ref2,offset,filter,occlusion):
    #     ref0_offset = ip0(ref0, offset[:, :2, ...])
    #     ref2_offset = ip1(ref2, offset[:, 2:, ...])
    #     return  ref0_offset/2.0 + ref2_offset/2.0
    @staticmethod
    def FlowProject(inputs, depth = None, FlowProjectionthreshold = None, refinputs=None):
        # print(input.requires_grad)
        # start  = time.time()
        # print("flow projection")
        if depth is not None:
            # print("using Depth aware projection")
            outputs = [DepthFlowProjectionModule(input.requires_grad)(input,depth) for input in inputs]
        elif FlowProjectionthreshold is not None:
            outputs = [ WeightedFlowProjectionModule(FlowProjectionthreshold,input.requires_grad)(input,refinputs[0],refinputs[1]) for input in inputs]
        else:
            outputs = [ FlowProjectionModule(input.requires_grad)(input) for input in inputs]
        # print("Flow Project time is " + str(time.time() - start))
        # if output.requires_grad == True:
        #     output = FlowFillholeModule()(input,hole_value = -10000.0)
        return outputs

    @staticmethod
    def fillHole(input,ref0,ref2, hole_value = 0.0):
        index = input == hole_value
        output = input.clone()
        output[index] = (ref0[index] + ref2[index]) /2.0

        return output
    @staticmethod
    def FilterInterpolate_ctx(ctx0,ctx2,offset,filter,timeoffset=None):  # shenwang add
        ##TODO: which way should I choose

        ctx0_offset = FilterInterpolationModule()(ctx0,offset[0].detach(),filter[0].detach())
        ctx2_offset = FilterInterpolationModule()(ctx2,offset[1].detach(),filter[1].detach())

        return ctx0_offset, ctx2_offset
        # ctx0_offset = FilterInterpolationModule()(ctx0.detach(), offset[0], filter[0])
        # ctx2_offset = FilterInterpolationModule()(ctx2.detach(), offset[1], filter[1])
        #
        # return ctx0_offset, ctx2_offset

    '''Keep this function'''
    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter,filter_size2, time_offset): # shenwang add
        ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])

        # occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
        # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
        # output = * ref0_offset + occlusion[1] * ref2_offset
        # automatically broadcasting the occlusion to the three channels of and image.
        # return output
        # return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset
        return ref0_offset*(1.0 - time_offset) + ref2_offset*(time_offset), ref0_offset, ref2_offset

    # @staticmethod
    # def FilterInterpolate(ref0, ref2, offset, filter,filter_size2):
    #     ref0_offset = FilterInterpolationModule()(ref0, offset[0],filter[0])
    #     ref2_offset = FilterInterpolationModule()(ref2, offset[1],filter[1])
    #
    #     # occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
    #     # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
    #     # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
    #     # output = * ref0_offset + occlusion[1] * ref2_offset
    #     # automatically broadcasting the occlusion to the three channels of and image.
    #     # return output
    #     return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset

    # @staticmethod
    # def FilterInterpolate(ref0, ref2, offset, filter, occlusion,filter_size2):
    #     ref0_offset = FilterInterpolationModule()(ref0, offset[:, :2, ...],filter[:,:filter_size2,...])
    #     ref2_offset = FilterInterpolationModule()(ref2, offset[:, 2:, ...],filter[:,filter_size2:,...])
    #
    #     occlusion0, occlusion2 = torch.split(occlusion, 1, dim=1)
    #     # print((occlusion0[0,0,1,1] + occlusion2[0,0,1,1]))
    #     # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset) / (occlusion0 + occlusion2)
    #     output = occlusion0 * ref0_offset + occlusion2 * ref2_offset
    #     # automatically broadcasting the occlusion to the three channels of and image.
    #     return output
    #     # return ref0_offset/2.0 + ref2_offset/2.0
    @staticmethod
    def Interpolate_ch(ctx0,ctx2,offset,ch):
        ctx0_offset = InterpolationChModule(ch)(ctx0,offset[0].detach())
        ctx2_offset = InterpolationChModule(ch)(ctx2,offset[1].detach())

        return ctx0_offset, ctx2_offset
    @staticmethod
    def Interpolate(ref0,ref2,offset,filter=None,occlusion=None):

        ref0_offset = InterpolationModule()(ref0, offset[0])
        ref2_offset = InterpolationModule()(ref2, offset[1])

        # occlusion0, occlusion2 = torch.split(occlusion,1, dim=1)

        # output = (occlusion0 * ref0_offset + occlusion2 * ref2_offset)/(occlusion0 + occlusion2)
        # output = occlusion0 * ref0_offset + occlusion2 * ref2_offset
        # automatically broadcasting the occlusion to the three channels of and image.
        # return  output
        #return ref0_offset/2.0 + ref2_offset/2.0
        return ref0_offset/2.0 + ref2_offset/2.0, ref0_offset,ref2_offset

    # def Interpolate(self,ref0,ref2,offset,filter,occlusion):
    #     '''
    #     A self-customized layer for interpolating a frame guided by the reference frame with offset, filter, and occlusion maps
    #
    #     :param ref0: previous reference frame
    #     :param ref2: next reference frame
    #     :param offset: optical flow or say motion offset from the reference frame to the intermediate frame
    #     :param filter: the interpolation filter
    #     :param occlusion: the occlusion map of both reference frame with respect to the intermediate frame
    #
    #     :return iframe: the generated intermediate frame
    #     '''
    #     x_shape = ref0.size()
    #
    #     offset = 16 * self._to_bc_h_w_4(offset,x_shape) # to manually increase the sensitivity by 10 times
    #
    #     # to permute the reference frame to be CBHW order
    #     ref0 = self._permute_b_c(ref0)
    #     ref2 = self._permute_b_c(ref2)
    #
    #     ref0_offset, ref2_offset = th_batch_map_offsets(ref0, ref2, offset, grid = self._get_grid(self,ref0))
    #
    #     # x_offset: (b, h, w, c)
    #     ref0_offset = self._to_b_c_h_w(ref0_offset, x_shape)
    #     ref2_offset = self._to_b_c_h_w(ref2_offset, x_shape)
    #
    #     occlusion = self._permute_b_c(occlusion)
    #     # use the occlusion map
    #     # iframe = torch.stack(
    #     #     (
    #     #         ref0_offset[0, ...] * occlusion[0, ...] + ref2_offset[0, ...] * occlusion[1,...],
    #     #         ref0_offset[1, ...] * occlusion[0,...] + ref2_offset[1, ...] * occlusion[1,...],
    #     #         ref0_offset[2, ...] * occlusion[0,...] + ref2_offset[2, ...] * occlusion[1,...]
    #     #     ),
    #     #     dim=0
    #     # )
    #
    #     iframe = ref0_offset/2.0 + ref2_offset/2.0
    #
    #     iframe = self._permute_b_c(iframe)
    #
    #     return iframe

    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size,
                        padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(output_filter, output_filter, kernel_size, 1, padding),
        )
        return layers


    @staticmethod
    def sigmoid_activation():
        layers = nn.Sequential(
            # No need to use a Sigmoid2d, since we just focus on one
            nn.Sigmoid()
        )
        return layers

    @staticmethod
    def relu_activation():
        layers = nn.Sequential(
            # No need to use a Sigmoid2d, since we just focus on one
            nn.ReLU(inplace=False)
        )
        return layers

    @staticmethod
    def prelu_activation():
        layers = nn.Sequential(
            # No need to use a Sigmoid2d, since we just focus on one
            nn.PReLU()
        )
        return layers

    @staticmethod
    def softmax_activation():
        layers = nn.Sequential(
            nn.Softmax2d()
        )
        return layers


    @staticmethod
    def conv_relu_bnorm(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter)
        ])
        return layers

    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size,
                        padding):
        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False)
        ])
        return layers

    @staticmethod
    def conv_relu_maxpool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),

            nn.MaxPool2d(kernel_size_pooling)
        ])
        return layers


    @staticmethod
    def conv_relu_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear'),

            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            # nn.BatchNorm2d(output_filter),


            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers


    @staticmethod
    def conv_relu_bnorm_pool(input_filter, output_filter, kernel_size,
                            padding,kernel_size_pooling):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter),

            nn.AvgPool2d(kernel_size_pooling)
        ])
        return layers


    @staticmethod
    def conv_relu_bnorm_unpool(input_filter, output_filter, kernel_size,
                            padding,unpooling_factor):

        layers = nn.Sequential(*[
            nn.Conv2d(input_filter,output_filter,kernel_size,1, padding),

            nn.ReLU(inplace=False),

            nn.BatchNorm2d(output_filter),

            nn.Upsample(scale_factor=unpooling_factor, mode='bilinear')
            # nn.UpsamplingBilinear2d(unpooling_size,scale_factor=unpooling_size[0])
        ])
        return layers

