import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch
from Resblock.BasicBlock import  BasicBlock
__all__ =['get_RectifyNet3']


def conv_relu(input_filter, output_filter, kernel_size,
              padding):
    layers = nn.Sequential(*[
        nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),

        nn.ReLU(inplace=False)
    ])
    return layers
def get_RectifyNet3(channel_in, channel_out):
    model = []
    # model += self.conv_relu_conv(channel_in, channel_out, (7, 7), (3, 3))
    model.append(conv_relu(channel_in, 64, (3 ,3), (1 ,1)))

    model.append(BasicBlock(64, 64, dilation = 1))
    model.append(BasicBlock(64, 64, dilation = 1))
    model.append(BasicBlock(64, 64, dilation = 1))
    model.append(BasicBlock(64, 64, dilation = 1))
    model.append(nn.Sequential(*[nn.Conv2d(64 ,channel_out ,(3 ,3) ,1, (1 ,1))]))
    return nn.ModuleList(model)