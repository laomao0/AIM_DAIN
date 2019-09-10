# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import _ext.my_lib as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class WeightedFlowProjectionLayer(Function):
    def __init__(self,threshold, requires_grad):
        super(WeightedFlowProjectionLayer,self).__init__()
        self.threshold = threshold
        self.requires_grad = requires_grad

    def forward(self, input1,input2,input3):
        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        self.input2 = input2.contiguous()
        self.input3 = input3.contiguous()

        self.fillhole = 1 if self.requires_grad == False else 0
        # if input1.is_cuda:
        #    self.device = torch.cuda.current_device()
        # else:
        #     self.device = -1

        # count = torch.zeros(input1.size(0),1,input1.size(2),input1.size(3)) # for accumulating the homography projections
        # weight = torch.zeros(input1.size(0), 1, input1.size(2), input1.size(3))
        # output = torch.zeros(input1.size())

        if input1.is_cuda :

            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            weight = torch.cuda.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            count = torch.cuda.FloatTensor().resize_(input1.size(0),1,input1.size(2),input1.size(3)).zero_()
            # output = output.cuda()
            # weight  =weight.cuda()
            # count = count.cuda()

            err = my_lib.WeightedFlowProjectionLayer_gpu_forward(input1, input2,input3,
                                                                 count,weight,  output, self.fillhole,self.threshold)
        else:
            output = torch.FloatTensor().resize_(input1.size()).zero_()
            weight = torch.FloatTensor().resize_(input1.size(0), 1, input1.size(2), input1.size(3)).zero_()
            count = torch.FloatTensor().resize_(input1.size(0),1,input1.size(2),input1.size(3)).zero_()
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.WeightedFlowProjectionLayer_cpu_forward(input1, input2,input3,
                                                                 count, weight, output,self.fillhole,self.threshold)
        if err != 0:
            print(err)
        # output = output/count # to divide the counter

        self.count = count #to keep this
        self.weight = weight
        # print(self.input1[0, 0, :10, :10])
        # print(self.count[0, 0, :10, :10])
        # print(self.input1[0, 0, -10:, -10:])
        # print(self.count[0, 0, -10:, -10:])

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of Filter Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        # gradinput1 = torch.zeros(self.input1.size())
        # gradinput2 = torch.zeros(self.input2.size()) # but actually, input2 and input3 don't need gradients at all
        # gradinput3 = torch.zeros(self.input3.size())

        # gradinput1 = torch.cuda.FloatTensor().resize_(self.input1.size()).zero_()
        # gradinput2 = torch.cuda.FloatTensor().resize_(self.input2.size()).zero_()
        # gradinput3 = torch.cuda.FloatTensor().resize_(self.input3.size()).zero_()

        if self.input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            gradinput1 = torch.cuda.FloatTensor().resize_(self.input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(self.input2.size()).zero_()
            gradinput3 = torch.cuda.FloatTensor().resize_(self.input3.size()).zero_()
            err = my_lib.WeightedFlowProjectionLayer_gpu_backward(self.input1,self.input2,self.input3,
                                                                  self.count, self.weight, gradoutput, gradinput1,  self.threshold)
            # print(err)
            if err != 0 :
                print(err)
        else:
            # print("CPU backward")
            # print(gradoutput)
            gradinput1 = torch.FloatTensor().resize_(self.input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(self.input2.size()).zero_()
            gradinput3 = torch.FloatTensor().resize_(self.input3.size()).zero_()
            err = my_lib.WeightedFlowProjectionLayer_cpu_backward(self.input1, self.input2,  self.input3,
                                                                  self.count, self.weight,  gradoutput, gradinput1, self.threshold)
            # print(err)
            if err != 0:
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)
        return gradinput1,gradinput2 ,gradinput3


class FlowFillholelayer(Function):
    def __init__(self):
        super(FlowFillholelayer,self).__init__()

    def forward(self, input1):
        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        # count = torch.zeros(input1.size(0),1,input1.size(2),input1.size(3)) # for accumulating the homography projections
        output = torch.zeros(input1.size())

        if input1.is_cuda :
            output = output.cuda()
            # count = count.cuda()
            err = my_lib.FlowFillholelayer_gpu_forward(input1, output)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            err = my_lib.FlowFillholelayer_cpu_forward(input1, output)
        if err != 0:
            print(err)
        # output = output/count # to divide the counter

        # self.count = count #to keep this
        # print(self.input1[0, 0, :10, :10])
        # print(self.count[0, 0, :10, :10])
        # print(self.input1[0, 0, -10:, -10:])
        # print(self.count[0, 0, -10:, -10:])

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    # def backward(self, gradoutput):
    #     # print("Backward of Filter Interpolation Layer")
    #     # gradinput1 = input1.new().zero_()
    #     # gradinput2 = input2.new().zero_()
    #     gradinput1 = torch.zeros(self.input1.size())
    #     if self.input1.is_cuda:
    #         # print("CUDA backward")
    #         gradinput1 = gradinput1.cuda(self.device)
    #         err = my_lib.WeightedFlowProjectionLayer_gpu_backward(self.input1, self.count, gradoutput, gradinput1)
    #         # print(err)
    #         if err != 0 :
    #             print(err)
    #
    #     else:
    #         # print("CPU backward")
    #         # print(gradoutput)
    #         err = my_lib.WeightedFlowProjectionLayer_cpu_backward(self.input1, self.count,  gradoutput, gradinput1)
    #         # print(err)
    #         if err != 0:
    #             print(err)
    #         # print(gradinput1)
    #         # print(gradinput2)
    #
    #     # print(gradinput1)
    #
    #     return gradinput1