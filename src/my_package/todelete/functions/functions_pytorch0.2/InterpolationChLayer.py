# this is for wrapping the customized layer
import torch
from torch.autograd import Function
import _ext.my_lib as my_lib

#Please check how the STN FUNCTION is written :
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/gridgen.py
#https://github.com/fxia22/stn.pytorch/blob/master/script/functions/stn.py

class InterpolationChLayer(Function):
    def __init__(self,ch):
        super(InterpolationChLayer,self).__init__()
        self.ch = ch

    def forward(self, input1,input2):

        # assert(input1.is_contiguous())
        # assert(input2.is_contiguous())
        self.input1 = input1.contiguous() # need to use in the backward process, so we need to cache it
        self.input2 = input2.contiguous() # TODO: Note that this is simply a shallow copy?
        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        # output =  torch.zeros(input1.size())

        if input1.is_cuda :
            # output = output.cuda()
            output = torch.cuda.FloatTensor().resize_(self.input1.size()).zero_()
            my_lib.InterpolationChLayer_gpu_forward(input1, input2, output)
        else:
            # output = torch.cuda.FloatTensor(input1.data.size())
            output = torch.FloatTensor().resize_(self.input1.size()).zero_()
            my_lib.InterpolationChLayer_cpu_forward(input1, input2, output)

        # the function returns the output to its caller
        return output

    #TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        # print("Backward of Interpolation Layer")
        # gradinput1 = input1.new().zero_()
        # gradinput2 = input2.new().zero_()
        # gradinput1 = torch.zeros(self.input1.size())
        # gradinput2 = torch.zeros(self.input2.size())
        if self.input1.is_cuda:
            # print("CUDA backward")
            # gradinput1 = gradinput1.cuda(self.device)
            # gradinput2 = gradinput2.cuda(self.device)
            gradinput1 = torch.cuda.FloatTensor().resize_(self.input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(self.input2.size()).zero_()
            # the input1 image should not require any gradients
            # print("Does input1 requires gradients? " + str(self.input1.requires_grad))

            err = my_lib.InterpolationChLayer_gpu_backward(self.input1,self.input2,gradoutput,gradinput1,gradinput2)
            if err != 0 :
                print(err)

        else:
            # print("CPU backward")
            # print(gradoutput)
            gradinput1 = torch.FloatTensor().resize_(self.input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(self.input2.size()).zero_()

            err = my_lib.InterpolationChLayer_cpu_backward(self.input1, self.input2, gradoutput, gradinput1, gradinput2)
            # print(err)
            if err != 0 :
                print(err)
            # print(gradinput1)
            # print(gradinput2)

        # print(gradinput1)

        return gradinput1, gradinput2