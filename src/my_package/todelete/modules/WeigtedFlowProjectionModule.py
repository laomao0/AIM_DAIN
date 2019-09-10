# modules/FlowProjectionModule.py
from torch.nn import Module
from functions.WeightedFlowProjectionLayer import WeightedFlowProjectionLayer

class WeightedFlowProjectionModule(Module):
    def __init__(self, threshold = 20.0/255.0, requires_grad = True):
        super(WeightedFlowProjectionModule, self).__init__()

        self.threshold = threshold
        self.requires_grad= requires_grad
        # self.f = WeightedFlowProjectionLayer(threshold,requires_grad)

    def forward(self, input1,input2,input3 ):
        return  WeightedFlowProjectionLayer.apply(input1,input2,input3, self.threshold,self.requires_grad)

# class FlowFillholeModule(Module):
#     def __init__(self,hole_value = -10000.0):
#         super(FlowFillholeModule, self).__init__()
#         self.f = FlowFillholeLayer()
#
#     def forward(self, input1):
#         return self.f(input1)

    #we actually dont need to write the backward code for a module, since we have

