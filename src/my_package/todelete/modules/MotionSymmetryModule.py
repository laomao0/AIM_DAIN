# modules/InterpolationLayer.py
from torch.nn import Module
from functions.MotionSymmetryLayer import MotionSymmetryLayer

class MotionSymmetryModule(Module):
    def __init__(self):
        super(MotionSymmetryModule, self).__init__()
        self.f = MotionSymmetryLayer()

    def forward(self, input1, input2):
        return self.f(input1, input2)

    #we actually dont need to write the backward code for a module, since we have

