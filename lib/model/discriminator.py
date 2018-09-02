import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    This script define the structure of discriminator
    According to the original Re-cycle GAN paper, 
    the structure of discriminator is 70x70 PatchGAN
    And also it is also used in original CycleGAN official implementation
    Thus we borrow the implementation from: https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/models.py
"""

class Discriminator(nn.Module):
    def __init__(self, n_in = 3, r = 1):
        super().__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(n_in, 64 // r, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64 // r, 128 // r, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128 // r), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128 // r, 256 // r, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256 // r), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256 // r, 512 // r, 4, padding=1),
                    nn.InstanceNorm2d(512 // r), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512 // r, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)