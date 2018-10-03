import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    This script define the structure of discriminator
    According to the original Re-cycle GAN paper, 
    the structure of discriminator is 70x70 PatchGAN
    And also it is also used in original CycleGAN official implementation
    The code is revised from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
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
        # ================================================================================================================
        # Notice:   The result in the readme is gotten by taking the average pooling
        #           However, the original design of PatchGAN doesn't do the pooling
        #           I refer the code might not be correct in the original reference URL
        #
        #           We change the URL as the official one, and remove this operation
        #           The result might become worse
        #           If you want to re-produce the result which is shown in the readme, just un-comment (1) code and comment (2)
        # ================================================================================================================
        x =  self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)        # (1)
        # return x                                                            # (2)