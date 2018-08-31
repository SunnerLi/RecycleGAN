from lib.model.spatial_translation import SpatialTranslationModel
from lib.model.temporal_predictor import TemporalPredictorModel
from lib.model.discriminator import Discriminator
from lib.loss import GANLoss
from lib.utils import INFO

from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import itertools

"""
    This script define the ReCycleGAN with the approach whose rank=6
"""

class ReCycleGAN(nn.Module):
    def __init__(self, A_channel = 3, B_channel = 3, r = 1, t = 2, T = 30, device = 'cpu'):
        super().__init__()
        self.t = t
        self.T = T + math.ceil(t / 2)
        self.device = device
        self.loss_G = 0.0
        self.loss_D = 0.0
        self.G_A_to_B = SpatialTranslationModel(n_in = A_channel, n_out = B_channel, r = r)
        self.G_B_to_A = SpatialTranslationModel(n_in = B_channel, n_out = A_channel, r = r)
        self.P_A = TemporalPredictorModel(n_in = 2 * A_channel, n_out = A_channel, r = r)
        self.P_B = TemporalPredictorModel(n_in = 2 * B_channel, n_out = B_channel, r = r)
        self.D_A = Discriminator(n_in = A_channel, r = r)
        self.D_B = Discriminator(n_in = B_channel, r = r)
        self.criterion_adv = GANLoss()
        self.criterion_l2 = nn.MSELoss()
        self.optim_G = Adam(itertools.chain(self.G_A_to_B.parameters(), self.G_B_to_A.parameters(), self.P_A.parameters(), self.P_B.parameters()), lr = 0.001)
        self.optim_D = Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr = 0.001)
        self.to(self.device)

    def setInput(self, true_a_seq, true_b_seq):
        """
            Set the input, and move to corresponding device
            * Notice: The stack object is the tensor, and rank format is TCHW (not tCHW)
        """
        self.true_a_seq = true_a_seq.float()
        self.true_b_seq = true_b_seq.float()
        self.true_a_seq = self.true_a_seq.to(self.device)
        self.true_b_seq = self.true_b_seq.to(self.device)

    def forward(self, true_a = None, true_b = None, true_a_seq = None, true_b_seq = None, warning = True):
        """
            The usual forward process of the Re-cycleGAN
            * You should notice that the tensor should move to device previously!!!!!
        """
        # Warn the user not to call this function during training
        if warning:
            INFO("This function can be called during inference, you should call <backward> function to update the model!")

        # Get the tuple object before proceeding temporal predictor
        fake_a_tuple = []
        fake_b_tuple = []
        for i in range(self.t - 1):
            fake_a_tuple.append(self.G_B_to_A(true_b_seq[i]))
            fake_b_tuple.append(self.G_A_to_B(true_a_seq[i]))
        fake_a_tuple = torch.cat(fake_a_tuple, dim = 1)
        fake_b_tuple = torch.cat(fake_b_tuple, dim = 1)
        true_a_tuple = torch.cat(true_a_seq, dim = 1)
        true_b_tuple = torch.cat(true_b_seq, dim = 1)
        true_a = true_a
        true_b = true_b

        # Generate
        fake_b = 0.5 * (self.G_A_to_B(true_a) + self.P_B(fake_b_tuple))
        reco_a = 0.5 * (self.G_B_to_A(self.P_B(fake_b_tuple)) + self.P_A(true_a_tuple))
        fake_a = 0.5 * (self.G_B_to_A(true_b) + self.P_A(fake_a_tuple))
        reco_b = 0.5 * (self.G_A_to_B(self.P_A(fake_a_tuple)) + self.P_B(true_b_tuple))
        return {
            'true_a': true_a,
            'fake_b': fake_b,
            'reco_a': reco_a,
            'true_b': true_b,
            'fake_a': fake_a,
            'reco_b': reco_b
        }
        
    # def validate(self):
    #     """
    #         Render for the first image

    #         You should notice the index!
    #         For example, if self.t is 3, then it represent that we consider the front 2 image, and predict for the 3rd frame
    #         So the tuple index is 0~2 which means tuple[:self.t - 1]
    #         Also, the 3rd frame index is 2 which also means tuple[self.t - 1]
    #     """
    #     # BTCHW -> T * BCHW
    #     true_a_seq = [frame.squeeze(1) for frame in torch.chunk(self.true_a_seq, self.true_a_seq.size(1), dim = 1)]
    #     true_b_seq = [frame.squeeze(1) for frame in torch.chunk(self.true_b_seq, self.true_b_seq.size(1), dim = 1)]
        
    #     # Form the input frame in original domain
    #     true_a = true_a_seq[self.t - 1]
    #     true_b = true_b_seq[self.t - 1]

    #     # Form the tuple to generate the image in opposite domain
    #     true_a_tuple = torch.cat(true_a_seq[:self.t - 1], dim = 1)
    #     true_b_tuple = torch.cat(true_b_seq[:self.t - 1], dim = 1)
    #     return self.forward(true_a, true_b, true_a_tuple, true_b_tuple)

    def updateGenerator(self, true_x_tuple, fake_y_tuple, true_x_next, fake_pred, P_X, P_Y, net_G, net_D):
        """
            Update the generator for the given tuples in both domain
            -----------------------------------------------------------------------------------
            You should update the discriminator and obtain the fake prediction first
            -----------------------------------------------------------------------------------

            Arg:    true_x_tuple    - The tuple tensor object in X domain, rank is BtCHW
                    fake_y_tuple    - The generated tuple tensor object in Y domain, rank is B(t-1)CHW
                    fake_pred       - The fake prediction of last frame
                    P_X             - The temporal predictor of X domain
                    P_Y             - The temporal predictor of Y domain
                    net_G           - The generator (Y -> X)
                    net_D           - The discriminator (Y)
            Ret:    The generator loss
        """
        fake_x_next = P_X(true_x_tuple)
        reco_x_next = net_G(P_Y(fake_y_tuple))
        self.loss_G += (self.criterion_adv(fake_pred, True) + 
            10 * self.criterion_l2(fake_x_next, true_x_next) + 
            10 * self.criterion_l2(reco_x_next, true_x_next)
        )

    def updateDiscriminator(self, fake_frame, true_frame, net_D):
        """
            Update the discriminator loss for the given input tuple
            -----------------------------------------------------------------------------------
            You should notice that the discriminator loss will only consider the last frame
            This mechanism can avoid accumulate the adv loss for duplicated times
            -----------------------------------------------------------------------------------

            Arg:    true_x_tuple    - The tuple tensor object in X domain, rank is BtCHW
                    true_y_tuple    - The tuple tensor object in Y domain, rank is BtCHW
                    net_G           - The generator (X -> Y)
                    net_D           - The discriminator (Y)
                    true_y          - The last tensor object in Y domain, rank is BtCHW
            Ret:    1. The generated tuple tensor in Y domain, rank is B(t-1)CHW
                    2. The fake prediction of last frame
                    3. Revised discriminator loss
        """
        fake_pred = net_D(fake_frame)
        true_pred = net_D(true_frame)
        self.loss_D += self.criterion_adv(true_pred, True) + self.criterion_adv(fake_pred, False)
        return fake_pred

    def backward(self):
        """
            The backward process of Re-cycle GAN
            Loss include:
                1. adversarial loss
                3. recurrent loss
                4. recycle loss
        """       
        true_a_frame_list = [frame.squeeze(1) for frame in torch.chunk(self.true_a_seq, self.T, dim = 1)] # 10 * [1, 3, 720, 1080]
        true_b_frame_list = [frame.squeeze(1) for frame in torch.chunk(self.true_b_seq, self.T, dim = 1)]
        self.loss_D = 0
        self.loss_G = 0

        # Generate fake_tuple in opposite domain
        fake_b_frame_list = []
        fake_a_frame_list = []
        for i in range(self.T):
            fake_b_frame_list.append(self.G_A_to_B(true_a_frame_list[i]))
            fake_a_frame_list.append(self.G_B_to_A(true_b_frame_list[i]))

        for i in range(self.T):
            fake_b_pred = self.updateDiscriminator(fake_b_frame_list[i], true_a_frame_list[i], self.D_B)
            fake_a_pred = self.updateDiscriminator(fake_a_frame_list[i], true_b_frame_list[i], self.D_A)
            if i < self.T - self.t + 1:
                true_a_tuple = torch.cat(true_a_frame_list[i : i + self.t - 1], dim = 1)
                true_b_tuple = torch.cat(true_b_frame_list[i : i + self.t - 1], dim = 1)
                fake_a_tuple = torch.cat(fake_a_frame_list[i : i + self.t - 1], dim = 1)
                fake_b_tuple = torch.cat(fake_b_frame_list[i : i + self.t - 1], dim = 1)
                self.updateGenerator(true_a_tuple, fake_b_tuple, true_a_frame_list[i + self.t - 1], fake_b_pred, self.P_A, self.P_B, self.G_B_to_A, self.D_B)
                self.updateGenerator(true_b_tuple, fake_a_tuple, true_b_frame_list[i + self.t - 1], fake_a_pred, self.P_B, self.P_A, self.G_A_to_B, self.D_A)           

        # Update discriminator
        self.optim_D.zero_grad()
        self.loss_D.backward(retain_graph = True)
        self.optim_D.step()

        # Update generator
        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()