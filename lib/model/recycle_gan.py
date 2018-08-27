from lib.model.spatial_translation import SpatialTranslationModel
from lib.model.temporal_predictor import TemporalPredictorModel
from lib.model.discriminator import Discriminator
from lib.loss import GANLoss
from lib.utils import INFO

from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch

import itertools

class ReCycleGAN(nn.Module):
    def __init__(self, A_channel = 3, B_channel = 3, r = 1, t = 2, T = 30):
        super().__init__()
        self.t = t
        self.T = T
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

    def setInput(self, true_a_seq, true_b_seq, device = 'cpu'):
        """
            Set the input, and move to corresponding device
            * Notice: The stack object is the tensor, and rank format is TCHW (not tCHW)
        """
        self.true_a_seq = true_a_seq
        self.true_b_seq = true_b_seq
        self.true_a_seq = self.true_a_seq.to(device)
        self.true_b_seq = self.true_b_seq.to(device)

    def forward(self, warning = True):
        if warning:
            INFO("This function can be called during inference, you should call <backward> function to update the model!")

    def updateGenerator(self, true_x_stack, fake_y_stack, P_X, P_Y, net_G, net_D, true_x_next, fake_pred, loss_G = 0.0):
        """
            Update the generator for the given tuples in both domain
            -----------------------------------------------------------------------------------
            You should update the discriminator and obtain the fake prediction first
            -----------------------------------------------------------------------------------

            Arg:    true_x_stack    - The tuple tensor object in X domain, rank is tCHW
                    fake_y_stack    - The generated tuple tensor object in Y domain, rank is TCHW
                    P_X             - The temporal predictor of X domain
                    P_Y             - The temporal predictor of Y domain
                    net_G           - The generator (Y -> X)
                    net_D           - The discriminator (Y)
                    true_x_next     - The future frame in X domain, rank is 1CHW
                    fake_pred       - The fake prediction of last frame
                    loss_G          - The generator loss, default is 0.0
            Ret:    The generator loss
        """
        fake_x_next = P_X(true_x_stack)
        reco_x_next = net_G(P_Y(fake_y_stack))
        loss_G += (self.criterion_adv(fake_pred, True) + self.criterion_l2(fake_x_next, true_x_next) + self.criterion_l2(reco_x_next, true_x_next))
        return loss_G

    def updateDiscriminator(self, true_x_stack, net_G, net_D, true_y, loss_D = 0.0):
        """
            Update the discriminator loss for the given input tuple
            -----------------------------------------------------------------------------------
            You should notice that the discriminator loss will only consider the last frame
            This mechanism can avoid accumulate the adv loss for duplicated times
            -----------------------------------------------------------------------------------

            Arg:    true_x_stack    - The tuple tensor object in X domain, rank is tCHW
                    net_G           - The generator (X -> Y)
                    net_D           - The discriminator (Y)
                    true_y          - The last tensor object in Y domain, rank is 1CHW
                    loss_D          - The discriminator loss, default is 0.0
            Ret:    1. The generated tuple tensor in Y domain, rank is TCHW
                    2. The fake prediction of last frame
                    3. Revised discriminator loss
        """
        fake_y_stack = net_G(true_x_stack)
        fake_pred = net_D(fake_y_stack[-1].clone())
        true_pred = net_D(true_x_stack[-1].clone())
        loss_D += self.criterion_adv(true_pred, True) + self.criterion_adv(fake_pred, False)
        return fake_y_stack, fake_pred, loss_D

    def backward(self):
        """
            The backward process of Re-cycle GAN
            Loss include:
                1. adversarial loss
                3. recurrent loss
                4. recycle loss
        """
        # TODO: revise as rank=6 version updating
        
        true_a_list = torch.chunk(self.true_a_seq, 3, dim = 0)
        true_b_list = torch.chunk(self.true_b_seq, 3, dim = 0)
        loss_D = 0
        loss_G = 0

        # Accumulate for ((T-t)/t+1) step
        for i in range(self.t, self.T, 1):
            true_a_stack = torch.cat(true_a_list[i-self.t : i], dim = 0)
            true_b_stack = torch.cat(true_b_list[i-self.t : i], dim = 0)
            fake_b_stack, fake_pred, loss_D = self.updateDiscriminator(true_a_stack, self.G_A_to_B, self.D_B, true_b_list[i-1], loss_D)
            loss_G = self.updateGenerator(true_a_stack, fake_b_stack, self.P_A, self.P_B, self.G_B_to_A, self.D_B, true_a_list[i], fake_pred, loss_G)
            fake_a_stack, fake_pred, loss_D = self.updateDiscriminator(true_b_stack, self.G_B_to_A, self.D_A, true_a_list[i-1], loss_D)
            loss_G = self.updateGenerator(true_b_stack, fake_a_stack, self.P_B, self.P_A, self.G_A_to_B, self.D_A, true_b_list[i], fake_pred, loss_G)            

        # Update discriminator
        self.optim_D.zero_grad()
        loss_D.backward()
        self.optim_D.step()

        # Update generator
        self.optim_G.zero_grad()
        loss_G.backward()
        self.optim_G.step()