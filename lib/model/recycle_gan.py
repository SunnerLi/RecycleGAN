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

    def setInput(self, true_a_seq, true_b_seq, device = 'cpu'):
        """
            Set the input, and move to corresponding device
            * Notice: The stack object is the tensor, and rank format is TCHW (not tCHW)
        """
        self.true_a_seq = true_a_seq.float()
        self.true_b_seq = true_b_seq.float()
        self.true_a_seq = self.true_a_seq.to(device)
        self.true_b_seq = self.true_b_seq.to(device)

    def forward(self, warning = True):
        if warning:
            INFO("This function can be called during inference, you should call <backward> function to update the model!")

    def updateGenerator(self, true_x_tuple, fake_y_tuple, fake_pred, P_X, P_Y, net_G, net_D):
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
        # Get the tuple expect the last frame (future frame)
        true_x_tuple = torch.chunk(true_x_tuple, self.t, dim = 1)   # t * B1CHW
        true_x_tuple = [frame.squeeze(1) for frame in true_x_tuple] # t * BCHW
        true_x_next  = true_x_tuple[-1]
        true_x_tuple = torch.cat(true_x_tuple[:-1], dim = 1)

        # Form next frame by temporal prediction
        fake_x_next = P_X(true_x_tuple)
        reco_x_next = net_G(P_Y(fake_y_tuple))
        self.loss_G += (self.criterion_adv(fake_pred, True) + self.criterion_l2(fake_x_next, true_x_next) + self.criterion_l2(reco_x_next, true_x_next))

    def updateDiscriminator(self, true_x_tuple, true_y_tuple, net_G, net_D):
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
        # Split as frame list and initialize the parameters
        true_x_frame_list = torch.chunk(true_x_tuple, self.t, dim = 1) # t * B1CHW
        true_y_frame_list = torch.chunk(true_y_tuple, self.t, dim = 1) # t * B1CHW
        assert self.t == true_x_tuple.size(1)
        center_idx = true_x_tuple.size(1) // 2
        fake_y_stack = []

        # Generate prediction frame by frame
        for i in range(len(true_x_frame_list) - 1):
            true_x_frame = true_x_frame_list[i].squeeze(1)  # BCHW
            fake_y_stack.append(net_G(true_x_frame))

        # Accumulate adversarial loss
        true_y_frame = true_y_frame_list[center_idx].clone().squeeze(1)  # BCHW
        fake_y_frame = fake_y_stack[center_idx].clone()
        fake_pred = net_D(fake_y_frame)
        true_pred = net_D(true_y_frame)
        self.loss_D += self.criterion_adv(true_pred, True) + self.criterion_adv(fake_pred, False)

        # Return
        fake_y_stack = torch.cat(fake_y_stack, dim = 1)
        return fake_y_stack, fake_pred

    def backward(self):
        """
            The backward process of Re-cycle GAN
            Loss include:
                1. adversarial loss
                3. recurrent loss
                4. recycle loss
        """
        # TODO: Revise as rank=5 approach (hope can save memory usage)
        
        true_a_tuple_list = torch.chunk(self.true_a_seq, self.T, dim = 1) # 10 * [1, 1, 3, 720, 1080, 3]
        true_b_tuple_list = torch.chunk(self.true_b_seq, self.T, dim = 1)
        self.loss_D = 0
        self.loss_G = 0
        for true_a_tuple, true_b_tuple in zip(true_a_tuple_list, true_b_tuple_list):
            # Form the input tensor in single time step
            true_a_tuple = torch.squeeze(true_a_tuple, dim = 1) # [1, 3, 720, 1080, 3]
            true_b_tuple = torch.squeeze(true_b_tuple, dim = 1) # [1, 3, 720, 1080, 3]

            # Compute loss
            fake_b_stack, fake_pred = self.updateDiscriminator(true_a_tuple, true_b_tuple, self.G_A_to_B, self.D_B)
            self.updateGenerator(true_a_tuple, fake_b_stack, fake_pred, self.P_A, self.P_B, self.G_B_to_A, self.D_B)
            fake_a_stack, fake_pred = self.updateDiscriminator(true_b_tuple, true_a_tuple, self.G_B_to_A, self.D_A)
            self.updateGenerator(true_b_tuple, fake_a_stack, fake_pred, self.P_B, self.P_A, self.G_A_to_B, self.D_A)

        # Update discriminator
        self.optim_D.zero_grad()
        self.loss_D.backward(retain_graph = True)
        self.optim_D.step()

        # Update generator
        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()