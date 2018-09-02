from lib.model.spatial_translation import SpatialTranslationModel
from lib.model.temporal_predictor import TemporalPredictorModel
from lib.model.discriminator import Discriminator
from lib.buffer import ReplayBuffer
from lib.loss import GANLoss
from lib.utils import INFO

from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import itertools

class ReCycleGAN(nn.Module):
    def __init__(self, A_channel = 3, B_channel = 3, T = 30, t = 2, r = 1, device = 'cpu'):
        super().__init__()

        # Store variables
        self.t = t
        self.T = T + self.t
        self.device = device

        # Define network object
        self.G_A_to_B = SpatialTranslationModel(n_in = A_channel, n_out = B_channel, r = r)
        self.G_B_to_A = SpatialTranslationModel(n_in = B_channel, n_out = A_channel, r = r)
        self.P_A = TemporalPredictorModel(n_in = self.t * A_channel, n_out = A_channel, r = r)
        self.P_B = TemporalPredictorModel(n_in = self.t * B_channel, n_out = B_channel, r = r)
        self.D_A = Discriminator(n_in = A_channel, r = r)
        self.D_B = Discriminator(n_in = B_channel, r = r)

        # Define loss and optimizer
        self.criterion_adv = GANLoss()
        self.criterion_l2 = nn.MSELoss()
        self.optim_G = Adam(itertools.chain(self.G_A_to_B.parameters(), self.G_B_to_A.parameters(), self.P_A.parameters(), self.P_B.parameters()), lr = 0.0002)
        self.optim_D = Adam(itertools.chain(self.D_A.parameters(), self.D_B.parameters()), lr = 0.0001)

        # Define replay buffer (used in original CycleGAN)
        self.fake_a_buffer = ReplayBuffer()
        self.fake_b_buffer = ReplayBuffer()
        self.to(self.device)

    def setInput(self, true_a_seq, true_b_seq):
        self.true_a_seq = true_a_seq.float().to(self.device)
        self.true_b_seq = true_b_seq.float().to(self.device)

    def forward(self, true_a = None, true_b = None, true_a_seq = None, true_b_seq = None):
        fake_b_spat, fake_b_temp, fake_b, reco_a, fake_a_spat, fake_a_temp, fake_a, reco_b = None, None, None, None, None, None, None, None

        # ----------------------------------------------------------------------------
        # A -> B -> A
        # ----------------------------------------------------------------------------
        if true_a is not None and true_a_seq is not None:
            # Move to specific device
            true_a = true_a.to(self.device)
            true_a_seq = [frame.to(self.device) for frame in true_a_seq]

            # Get the tuple object before proceeding temporal predictor
            fake_b_tuple = []
            for i in range(self.t):
                fake_b_tuple.append(self.G_A_to_B(true_a_seq[i]))
            fake_b_tuple = torch.cat(fake_b_tuple, dim = 1)
            true_a_tuple = torch.cat(true_a_seq, dim = 1)
            true_a = true_a

            # Generate
            fake_b_spat = self.G_A_to_B(true_a)
            fake_b_temp = self.P_B(fake_b_tuple)
            fake_b = 0.5 * (fake_b_spat + fake_b_temp)
            reco_a = 0.5 * (self.G_B_to_A(self.P_B(fake_b_tuple)) + self.P_A(true_a_tuple))
        # ----------------------------------------------------------------------------
        # B -> A -> B
        # ----------------------------------------------------------------------------
        if true_b is not None and true_b_seq is not None:
            # Move to specific device
            true_b = true_b.to(self.device)
            true_b_seq = [frame.to(self.device) for frame in true_b_seq]

            # Get the tuple object before proceeding temporal predictor
            fake_a_tuple = []
            for i in range(self.t):
                fake_a_tuple.append(self.G_B_to_A(true_b_seq[i]))
            fake_a_tuple = torch.cat(fake_a_tuple, dim = 1)        
            true_b_tuple = torch.cat(true_b_seq, dim = 1)
            true_b = true_b

            # Generate
            fake_a_spat = self.G_B_to_A(true_b)
            fake_a_temp = self.P_A(fake_a_tuple)
            fake_a = 0.5 * (fake_a_spat + fake_a_temp)
            reco_b = 0.5 * (self.G_A_to_B(self.P_A(fake_a_tuple)) + self.P_B(true_b_tuple))
        return {
            'true_a': true_a, 'fake_b_spat': fake_b_spat, 'fake_b_temp': fake_b_temp, 'fake_b': fake_b, 'reco_a': reco_a, \
            'true_b': true_b,'fake_a_spat': fake_a_spat,'fake_a_temp': fake_a_temp,'fake_a': fake_a,'reco_b': reco_b
        }

    def updateGenerator(self, true_x_tuple, fake_y_tuple, true_x_next, fake_y_next, P_X, P_Y, net_Y_to_X, net_D):
        fake_x_next = P_X(true_x_tuple)
        reco_x_next = net_Y_to_X(P_Y(fake_y_tuple))
        fake_pred = net_D(fake_y_next)
        loss_G = self.criterion_adv(fake_pred, True) + self.criterion_l2(reco_x_next, true_x_next) * 10. + self.criterion_l2(fake_x_next, true_x_next) * 10.
        return loss_G                

    def updateDiscriminator(self, fake_frame, true_frame, net_D):
        fake_pred = net_D(fake_frame.detach())
        true_pred = net_D(true_frame)
        loss_D = self.criterion_adv(true_pred, True) + self.criterion_adv(fake_pred, False)
        return loss_D       

    def backward(self):
        # Prepare input sequence (BTCHW -> T * BCHW)
        true_a_frame_list = [frame.squeeze(1) for frame in torch.chunk(self.true_a_seq, self.T, dim = 1)]
        true_b_frame_list = [frame.squeeze(1) for frame in torch.chunk(self.true_b_seq, self.T, dim = 1)]
        self.loss_D = 0.0
        self.loss_G = 0.0

        # Generate fake_tuple in opposite domain
        fake_b_frame_list = []
        fake_a_frame_list = []
        for i in range(self.T):
            fake_b_frame_list.append(self.G_A_to_B(true_a_frame_list[i]))
            fake_a_frame_list.append(self.G_B_to_A(true_b_frame_list[i]))

        # ---------- Accumulate weight by each time step ----------
        for i in range(self.t, self.T):
            # generator and predictor
            true_a_tuple = torch.cat(true_a_frame_list[i - self.t: i], dim = 1).detach()
            fake_b_tuple = torch.cat(fake_b_frame_list[i - self.t: i], dim = 1).detach()
            true_b_tuple = torch.cat(true_b_frame_list[i - self.t: i], dim = 1).detach()
            fake_a_tuple = torch.cat(fake_a_frame_list[i - self.t: i], dim = 1).detach()
            self.loss_G += self.updateGenerator(true_a_tuple, fake_b_tuple, true_a_frame_list[i], true_b_frame_list[i], self.P_A, self.P_B, self.G_B_to_A, self.D_B)
            self.loss_G += self.updateGenerator(true_b_tuple, fake_a_tuple, true_b_frame_list[i], true_a_frame_list[i], self.P_B, self.P_A, self.G_A_to_B, self.D_A)
            # discriminator
            alter_fake_b = self.fake_b_buffer.push_and_pop(fake_b_frame_list[i])
            alter_fake_a = self.fake_a_buffer.push_and_pop(fake_a_frame_list[i])            
            self.loss_D += self.updateDiscriminator(alter_fake_b, true_b_frame_list[i], self.D_B)
            self.loss_D += self.updateDiscriminator(alter_fake_a, true_a_frame_list[i], self.D_A)
            
        # ---------- Update ----------
        # generator and predictor
        self.optim_G.zero_grad()
        self.loss_G.backward()
        self.optim_G.step()
        # discriminator
        self.optim_D.zero_grad()
        self.loss_D.backward()
        self.optim_D.step()