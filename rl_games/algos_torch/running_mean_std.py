from rl_games.algos_torch import torch_ext
import torch
import torch.nn as nn
import numpy as np
'''
updates statistic from a full data
'''
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        assert not norm_only
        assert not per_channel
        self.axis = [0]
        in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False, mask=None):
        if self.training:
            assert not unnorm
            assert mask is None
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

        # change shape
        current_mean = self.running_mean
        current_var = self.running_var
        # get output
        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
            y = torch.clamp(y, min=-5.0, max=5.0)
        return y