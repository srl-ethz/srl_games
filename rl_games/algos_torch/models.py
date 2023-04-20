import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class BaseModel():
    def __init__(self, model_class):
        self.model_class = model_class


    def build(self, config):
        obs_shape = config['input_shape']
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        return self.Network(self.network_builder.build(self.model_class, **config), obs_shape=obs_shape,
            normalize_value=normalize_value, normalize_input=normalize_input)

class BaseModelNetwork(nn.Module):
    def __init__(self, obs_shape, normalize_value, normalize_input):
        nn.Module.__init__(self)
        self.obs_shape = obs_shape
        self.normalize_value = normalize_value
        self.normalize_input = normalize_input

        if normalize_value:
            self.value_mean_std = RunningMeanStd((1,))
        if normalize_input:
            assert not isinstance(obs_shape, dict), "function removed in rl_games_simplified"
            self.running_mean_std = RunningMeanStd(obs_shape)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.running_mean_std(observation) if self.normalize_input else observation

    def unnorm_value(self, value):
        with torch.no_grad():
            return self.value_mean_std(value, unnorm=True) if self.normalize_value else value



class ModelA2CContinuousLogStd(BaseModel):
    def __init__(self, network):
        BaseModel.__init__(self, 'a2c')
        self.network_builder = network

    class Network(BaseModelNetwork):
        def __init__(self, a2c_network, **kwargs):
            BaseModelNetwork.__init__(self, **kwargs)
            self.a2c_network = a2c_network

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = - distr.log_prob(prev_actions).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = - distr.log_prob(selected_action).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
