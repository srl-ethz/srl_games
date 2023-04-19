from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np


def _create_initializer(func, **kwargs):
    return lambda v : func(v, **kwargs)

class NetworkBuilder:
    def __init__(self, **kwargs):
        pass


    class BaseNetwork(nn.Module):
        def __init__(self, **kwargs):
            nn.Module.__init__(self, **kwargs)

            self.activations_factory = object_factory.ObjectFactory()
            self.activations_factory.register_builder('relu', lambda **kwargs : nn.ReLU(**kwargs))
            self.activations_factory.register_builder('tanh', lambda **kwargs : nn.Tanh(**kwargs))
            self.activations_factory.register_builder('sigmoid', lambda **kwargs : nn.Sigmoid(**kwargs))
            self.activations_factory.register_builder('elu', lambda  **kwargs : nn.ELU(**kwargs))
            self.activations_factory.register_builder('selu', lambda **kwargs : nn.SELU(**kwargs))
            self.activations_factory.register_builder('swish', lambda **kwargs : nn.SiLU(**kwargs))
            self.activations_factory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
            self.activations_factory.register_builder('softplus', lambda **kwargs : nn.Softplus(**kwargs))
            self.activations_factory.register_builder('None', lambda **kwargs : nn.Identity())



        def _build_sequential_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func,
        norm_only_first_layer=False, 
        norm_func_name = None):
            print('build mlp:', input_size)
            in_size = input_size
            layers = []
            need_norm = True
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))

                if not need_norm:
                    continue
                if norm_only_first_layer and norm_func_name is not None:
                   need_norm = False 
                assert not norm_func_name in ['layer_norm', 'batch_norm']
                in_size = unit

            return nn.Sequential(*layers)

        def _build_mlp(self, 
        input_size, 
        units, 
        activation,
        dense_func, 
        norm_only_first_layer=False,
        norm_func_name = None,
        d2rl=False):
            assert not d2rl
            return self._build_sequential_mlp(input_size, units, activation, dense_func, norm_func_name = None,)

class A2CBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            
            assert not self.central_value

            mlp_input_shape = input_shape[0]

            in_mlp_shape = mlp_input_shape
            out_size = self.units[-1]


            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)

            '''
                for multidiscrete actions num is a tuple
            '''
            assert not self.is_multi_discrete
            assert self.is_continuous
            self.mu = torch.nn.Linear(out_size, actions_num)
            sigma_init_val = self.space_config['sigma_init']['val']

            self.sigma = nn.Parameter(sigma_init_val * torch.ones(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            print(f"{self.actor_mlp=}")
            print(f"{self.value=}")
            print(f"{self.mu=}")
            print(f"{self.sigma=}")

            for m in self.modules():         
                if isinstance(m, nn.Linear):
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                        print(f"set bias to zero {m}")


        def forward(self, obs_dict):
            obs = obs_dict['obs']
            obs = obs.flatten(1)                

            out = self.actor_mlp(out)
            value = self.value(out)

            mu = self.mu(out)
            return mu, mu*0 + self.sigma, value
                    

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete'in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']

    def build(self, name, **kwargs):
        net = A2CBuilder.Network(self.params, **kwargs)
        return net
