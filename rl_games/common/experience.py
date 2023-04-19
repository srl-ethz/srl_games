import numpy as np
import random
import gym
import torch

class ExperienceBuffer:
    '''
    More generalized than replay buffers.
    Implemented for on-policy algos
    '''
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']
        self.obs_space = env_info['observation_space']
        
        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        batch_size = self.num_actors * self.num_agents
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        assert type(self.action_space) is gym.spaces.Box
        self.actions_shape = (self.action_space.shape[0],) 
        self.actions_num = self.action_space.shape[0]
        self.obs_shape = (self.obs_space.shape[0],)
        self.tensor_dict = {}
        self._init_from_env_info()

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self):

        self.tensor_dict['obses'] = self._create_tensor_from_space(self.obs_shape)
        
        val_shape = (1,)
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(())
        self.tensor_dict['dones'] = self._create_tensor_from_space((), torch.int8)

        self.tensor_dict['actions'] = self._create_tensor_from_space(self.actions_shape)
        self.tensor_dict['mus'] = self._create_tensor_from_space(self.actions_shape)
        self.tensor_dict['sigmas'] = self._create_tensor_from_space(self.actions_shape)

    def _create_tensor_from_space(self, shape, dtype=torch.float32):
        return torch.zeros(self.obs_base_shape + shape, dtype= dtype, device = self.device)


    def update_data(self, name, index, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_dict[name][k][index,:] = v
        else:
            self.tensor_dict[name][index,:] = val




    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        
        return res_dict
