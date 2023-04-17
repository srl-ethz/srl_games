import ray
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.env_configurations import configurations
import numpy as np
import gym
import random
from time import sleep
import torch


vecenv_config = {}

def register(config_name, func):
    vecenv_config[config_name] = func

def create_vec_env(config_name, num_actors, **kwargs):
    vec_env_name = configurations[config_name]['vecenv_type']
    return vecenv_config[vec_env_name](config_name, num_actors, **kwargs)

