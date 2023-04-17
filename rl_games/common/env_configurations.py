from rl_games.common import tr_helpers
import gym
from gym.wrappers import FlattenObservation, FilterObservation
import numpy as np
import math


configurations = {
}

def register(name, config):
    configurations[name] = config