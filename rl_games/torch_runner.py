import os
import time
import numpy as np
import random
from copy import deepcopy
import torch
#import yaml

#from rl_games import envs
from rl_games.common import object_factory

from rl_games.algos_torch import a2c_continuous

class Runner:
    def __init__(self, algo_observer=None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs : a2c_continuous.A2CAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()    

        self.algo_observer = algo_observer
        torch.backends.cudnn.benchmark = True
        ### it didnot help for lots for openai gym envs anyway :(
        #torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)

    def load_config(self, params):
        self.seed = params.get('seed', None)
        if self.seed is None:
            self.seed = int(time.time())

        if params["config"].get('multi_gpu', False):
            self.seed += int(os.getenv("LOCAL_RANK", "0"))
        print(f"self.seed = {self.seed}")

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        config = params['config']
        reward_shaper_config = config['reward_shaper']
        reward_shaper_errormsg = "srl_games does not support reward_shaper to avoid confusion, please adjust the reward scales within the environment config"
        assert reward_shaper_config.get('scale_value', 1) == 1, reward_shaper_errormsg
        assert reward_shaper_config.get('shift_value', 0) == 0, reward_shaper_errormsg
        if 'features' not in config:
            config['features'] = {}
        config['features']['observer'] = self.algo_observer
        self.params = params

    def load(self, yaml_config):
        config = deepcopy(yaml_config)
        self.default_config = deepcopy(config['params'])
        self.load_config(params=self.default_config)

    def run_train(self, args):
        print('Started to train')
        agent = self.algo_factory.create(self.algo_name, base_name='run', params=self.params)
        agent.train()


    def reset(self):
        pass

    def run(self, args):
        load_path = None

        assert args['train']
        self.run_train(args)