from rl_games.algos_torch import torch_ext
import torch
import numpy as np


class AlgoObserver:
    def __init__(self):
        pass

    def before_init(self, base_name, config, experiment_name):
        pass

    def after_init(self, algo):
        pass

    def process_infos(self, infos, done_indices):
        pass

    def after_steps(self):
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        pass
