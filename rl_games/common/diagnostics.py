import torch
import rl_games.algos_torch.torch_ext as torch_ext

class DefaultDiagnostics(object):
    def __init__(self):
        pass
    def send_info(self, writter):
        pass    
    def epoch(self, agent, current_epoch):
        pass
    def mini_epoch(self, agent, miniepoch):
        pass
    def mini_batch(self, agent, batch, e_clip, minibatch):
        pass

