import torch
import copy
from torch.utils.data import Dataset

class PPODataset(Dataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, device, seq_len):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        total_games = self.batch_size // self.seq_len
        self.num_games_batch = self.minibatch_size // self.seq_len
        self.game_indexes = torch.arange(total_games, dtype=torch.long, device=self.device)
        self.flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device=self.device).reshape(total_games, self.seq_len)


    def update_values_dict(self, values_dict):
        self.values_dict = values_dict     

    def update_mu_sigma(self, mu, sigma):	    
        start = self.last_range[0]	           
        end = self.last_range[1]	
        self.values_dict['mu'][start:end] = mu	
        self.values_dict['sigma'][start:end] = sigma 

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.values_dict.items():
            if v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]
                
        return input_dict
