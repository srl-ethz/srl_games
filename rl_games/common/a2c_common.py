import copy
import os

from rl_games.common import vecenv

from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.algos_torch import  model_builder
from rl_games.interfaces.base_algorithm import  BaseAlgorithm
import numpy as np
import time

from datetime import datetime
from tensorboardX import SummaryWriter
import torch 
from torch import nn
import torch.distributed as dist
 
from time import sleep

from rl_games.common import common_losses


def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class A2CBase(BaseAlgorithm):

    def __init__(self, base_name, params):

        self.config = config = params['config']
        full_experiment_name = config.get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = config['name'] + datetime.now().strftime("_%d-%H-%M-%S")

        self.config = config
        self.algo_observer = config['features']['observer']
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.load_networks(params)
        self.multi_gpu = config.get('multi_gpu', False)
        self.rank = 0
        self.rank_size = 1
        self.curr_frames = 0


        self.network_path = config.get('network_path', "./nn/")
        self.log_path = config.get('log_path', "runs/")
        self.env_config = config.get('env_config', {})
        self.num_actors = config['num_actors']
        self.env_name = config['env_name']

        self.vec_env = None
        self.env_info = config.get('env_info')
        if self.env_info is None:
            self.vec_env = vecenv.create_vec_env(self.env_name, self.num_actors, **self.env_config)
            self.env_info = self.vec_env.get_env_info()
        else:
            self.vec_env = config.get('vec_env', None)

        self.ppo_device = config.get('device', 'cuda:0')
        self.value_size = self.env_info.get('value_size',1)
        self.observation_space = self.env_info['observation_space']
        self.weight_decay = config.get('weight_decay', 0.0)
        self.is_train = config.get('is_train', True)

        self.central_value_config = self.config.get('central_value_config', None)
        self.has_central_value = self.central_value_config is not None
        self.truncate_grads = self.config.get('truncate_grads', False)


        self.self_play_config = self.config.get('self_play_config', None)
        self.has_self_play_config = self.self_play_config is not None

        self.self_play = config.get('self_play', False)
        self.save_freq = config.get('save_frequency', 0)
        self.save_best_after = config.get('save_best_after', 100)
        self.print_stats = config.get('print_stats', True)
        self.name = base_name

        self.ppo = config.get('ppo', True)
        self.max_epochs = self.config.get('max_epochs', -1)
        self.max_frames = self.config.get('max_frames', -1)

        self.is_adaptive_lr = config['lr_schedule'] == 'adaptive'
        self.linear_lr = config['lr_schedule'] == 'linear'
        self.schedule_type = config.get('schedule_type', 'legacy')

        # Setting learning rate scheduler
        if self.is_adaptive_lr:
            self.kl_threshold = config['kl_threshold']
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.e_clip = config['e_clip']
        self.clip_value = config['clip_value']
        self.network = config['network']
        self.num_agents = self.env_info.get('agents', 1)
        self.horizon_length = config['horizon_length']
        self.seq_len = self.config.get('seq_length', 4)
        self.bptt_len = self.config.get('bptt_length', self.seq_len) # not used right now. Didn't show that it is usefull
        self.normalize_advantage = config['normalize_advantage']
        self.normalize_rms_advantage = config.get('normalize_rms_advantage', False)
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.truncate_grads = self.config.get('truncate_grads', False)

        self.obs_shape = self.observation_space.shape
 
        self.critic_coef = config['critic_coef']
        self.grad_norm = config['grad_norm']
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']

        self.games_to_track = self.config.get('games_to_track', 100)
        print('current training device:', self.ppo_device)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.obs = None
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        assert(('minibatch_size_per_env' in self.config) or ('minibatch_size' in self.config))
        self.minibatch_size_per_env = self.config.get('minibatch_size_per_env', 0)
        self.minibatch_size = self.config.get('minibatch_size', self.num_actors * self.minibatch_size_per_env)
        self.mini_epochs_num = self.config['mini_epochs']
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert(self.batch_size % self.minibatch_size == 0)


        self.last_lr = self.config['learning_rate']
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -100500
        self.play_time = 0
        self.epoch_num = 0
        self.curr_frames = 0
        # allows us to specify a folder where all experiments will reside
        self.train_dir = config.get('train_dir', 'runs')

        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)

        # folders inside <train_dir>/<experiment_dir> for a specific purpose
        self.nn_dir = os.path.join(self.experiment_dir, 'nn')
        self.summaries_dir = os.path.join(self.experiment_dir, 'summaries')

        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)

        self.entropy_coef = self.config['entropy_coef']

        if self.rank == 0:
            writer = SummaryWriter(self.summaries_dir)
            self.writer = writer
        else:
            self.writer = None

        self.value_bootstrap = self.config.get('value_bootstrap')

        self.last_state_indices = None

        # features
        self.algo_observer = config['features']['observer']


    def load_networks(self, params):
        builder = model_builder.ModelBuilder()
        self.config['network'] = builder.load(params)


    def write_stats(self, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.writer.add_scalar('performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar('performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar('performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar('performance/rl_update_time', update_time, frame)
        self.writer.add_scalar('performance/step_inference_time', play_time, frame)
        self.writer.add_scalar('performance/step_time', step_time, frame)
        self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
                
        self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar('info/last_lr', last_lr, frame)
        self.writer.add_scalar('info/e_clip', self.e_clip, frame)
        self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar('info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def update_lr(self, lr):

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        



    def get_action_values(self, obs):

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs['obs'],
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            self.model.eval()

            input_dict = {
                'is_train': False,
                'prev_actions': None, 
                'obs' : obs['obs'],
            }
            result = self.model(input_dict)
            value = result['values']
            return value

    @property
    def device(self):
        return self.ppo_device

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

    def env_step(self, actions):
        actions = torch.clamp(actions, -1.0, 1.0)
        obs, rewards, dones, infos = self.vec_env.step(actions)

        assert self.value_size == 1
        rewards = rewards.unsqueeze(1)
        return obs, rewards.to(self.ppo_device), dones.to(self.ppo_device), infos

    def env_reset(self):
        obs = self.vec_env.reset()
        return obs

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t+1]
                nextvalues = mb_extrinsic_values[t+1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def update_epoch(self):
        pass

    def train(self):
        pass

    def prepare_dataset(self, batch_dict):
        pass

    def train_epoch(self):
        self.vec_env.set_train_info(self.frame, self)

    def train_actor_critic(self, obs_dict, opt_step=True):
        pass

    def calc_gradients(self):
        pass

    def get_central_value(self, obs_dict):
        return self.central_value_net.get_value(obs_dict)

    def train_central_value(self):
        return self.central_value_net.train_net()

    def get_full_state_weights(self):
        state = self.get_weights()
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        if self.has_central_value:
            state['assymetric_vf_nets'] = self.central_value_net.state_dict()
        state['frame'] = self.frame

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state['last_mean_rewards'] = self.last_mean_rewards

        if self.vec_env is not None:
            env_state = self.vec_env.get_env_state()
            state['env_state'] = env_state

        return state

    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch'] # frames as well?
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False):
        state = {}
        if self.has_central_value:
            state['central_val_stats'] = self.central_value_net.get_stats_weights(model_stats)
        if model_stats:
            if self.normalize_input:
                state['running_mean_std'] = self.model.running_mean_std.state_dict()
            if self.normalize_value:
                state['reward_mean_std'] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights):
        if self.normalize_rms_advantage:
            self.advantage_mean_std.load_state_dic(weights['advantage_mean_std'])
        if self.normalize_input and 'running_mean_std' in weights:
            self.model.running_mean_std.load_state_dict(weights['running_mean_std'])
        if self.normalize_value and 'normalize_value' in weights:
            self.model.value_mean_std.load_state_dict(weights['reward_mean_std'])

    def set_weights(self, weights):
        self.model.load_state_dict(weights['model'])
        self.set_stats_weights(weights)

    def play_steps(self):
        step_time = 0.0

        for n in range(self.horizon_length):
            res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            # print(f"{self.dones=}")

            for k in self.update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 

            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # print(f"{rewards.mean()=}")
            step_time_end = time.time()

            step_time += (step_time_end - step_time_start)

            # print(f"{shaped_rewards.mean()=}")
            if self.value_bootstrap and 'time_outs' in infos:
                rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]
     
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict



class ContinuousA2CBase(A2CBase):

    def __init__(self, base_name, params):
        A2CBase.__init__(self, base_name, params)

        self.is_discrete = False
        action_space = self.env_info['action_space']
        self.actions_num = action_space.shape[0]
        self.bounds_loss_coef = self.config.get('bounds_loss_coef', None)

        self.clip_actions = self.config.get('clip_actions', True)
        assert self.clip_actions

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)


    def init_tensors(self):
        A2CBase.init_tensors(self)
        self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
        self.tensor_list = self.update_list + ['obses', 'states', 'dones']

    def train_epoch(self):
        super().train_epoch()

        self.model.eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        self.model.train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        assert self.mini_epochs_num == 5
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            # print(len(self.dataset))
            assert len(self.dataset) == 8
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)

            av_kls = torch_ext.mean_list(ep_kls)
            if self.schedule_type == 'standard':
                self.last_lr = self.scheduler.update(self.last_lr, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            # check if it is in train mode
            # print(f"{self.model.running_mean_std.training}")
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']

        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        # print(f"{advantages.shape=}")
        advantages = torch.sum(advantages, axis=1)

        # print(f"{advantages.shape=}")
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.rank == 0:
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.rank_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)


                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards[0] <= self.last_mean_rewards):
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num
