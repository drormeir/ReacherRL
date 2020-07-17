import numpy as np
from replay_buffer import ReplayBuffer
from OU_noise import OUNoise
from actor_critic_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil

def next_value(curr_value, target_value):
    # curr_value starts at 1.0, target_value is smaller
    ret = curr_value / (1.0 + curr_value - target_value)
    return min(max(ret,target_value),curr_value) # numerical fix

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size,\
                 replay_buffer_size = int(1e6),\
                 replay_batch_size  = 128,\
                 random_seed        = 1,\
                 gamma              = 0.95,\
                 tau                = 1e-3,\
                 update_every       = 20,\
                 update_times       = 4,\
                 lr_actor           = 1e-4,\
                 lr_critic          = 1e-3,\
                 actor_clip_grad    = None,\
                 critic_clip_grad   = None,\
                 noise_sigma        = 0.2,\
                 noise_theta        = 0.15,\
                 no_reward_value    = -0.1,\
                 no_reward_dropout  = 0.9,\
                 actor_arch         = ['b',128,'b','r',256,'b','r'],\
                 critic_arch        = [['b',128],['b',64],['b','r',256,'b','r']],\
                 use_cuda           = False,\
                 verbose_level      = 1):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):         dimension of each state
            action_size (int):        dimension of each action
            replay_buffer_size (int): replay buffer size
            replay_batch_size (int):  minibatch size
            random_seed (int):        random seed
            gamma(float):             discount factor of next state value
            tau(float):               for soft update of target parameters
            lr_actor(float):          learning rate of the actor 
            lr_critic(float):         learning rate of the critic
        """
        self.verbose_level    = verbose_level
        self.state_size       = state_size
        self.action_size      = action_size
        self.gamma            = gamma
        self.current_gamma    = 0 # next state value is irrelevant at first steps, and grows upto gamma during trainings
        if use_cuda:
            use_cuda = torch.cuda.is_available()
        device_name = "cuda:0" if use_cuda else "cpu"
        if self.verbose_level > 0:
            print("Initializing DDPG_Agent with PyTorch device named:",device_name)
            print("replay_buffer_size =",replay_buffer_size)
            print("replay_batch_size  =",replay_batch_size)
            print("random_seed        =",random_seed)
            print("gamma              =",gamma)
            print("tau                =",tau)
            print("update_every       =",update_every)
            print("update_times       =",update_times)
            print("lr_actor           =",lr_actor)
            print("lr_critic          =",lr_critic)
            print("actor_clip_grad    =",actor_clip_grad)
            print("critic_clip_grad   =",critic_clip_grad)
            print("noise_sigma        =",noise_sigma)
            print("noise_theta        =",noise_theta)
            print("no_reward_value    =",no_reward_value)
            print("no_reward_dropout  =",no_reward_dropout)
            print("actor_arch         =",actor_arch)
            print("critic_arch        =",critic_arch)
        self.device = torch.device(device_name)
        torch.manual_seed(random_seed)
        # Actor Network (w/ Target Network)
        self.actor_local       = Actor(state_size=state_size, action_size=action_size, layers_arch=actor_arch, pytorch_device=self.device)
        self.actor_target      = self.actor_local.clone()
        self.actor_lr_max      = lr_actor
        self.actor_clip_grad   = actor_clip_grad

        # Critic Network (w/ Target Network)
        self.critic_local      = Critic(state_size=state_size, action_size=action_size, layers_arch=critic_arch, pytorch_device=self.device)
        self.critic_target     = self.critic_local.clone()
        self.critic_lr_max     = lr_critic
        self.critic_clip_grad  = critic_clip_grad

        self.lr_min            = 1e-5
        self.lr_decay          = 0.5
        self.reset_lr()
        self.noise             = OUNoise(size=action_size, seed=random_seed, sigma=noise_sigma, theta=noise_theta)

        # Replay memory
        self.memory            = ReplayBuffer(state_size=state_size, action_size=action_size, action_type=np.float32,\
                                             buffer_size=replay_buffer_size, batch_size=replay_batch_size, seed=random_seed,\
                                             no_reward_value=no_reward_value, no_reward_dropout=no_reward_dropout,\
                                             pytorch_device=self.device)
        self.update_every      = update_every
        self.last_update       = 0
        self.n_steps           = 0
        self.update_times      = update_times
        self.tau               = tau
        self.current_tau       = 1.0
        self.update_target_networks() # current_tau == 1   --> copy networks
        self.prepare_for_new_episode()


    def prepare_for_new_episode(self):
        pass

    def act(self, state, add_noise=None):
        """Returns actions for given state as per current policy."""
        raw_action = self.actor_local.eval_numpy(state)
        if add_noise is not None and add_noise:
            raw_action += self.noise.sample()
        self.raw_action = raw_action # save logits to be used in step()
        return np.tanh(raw_action) # convert logit to real actions [-1,+1] for environment

    def random_action(self):
        return np.random.uniform(low=-1.0,high=1.0,size=self.action_size)

    def step(self, state, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, self.raw_action, reward, next_state, done)
        self.n_steps += 1
        if not done and self.n_steps < self.last_update + self.update_every:
            return
        self.last_update = self.n_steps

        for i_update in range(self.update_times):
            # Learn, if enough samples are available in memory
            experiences = self.memory.sample()
            if experiences is None:
                return
            """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
            where:
                actor_target(state) -> action
                critic_target(state, action) -> Q-value

            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            states, actions, rewards, next_states, dones = experiences

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models
            actions_next   = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, actions_next)
            # Compute Q targets for current states (y_i)
            Q_targets      = rewards + self.get_gamma() * Q_targets_next * (1 - dones)
            # Compute critic loss
            Q_expected     = self.critic_local(states, actions)
            critic_loss    = F.mse_loss(Q_expected, Q_targets)
            # Minimize the loss
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.critic_clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.critic_clip_grad)
            self.critic_optimizer.step()

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss   = -self.critic_local(states, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.actor_clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.actor_clip_grad)
            self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.update_target_networks()

    def get_gamma(self):
        ret = self.current_gamma
        self.current_gamma = 1 - next_value(1 - self.current_gamma, 1 - self.gamma)
        return ret

    def update_target_networks(self):
        self.soft_update(self.critic_local, self.critic_target, self.current_tau)
        self.soft_update(self.actor_local, self.actor_target, self.current_tau)                     
        self.current_tau = next_value(self.current_tau, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def reset_noise_level(self):
        self.noise.reset()
        
    def scale_noise(self, factor):
        self.noise.scale_noise(factor)
    
    def get_noise_level(self):
        return self.noise.calc_scale()
    
    def learning_rate_step(self):
        if self.lr_at_minimum:
            return
        self.actor_lr      = max(self.actor_lr*self.lr_decay, self.lr_min)
        self.critic_lr     = max(self.critic_lr*self.lr_decay, self.lr_min)
        self.lr_at_minimum = self.critic_lr <= self.lr_min and self.actor_lr <= self.lr_min
        if self.verbose_level > 1:
            print("\nChanging learning rates to: actor:{:.4e} critic:{:.4e}".format(self.actor_lr,self.critic_lr))
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.critic_lr
        

    def is_lr_at_minimum(self):
        if self.lr_at_minimum and self.verbose_level > 1:
            print("\nCannot reduce learning rate because it is already at the minimum: {:.4e}".format(self.lr_min))
        return self.lr_at_minimum
            
    def reset_lr(self):
        self.actor_lr         = self.actor_lr_max
        self.actor_optimizer  = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        self.critic_lr        = self.critic_lr_max
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr)
        self.lr_at_minimum    = False


    def save(self, filename):
        shutil.rmtree(filename,ignore_errors=True) # avoid file not found error
        os.makedirs(filename)
        torch.save(self.actor_local.state_dict(),      os.path.join(filename,"actor_local.pth"))
        torch.save(self.actor_target.state_dict(),     os.path.join(filename,"actor_target.pth"))
        torch.save(self.actor_optimizer.state_dict(),  os.path.join(filename,"actor_optimizer.pth"))
        torch.save(self.critic_local.state_dict(),     os.path.join(filename,"critic_local.pth"))
        torch.save(self.critic_target.state_dict(),    os.path.join(filename,"critic_target.pth"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(filename,"critic_optimizer.pth"))

    def load(self, filename):
        self.actor_local.load_state_dict     (torch.load(os.path.join(filename,"actor_local.pth")))
        self.actor_target.load_state_dict    (torch.load(os.path.join(filename,"actor_target.pth")))
        self.actor_optimizer.load_state_dict (torch.load(os.path.join(filename,"actor_optimizer.pth")))
        self.critic_local.load_state_dict    (torch.load(os.path.join(filename,"critic_local.pth")))
        self.critic_target.load_state_dict   (torch.load(os.path.join(filename,"critic_target.pth")))
        self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename,"critic_optimizer.pth")))