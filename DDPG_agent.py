import numpy as np
from replay_buffer import ReplayBuffer
from OU_noise import OUNoise
from actor_critic_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

import os
import shutil

class DDPG_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, replay_buffer_size = int(1e5), replay_batch_size=128, random_seed=0, gamma=0.99,\
                 tau=1e-3, lr_actor=1e-4, lr_critic=1e-3, weight_decay=0, use_cuda=False):
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
            weight_decay(float)       L2 weight decay
        """
        self.state_size       = state_size
        self.action_size      = action_size
        self.tau              = tau
        self.gamma            = gamma
        if use_cuda:
            use_cuda = torch.cuda.is_available()
        device_name = "cuda:0" if use_cuda else "cpu"
        print("Initializing DDPG_Agent with PyTorch device named:",device_name)
        self.device = torch.device(device_name)
        # Actor Network (w/ Target Network)
        self.actor_local      = Actor(state_size, action_size, random_seed, pytorch_device=self.device)
        self.actor_target     = Actor(state_size, action_size, random_seed, pytorch_device=self.device)
        self.actor_optimizer  = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local     = Critic(state_size, action_size, random_seed, pytorch_device=self.device)
        self.critic_target    = Critic(state_size, action_size, random_seed, pytorch_device=self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)


        # Replay memory
        self.memory = ReplayBuffer(state_size=state_size, action_size=action_size, action_type=np.float32,\
                                   buffer_size=replay_buffer_size, batch_size=replay_batch_size, seed=random_seed,\
                                   pytorch_device=self.device)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

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
        Q_targets      = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected     = self.critic_local(states, actions)
        critic_loss    = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        self.actor_local.train()
        actions_pred = self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)                     

    def act(self, state, add_noise=None):
        """Returns actions for given state as per current policy."""
        return self.actor_local.eval_numpy(state, add_noise)

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