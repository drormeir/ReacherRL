import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300, pytorch_device=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed   = torch.manual_seed(seed)
        self.fc1    = nn.Linear(state_size, fc1_units)
        self.fc2    = nn.Linear(fc1_units, fc2_units)
        self.fc3    = nn.Linear(fc2_units, action_size)
        self.noise  = None
        self.reset_parameters()
        self.device = pytorch_device
        self.noise  = OUNoise(size=action_size, seed=random_seed, pytorch_device=self.device)
        self.use_noise_once = False
        if self.device is not None:
            self.to(self.device)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def reset_noise_level(self):
        self.noise.reset_scale()
        
    def noise_decay(self, factor):
        self.noise.scale_noise(factor)
    
    def get_noise_level(self):
        return self.noise.calc_scale()
    
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.use_noise_once:
            x += self.noise.sample()
            self.use_noise_once = False
        return torch.tanh(x)

    def eval_numpy(self, state, add_noise=None):
        state = torch.from_numpy(state).float()
        if self.device:
            state = state.to(self.device)
        self.use_noise_once = add_noise is not None and add_noise
        self.eval() # set model to "eval" mode
        with torch.no_grad():
            action = self(state).cpu().data.numpy()
        return action
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300, pytorch_device=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2  = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3  = nn.Linear(fc2_units, 1)
        self.reset_parameters()
        if pytorch_device is not None:
            self.to(pytorch_device)

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x  = torch.cat((xs, action), dim=1)
        x  = F.relu(self.fc2(x))
        return self.fc3(x)
