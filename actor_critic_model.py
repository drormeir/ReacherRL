import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModuleListUtil(nn.ModuleList):
    def __init__(self, in_features = -1, layers_arch = [], out_features=0):
        super(ModuleListUtil,self).__init__()
        self.build(in_features=in_features, layers_arch=layers_arch, out_features=out_features)

    def build(self, in_features, layers_arch, out_features=0):
        self.out_features = in_features
        self.layers_arch  = layers_arch.copy()
        for layer in layers_arch:
            if isinstance(layer, str):
                if layer == 'b':
                    self.append(nn.BatchNorm1d(self.out_features))
                elif layer == 'r':
                    self.append(nn.LeakyReLU())
            elif isinstance(layer,int):
                self.append_linear(out_features=layer)
        if out_features > 0:
            self.append_linear(out_features=out_features)

    def append_linear(self, out_features):
        # reference for nn.init.xavier_uniform_() can be found at: https://pytorch.org/docs/stable/nn.init.html`
        linear = nn.Linear(in_features=self.out_features, out_features=out_features)
        nn.init.xavier_uniform_(linear.weight)
        self.append(linear)
        self.out_features = out_features

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, layers_arch = [100,100,50], pytorch_device=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()
        self.state_size     = state_size
        self.action_size    = action_size
        self.layers_list    = ModuleListUtil(in_features=self.state_size, layers_arch=layers_arch, out_features=self.action_size)
        self.pytorch_device = pytorch_device
        if self.pytorch_device is not None:
            self.to(self.pytorch_device)
    
    def clone(self):
        return Actor(state_size=self.state_size, action_size=self.action_size, layers_arch=self.layers_list.layers_arch.copy(), pytorch_device=self.pytorch_device)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.layers_list.forward(state)

    def eval_numpy(self, state):
        if state.ndim == 1:
            state = state.reshape((1,len(state)))
        state = torch.from_numpy(state).float()
        if self.pytorch_device:
            state = state.to(self.pytorch_device)
        self.eval() # set model to "eval" mode
        with torch.no_grad():
            action = self(state).cpu().data.numpy()
        self.train()
        return action
    
class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, layers_arch = [[100],[50],[50]], pytorch_device=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()
        self.state_size     = state_size
        self.action_size    = action_size
        self.pytorch_device = pytorch_device
        self.state_layers   = ModuleListUtil(in_features=self.state_size,  layers_arch=layers_arch[0])
        self.action_layers  = ModuleListUtil(in_features=self.action_size, layers_arch=layers_arch[1])
        self.combine_layers = ModuleListUtil(in_features=self.state_layers.out_features + self.action_layers.out_features, layers_arch=layers_arch[2], out_features=1)
        if pytorch_device is not None:
            self.to(pytorch_device)
        return

    def clone(self):
        return Critic(state_size=self.state_size, action_size=self.action_size,\
        layers_arch=[self.state_layers.layers_arch.copy(), self.action_layers.layers_arch.copy(), self.combine_layers.layers_arch.copy()],
        pytorch_device=self.pytorch_device)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.state_layers.forward(state)

        xa = self.action_layers.forward(action)

        x  = torch.cat((xs, xa), dim=1)

        return self.combine_layers.forward(x)
