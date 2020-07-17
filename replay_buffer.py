import numpy as np
# some users might not install torch at the first place
#import torch # optional import

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, action_type, buffer_size=int(1e5), batch_size=128, seed=0,\
                 no_reward_value    = 0.0,\
                 no_reward_dropout  = 0.0,\
                 pytorch_device     = None):
        """Initialize a ReplayBuffer object.

        Params
        ======
            state_size  (int): dimension of state vector
            action_size (int): dimension of action vector ( 1 for simple DQN)
            action_type (int): np.float32 for DDPG   np.int8 or np.int32 for DQN
            buffer_size (int): maximum size of buffer
            batch_size  (int): size of each training batch
            seed (int): random seed
            pytorch_device: torch.device("cuda:0" if use_cuda else "cpu")
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.action_type = action_type
        self.batch_size  = batch_size
        self.buffer_size = buffer_size
        self.seed        = seed
        self.rand        = np.random.default_rng(seed)
        self.no_reward_value   = no_reward_value
        self.no_reward_dropout = no_reward_dropout
        self.pytorch_device = pytorch_device
        self.states      = np.empty((self.buffer_size,self.state_size),  dtype=np.float32)
        self.actions     = np.empty((self.buffer_size,self.action_size), dtype=action_type)
        self.rewards     = np.empty((self.buffer_size,1),                dtype=np.float32)
        self.next_states = np.empty((self.buffer_size,self.state_size),  dtype=np.float32)
        self.dones       = np.empty((self.buffer_size,1),                dtype=np.float32)
        
        self.res_states      = np.empty((self.batch_size,self.state_size),  dtype=np.float32)
        self.res_actions     = np.empty((self.batch_size,self.action_size), dtype=action_type)
        self.res_rewards     = np.empty((self.batch_size,1),                dtype=np.float32)
        self.res_next_states = np.empty((self.batch_size,self.state_size),  dtype=np.float32)
        self.res_dones       = np.empty((self.batch_size,1),                dtype=np.float32)
        self.reset()

    def clone(self):
        return ReplayBuffer(state_size=self.state_size, action_size=self.action_size, action_type=self.action_type,\
        buffer_size=self.buffer_size, batch_size=self.batch_size, seed=self.seed,\
        no_reward_value=self.no_reward_value, no_reward_dropout=self.no_reward_dropout,\
        pytorch_device=self.pytorch_device)

    def reset(self):
        self.current_len = 0
        self.count_entered_no_reward = 0

    def __iadd__(self, other):
        for i in range(min(other.current_len,other.buffer_size)):
            self.add(other.states[i,:],other.actions[i,:],other.rewards[i,0],other.next_states[i,:],other.dones[i,0])
        return self # must return this object or else it will be destoryed


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        if not done and (abs(reward) < 1e-8):
            no_reward_dropout = self.count_entered_no_reward * self.no_reward_dropout / (self.current_len + 1)
            if (no_reward_dropout > 0.0) and (self.rand.uniform() < no_reward_dropout):
                return
            self.count_entered_no_reward += 1
            reward = self.no_reward_value
        ind_pos                     = self.current_len % self.buffer_size
        self.current_len           += 1
        self.states[ind_pos,:]      = state
        self.actions[ind_pos,:]     = action
        self.rewards[ind_pos][0]    = reward
        self.next_states[ind_pos,:] = next_state
        self.dones[ind_pos][0]      = done

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        curr_len = min(self.current_len,self.buffer_size)
        if curr_len < self.batch_size:
            return None
        
        indexes       =  self.rand.choice(range(curr_len), size=self.batch_size,replace=False)
        
        self.res_states[:]      = self.states[indexes,:]
        self.res_actions[:]     = self.actions[indexes,:]
        self.res_rewards[:]     = self.rewards[indexes,:]
        self.res_next_states[:] = self.next_states[indexes,:]
        self.res_dones[:]       = self.dones[indexes,:]
        if self.pytorch_device is None:
            # not using pytorch
            return (self.res_states, self.res_actions, self.res_rewards, self.res_next_states, self.res_dones)
        
        # using pytorch
        import torch # import for this scope only
        states      = torch.from_numpy(self.res_states).float().to(self.pytorch_device)
        if self.action_type == np.float32:
            actions = torch.from_numpy(self.res_actions).float().to(self.pytorch_device)
        else:
            actions = torch.from_numpy(self.res_actions).long().to(self.pytorch_device)
        rewards     = torch.from_numpy(self.res_rewards).float().to(self.pytorch_device)
        next_states = torch.from_numpy(self.res_next_states).float().to(self.pytorch_device)
        dones       = torch.from_numpy(self.res_dones).float().to(self.pytorch_device)
        return (states, actions, rewards, next_states, dones)
