from unityagents import UnityEnvironment
import numpy as np

class unity_env(UnityEnvironment):

    def __init__(self, file_name, ind_brain=0, no_graphics=False, verbose=True, max_steps=1000):
        super().__init__(file_name=file_name, no_graphics=no_graphics)
        self.ind_brain   = ind_brain        
        self.brain_name  = super().brain_names[ind_brain]
        self.brain       = super().brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info         = super().reset(train_mode=True)[self.brain_name] # reset the environment
        state            = env_info.vector_observations[self.ind_brain]    # get the current state
        self.state_size  = len(state)
        self.max_steps   = max_steps
        if verbose:
            print("Selected brain name: ", self.brain_name)
            print("Selected brain:      ", self.brain)
            print("Number of actions:   ", self.action_size)
            print("Number of agents:    ", len(env_info.agents))
            print("States look like:    ", state)
            print("States have length:  ", self.state_size)
                
    def reset(self, train_mode):
        env_info    = super().reset(train_mode=train_mode)[self.brain_name] # reset the environment
        state       = env_info.vector_observations[self.ind_brain]          # get the current state
        self.nsteps = 0
        self.score  = 0
        return state
    
    def step(self, action):
        env_info     = super().step(action)[self.brain_name]        # send the action to the environment
        next_state   = env_info.vector_observations[self.ind_brain] # get the next state
        reward       = env_info.rewards[self.ind_brain]             # get the reward
        done         = env_info.local_done[self.ind_brain]          # see if episode has finished
        self.nsteps += 1
        self.score  += reward
        return next_state, reward, done, env_info

    def full_run_agent_on_single_episode(self, agent, max_steps=None, verbose=True):
        if verbose:
            print("Agent: Full run on single episode starts...")
        if max_steps is None:
            max_steps = self.max_steps
        state = self.reset(train_mode=False) # reset the environment
        for i_step in range(max_steps):
            action            = agent.act(state)     # select an action
            state, _, done, _ = self.step(action)
            if done:
                break
        if verbose:
            print("Agent: Full run on single episode is over. Number of steps: {}  score: {5.2f}".format(self.nsteps,self.score))
        return self.score
    
    def test_agent_on_many_episodes(self, agent, n_episodes=100, max_steps=None, verbose=True):
        if verbose:
            print("Begin test agent on {} episodes...".format(n_episodes))
        if max_steps is None:
            max_steps = self.max_steps
        scores = []
        for i_episode in range(n_episodes):
            state = self.reset(train_mode=True) # reset the environment
            for i_step in range(max_steps):
                action            = agent.act(state)     # select an action with eploitation only
                state, _, done, _ = self.step(action)
                if done:
                    break
            scores.append(self.score)
            if verbose:
                print("\rEpisode: {} out of: {}".format(i_episode+1,n_episodes),end="")
        scores_mean      = np.mean(scores)
        scores_stdev     = np.std(scores,ddof=1) if len(scores) > 1 else np.inf
        scores_composite = scores_mean - scores_stdev
        if verbose:
            print("\nComposite={:5.2f}\t Average={:5.2f}\t Stdev={:5.2f}".format(scores_composite, scores_mean, scores_stdev))
        return scores_composite, scores_mean, scores_stdev, scores

    def train_agent_on_single_episode(self, agent, agent_param=None, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps
        state = self.reset(train_mode=True) # reset the environment
        for i_step in range(max_steps):
            action = agent.act(state, agent_param)                   # select an action with exploration
            next_state, reward, done, _ = self.step(action)
            agent.step(state, action, reward, next_state, done)
            if done:
                break
            state  = next_state
        return self.score
        