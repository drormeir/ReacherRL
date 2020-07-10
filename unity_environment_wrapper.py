from unityagents import UnityEnvironment
import numpy as np

class unity_env(UnityEnvironment):

    def __init__(self, file_name, ind_brain=0, no_graphics=False, verbose=True, max_steps_in_single_episode=1000, score_goal=None,\
                 score_window_size=100):
        super().__init__(file_name=file_name, no_graphics=no_graphics)
        self.ind_brain   = ind_brain        
        self.brain_name  = super().brain_names[ind_brain]
        self.brain       = super().brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        env_info         = super().reset(train_mode=True)[self.brain_name] # reset the environment
        state            = env_info.vector_observations[self.ind_brain]    # get the current state
        self.state_size  = len(state)
        self.max_steps_in_single_episode   = max_steps_in_single_episode
        self.score_goal  = score_goal
        self.goal        = False
        self.reset_all_scores(score_window_size)
        if verbose:
            print("Selected brain name: ", self.brain_name)
            print("Selected brain:      ", self.brain)
            print("Number of actions:   ", self.action_size)
            print("Number of agents:    ", len(env_info.agents))
            print("States look like:    ", state)
            print("States have length:  ", self.state_size)
          
    

    def train(self, agent, output_filename, n_episodes = 10000, score_window_size=None, noise_minimal=0.01, noise_decay=0.99):
        """Deep Q-Learning.

        Params
        ======
            env (banana_env): the environment
            agent (ddqn_agent): the agent
            output_filename (string): file name for check point
            n_episodes (int): maximum number of training episodes
            eps_end (float): minimum value of epsilon-greedy action selection
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.reset_all_scores(score_window_size)
        agent.reset_noise_level()                    # initialize noise
        temp_checkpoint_name = "temp_checkpoint.pth"
        shutil.rmtree(temp_checkpoint_name,ignore_errors=True) # avoid file not found error
        while env.improvement >= 0:
            self.train_agent_on_single_episode(agent)            
            if agent.get_noise_level() > noise_minimal:
                agent.scale_noise(noise_decay)
            self.__update_statistics_with_completed_episode_score()
            end         = "\n" if env.improvement > 0 else ""
            ind_episode = len(env.all_scores)
            print('\rEpisode {}\t {}'.format(ind_episode,env.str_curr_window_score()), end=end)
            if self.__goal_reached_first_time():
                print('\nEnvironment goal reached in {:d} episodes!'.format(ind_episode))
            if ind_episode >= n_episodes:
                # rare case: maximum number of episodes  -->  end training loop in any case
                self.__test_improvement(env, agent, output_filename)
                break
            if env.improvement > 0:
                print('Saving checkpoint...')
                agent.save(temp_checkpoint_name)
                continue
            if env.improvement == 0:
                continue
            # env.improvement < 0   -->  environment signals no more improvements...
            # test if reduce learning rate can make things any better
            agent.load(temp_checkpoint_name)
            if not self.__test_improvement(env, agent, output_filename):
                break
            if agent.is_lr_at_minimum():
                break
            agent.learning_rate_step()

        shutil.rmtree(temp_checkpoint_name,ignore_errors=True) # avoid file not found error
        print('\nNo more improvements. End of training.')
        return env.scores, env.best_test_score, env.best_test_average, env.best_test_stdev

    def reset_all_scores(self, score_window_size=None):
        if score_window_size is None:
            score_window_size = self.scores_window.maxlen
        self.all_scores        = []                              # list containing scores from each episode
        self.scores_window     = deque(maxlen=score_window_size) # last recent scores
        self.__update_curr_window_score()
        self.__set_curr_window_as_best()
        self.best_test_score   = -np.inf
        self.best_test_average = -np.inf
        self.best_test_stdev   = np.inf
        self.improvement       = 0

    def str_curr_window_score(self):
        return "Composite={:5.2f}\t Average={:5.2f}\t Stdev={:5.2f}".format(self.curr_window_score,\
                                                                            self.curr_window_average,\
                                                                            self.curr_window_stdev)
    
    def train_agent_on_single_episode(self, agent, max_steps_in_single_episode=None):
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        state = self.reset(train_mode=True) # reset the environment and current episode score
        for i_step in range(max_steps_in_single_episode):
            action = agent.act(state, add_noise=True) # select an action with noise exploration
            next_state, reward, done, _ = self.step(action)
            agent.step(state, action, reward, next_state, done)
            if done:
                break
            state  = next_state
        return self.total_score_in_episode
    
    def test_agent_on_single_episode(self, agent, max_steps_in_single_episode=None, verbose=True):
        if verbose:
            print("Agent: Test on single episode starts...")
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        state = self.reset(train_mode=False) # reset the environment
        for i_step in range(max_steps_in_single_episode):
            action            = agent.act(state)     # select an action
            state, _, done, _ = self.step(action)
            if done:
                break
        if verbose:
            print("Agent: Test on single episode is over. Number of steps: {}  score: {5.2f}"\
                  .format(self.number_steps_in_episode, self.total_score_in_episode))
        return self.total_score_in_episode
    
    def reset(self, train_mode):
        env_info    = super().reset(train_mode=train_mode)[self.brain_name] # reset the environment
        state       = env_info.vector_observations[self.ind_brain]          # get the current state
        self.number_steps_in_episode = 0
        self.total_score_in_episode  = 0
        return state
    
    def step(self, action):
        env_info     = super().step(action)[self.brain_name]        # send the action to the environment
        next_state   = env_info.vector_observations[self.ind_brain] # get the next state
        reward       = env_info.rewards[self.ind_brain]             # get the reward
        done         = env_info.local_done[self.ind_brain]          # see if episode has finished
        self.nsteps += 1
        self.total_score_in_episode += reward
        return next_state, reward, done, env_info

    def __test_improvement(self, agent, output_filename):
        print("\nCompare last checkpoint against best test score so far...");
        score_window_size = self.scores_window.maxlen
        for i_episode in range(score_window_size):
            state = env.reset(train_mode=True) # reset the environment
            done  = False
            while not done:
                action, _         = agent.act(state)     # select an action
                state, _, done, _ = env.step(action)
            self.scores_window.append(env.total_score_in_episode)
            print("\rEpisode: {} out of: {}".format(i_episode+1,score_window_size),end="")
        print("\n")
        self.__update_curr_window_score()
        if self.curr_window_score <= self.best_test_score:
            print("Current test score {:5.2f} is worse than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                   self.best_test_score))
            return False
        print("Current test score {:5.2f} is better than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                self.best_test_score))        
        # saving test score
        self.best_test_score   = self.curr_window_score
        self.best_test_average = self.curr_window_average
        self.best_test_stdev   = self.curr_window_stdev
        self.__set_curr_window_as_best()
        self.improvement = 0 # train may continue
        agent.save(output_filename)
        return True
    
    def test_agent_on_many_episodes(self, agent, n_episodes=None, max_steps_in_single_episode=None, verbose=True):
        if n_episodes is None:
            n_episodes = self.scores_window.maxlen
        if verbose:
            print("Begin test agent on {} episodes...".format(n_episodes))
        if max_steps_in_single_episode is None:
            max_steps_in_single_episode = self.max_steps_in_single_episode
        scores = []
        for i_episode in range(n_episodes):
            state = self.reset(train_mode=True) # reset the environment
            for i_step in range(max_steps_in_single_episode):
                action            = agent.act(state)     # select an action with eploitation only
                state, _, done, _ = self.step(action)
                if done:
                    break
            scores.append(self.total_score_in_episode)
            if verbose:
                print("\rEpisode: {} out of: {}".format(i_episode+1,n_episodes),end="")
        scores_mean      = np.mean(scores)
        scores_stdev     = np.std(scores,ddof=1) if len(scores) > 1 else np.inf
        scores_composite = scores_mean - scores_stdev
        if verbose:
            print("\nComposite={:5.2f}\t Average={:5.2f}\t Stdev={:5.2f}".format(scores_composite, scores_mean, scores_stdev))
        return scores_composite, scores_mean, scores_stdev, scores
    
    def __update_statistics_with_completed_episode_score(self):
        completed_episode_score = self.total_score_in_episode
        current_score_is_better_than_before = completed_episode_score > self.curr_window_average
        self.scores_window.append(completed_episode_score)       # save most recent score
        self.all_scores.append(completed_episode_score)          # save most recent score
        self.__update_curr_window_score()
        self.improvement  = 0 # train may continue
        num_episodes  = len(self.all_scores)
        if num_episodes < self.scores_window.maxlen:
            return
        if self.curr_window_score > self.best_window_score and current_score_is_better_than_before:
            # real score improvement :-)
            self.improvement = 1 # train should continue
            self.__set_curr_score_window_as_best()
        elif self.curr_window_average > self.temp_window_average:
            # no real score improvement but still try a bit more...
            self.temp_window_average = self.curr_window_average
            self.best_window_episode = num_episodes
        elif num_episodes >= self.best_window_episode + self.scores_window.maxlen:
            # no score improvement anymore :-(
            self.improvement = -1 # train should stop

    def __update_curr_window_score(self):
        curr_len = len(self.scores_window)
        self.curr_window_average = np.mean(self.scores_window) if curr_len > 0 else -np.inf
        # estimate standard deviation of entire population
        self.curr_window_stdev   = np.std(self.scores_window,ddof=1) if curr_len > 1 else np.inf
        self.curr_window_score   = self.curr_window_average - self.curr_window_stdev

    def __set_curr_score_window_as_best(self):
        self.best_window_score   = self.curr_window_score
        self.best_window_average = self.curr_window_average
        self.best_window_stdev   = self.curr_window_stdev
        self.temp_window_average = self.curr_window_average
        self.best_window_episode = len(self.all_scores) # used for check number of episodes since best episode
        
    def __goal_reached_first_time(self):
        if self.goal:
            return False
        self.goal = self.score_goal is not None and self.curr_window_average is not None\
        and self.curr_window_average >= self.score_goal
        return self.goal
    
