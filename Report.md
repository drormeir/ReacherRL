[//]: # (Image References)
[image1]: https://github.com/drormeir/ReacherRL/blob/master/Actor.jpg "Actor Network"
[image2]: https://github.com/drormeir/ReacherRL/blob/master/Critic.jpg "Critic Network"
[image3]: https://github.com/drormeir/ReacherRL/blob/master/training_Actor.jpg "Training Actor"
[image4]: https://github.com/drormeir/ReacherRL/blob/master/training_Critic.jpg "Training Critic"

# Reacher Report
This implementation of DDPG inherits it is a central idea from ["Udacity's bipedal"](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal)

According to the current state, the problem I solved in this project is calculating the robotic arm's best action to reach a moving object's location. The Unity environment generates a vector of floating-point numbers as the current state and accepts another vector of floating-point numbers as the next action. The state vector contains the measurement values of each part of the robotic arm's position and velocity. Also, it contains the position and velocity of the target object. The action vector represents the power needed to apply to each motor of the arm.

We chose the DDPG algorithm to solve this problem. It consists of two types of networks: The Actor and the Critic, where each one of them has two instances: the local and the target.

# Projects Source Files Description

### actor_critic_model.py
This file contains the architecture of the Actor-network and the Critic-network.

The class `Actor` receives its entire network architecture as a list in one of its parameters.

The class `Critic` can have the entire network architecture as a parameter that is consists of 3 lists:
* The first list is the layers architecture for the State vector input.
* The second list is the layers architecture for the Action vector input.
* The third list is the layers architecture after the union of the first two.

##### Actor Network
The Actor-network takes the state vector as an input and computes the best action vector to take for the next step. This chart describes its architecture:
![Actor Network][image1]

The final product of the DDPG algorithm is a trained Actor-network, which can calculate the best action for the current state in the inference stage. However, contrary to supervised learning, we do not have any human annotators who can teach the network the RL's best action.

##### Critic Network
The Critic-network receives two inputs: A state vector and an action vector. It calculates the total value of the next state directly without calculating the next state explicitly. This chart describes its architecture:
![Critic Network][image2]

### DDPG_agent.py
The class `DDPG_agent` contains the implementation of the RL agent's algorithm. 
Every 20 steps, the agent samples four mini-batches from the Replay Buffer and trains the networks according to this data.

##### Training the Actor-Network
In order to train the Actor-network, first, we need to train and freeze a Critic network, then we combine it with the Actor-network as in the following chart:
![Training Actor][image3]

The only input for this mini-training process is a state vector of values. The Actor-network calculates the best action and pass it into the Critic-network. There it also uses the original state vector to calculate the final value of the input state. We use that result value as the target function for maximizing in the Back Propagation process. The classic process is to minimize a function. Hence, we give the Back Propagation process a minus sign so it would be able to minimize the final negative value, which is equal to maximizing the positive value.

##### Training the Critic Network

As mentioned earlier, A Critic-network should calculate the couple's next state's value: state-action. On the other hand, a couple of Actor-Critic networks can calculate a state value assuming using the best action implicitly. Hence taking a replay buffer containing state, action, next-state gives us the necessary data for training the Critic-network. On the left side of the equation, we put the Critic-network result from the state-action values. On the right side of the equation, we put the next state value result using a constant Actor-Critic couple. Adding the rest of the data from the replay buffer to the right-hand side, we get the Bellman equation.
![Training Critic][image4]

##### Summarizing the training process of DDPG
The training process uses a replay buffer store experience and draw samples from it. It also uses constants instances of the Actor and Critic network as baseline calculations of the program's previous steps. These baseline versions are annotated as "target networks" in the program. They get updates very slowly using the `tau` parameter, which controls the blending between the local versions of the networks and the target versions.

The training/update phase of the DDPG algorithm has several stages:
1. Train the Critic local network:
* Generate a mini-batch of training data from a random sample of experience.
* Calculate the value of the next state using the baseline values from the target networks.
* Train the Critic local network with the state-action values as input and estimation of target values from the previous step and the rewards from the mini-batch
2. Train the Actor local network:
* Set the states from the mini-batch into the Actor input and also into the Critic input.
* Connect the output actions from the Actor-network into the input actions of the Critic.
* Perform Gradient Ascent on collaborative networks to maximize the mean output values from the Critic network while letting only the Actor's weights change during this process. The weights of the Critic are frozen.
3. Perform Soft Update on both of the target networks with a small weight (tau) on the local networks.

### unity_environment_wrapper.py
The `unity_env` class inherits from the `UnityEnvironment` class and encapsulates the brain's index and brain's name within for clarity.

The `step` member function gives a better API for the user to interact with the environment. It mimics the simple `step` function API of the `OpenAI-Gym` environment class.

This class also keep tracking after the whole scores graph and after the most recent scores window. This window score is a `deque` object with a maximum length of 100. The member function `_recalc_curr_window_score` calculates the window's average, standard deviation, and a composite score equal to the average minus standard deviation. 

Moreover, this class compares each new completed episode's score against the previous average window score and determines its improvement status. This calculation is in the member function: `train` and the resulted status has three possible values:
* A status of +1 means "training should continue": This module checks for two things: First, if the recently completed episode score is better than the previous average score. Second, the current window's composite score is better than the previous window's composite score. In case both are true, then the environment will signal that training should continue.
* A status of -1 means "training should stop": This situation results from two checks: First, if the current composite score is worse than the best composite score so far. Second, the number of completed episodes since the last improvement is larger than the window size. In other words, it means that the agent is way after its peak performance.
* A status of 0 means: "training can continue": It is the most likely result; still, the training process can decide to stop after too many tries.
* A unique approach when two cases happen together: First, the current window composite score is smaller than the best window's composite score. Second, the current average is more significant than the best window's average. An improvement may be ahead. Hence the index of the best composite score so far is moved to the current index. This approach gives the training process extra tries until it gives up.

This class also contains the highest level functions of controlling the training process.

The `train` member function is responsible for training a given agent in a given environment. After each episode, it reduces the exploration epsilon and checks the environment improvement status.
* If the improvement status is positive, then it saves the current agent state.

The `test_agent_on_many_episodes` member function takes a trained agent and conducts 100 executions (exploitation only) on the environment and saves their scores. After that, it prints the statistics of those tests.


### replay_buffer.py
This file contains the Replay Buffer needed for learning.
For each step in the episode, the agent stores a tuple of data into this "Round Robin" buffer. After a given number of steps, the agent pulls a sample from this buffer and uses it as a mini-batch for the deep learning network.

The buffer size is large (1M tuples), and each sample of a single mini-batch contains 128 tuples.

Please note the source of Udacity contains this Replay Buffer with a `deque` object, but here, the replay buffer using `np.array` for the sake of performance.


### OU_noise.py
For the exploration/exploitation of the problem environment space, we used the ["Ornstein-Uhlenbeck noise process"](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) together with "Epsilon-Greedy policy" for environment exploration. Then, we add the noise values to the Actor's output actions values giving the final input to the robotic arm's motors.

The OU-Noise process has three parameters:
* Mu - The mean distribution of the noise. In this project, we want to have a noise as a difference from the Actor's actual output. Hence Mu is always zero.
* Sigma - The standard deviation of the noise. We choose it to be 0.2
* Theta - This noise process means reverting the previous step values into the Mu value and then add new noise values. The Theta parameter controls the strength of the mean-reverting process. Zero means no reverting, and the results will represent a random walk. A value of one means full revert to the mean and forget previous values, which result in a random uniform distribution of values around the mean. 

The agent starts exploring the environment space using the noise from the OU process. As the training goes on, the Sigma parameters go to zero. The Theta parameter goes to one to exploit the knowledge from the trained networks.


## The Training Process
The training process was performed in the Jupyter notebook: `Continuous_Control.ipynb.`
At the bottom of this notebook, one can find the score graph.

##### Hyperparameters summary
Most of the hyperparameters were not tuned. We tested a few combinations to find better and faster results, but without any extensive research.

General parameters:
* random_seed        = 1       # A global seed for random numbers generator. Both the noise process and the replay buffer use it. Setting this parameter enables reproducible results.
* max_num_episodes   = 10000   # Maximum number of episodes for entire learning process.
* score_window_size  = 100     # Moving average window size for accumulating scores and calculation of average score
* num_episode_search = 500     # maximum number of episode to search for a better score

DDPG learning parameters:
* gamma              = 0.95    # discount factor of future rewards for the Bellman equation. A number close to one means future rewards have high importance.
* tau                = 0.001   # soft update factor. In each training step, we blend local networks into the target networks with minimal weight.
* update_every       = 20      # update networks during episode every number of steps.
* update_times       = 4       # when updating the networks, take several mini-batches from the replay buffer.
* lr_actor           = 0.0001  # learning rate of the Actor-network
* lr_critic          = 0.001   # learning rate of the Critic-network

Networks architecture:
* 'b' means: Batch Norm
* 'r' means: Relu
* number means: size of dense layer

* actor_arch         = ['b', 128, 'b', 'r', 256, 'b', 'r']
* critic_arch        = [['b', 128], ['b', 64], ['b', 'r', 256, 'b', 'r']]

Parameters for the replay buffer:
* replay_buffer_size = 1000000 # the maximum number of tuples that can behold in the replay buffer.
* replay_batch_size  = 128     # mini-batch size for training the actor-critic networks.
* no_reward_value    = -0.1    # An alternative value when the program encounters a zero reward. Overriding zero reward with a negative one will influence the Critic-network to avoid "playing safe" with zero rewards moves and focus on moves that bring positive reward.
* no_reward_dropout  = 0.9     # Especially initially, there is an "overflow" of steps with negative rewards, which obscures the network from learning the best action. We create this parameter to remove most of the bad moves from the replay buffer to balance the amount of "good" and "bad" moves. This method speeds up the learning process.


OU noise parameters:
* noise_sigma        = 0.2     # Standard deviation of the noise values
* noise_theta        = 0.15    # Mean reverting noise factor
* noise_minimal      = 0.01    # exploration/exploitation minimal value
* noise_decay        = 0.99    # exploration/exploitation decay factor


## Challenges and further discussion
There are several challenges when trying to estimate the current score of the agent:

* After each several steps, the training process performs a single learning phase with a mini-batch. It is impossible to have an exact measurement of the agent's current score to understand if we passed beyond the peak performance. Hence, the environment class uses the episode's score as a proxy to the exact value.
* The training process should stop when the agent reaches its peak performance. However, the moving average lags behind the real unknown average score. The program can calculate the correct average score by performing a real test (the one in `test` member function). The simple test process does not use random exploration moves and does not learn any new actions' values. The resources to make a real test after each episode are too scarce. Hence there is no reasonable solution besides the lagging moving average.
* I achieved the environment goal of score 30.0 after 1200 episode and continued the training process until it maxed out after 2350 episode with an average score of 48.9


## Future work:
The replay buffer is simple and does not take into account the different importance of each learning step. This project might use in the future a better experience manager called "Priority Experience Replay" (PER) that takes in to account the value misses and prioritizing them accordingly.

Dror Meirovich
