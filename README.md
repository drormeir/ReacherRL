[//]: # (Image References)
[image1]: https://github.com/drormeir/ReacherRL/blob/master/TrainedAgent.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# ReacherRL
This project was submitted as part of [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and is a solution to [UnityML "Reacher"](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

The purpose of the project is to build and train a single agent that tries to maintain its position at the target location for as many time steps as possible.
![Trained Agent][image1]

This problem is episodic, where each episode is consists of 1000 steps. The environment provides a reward of +0.1 for each step that the agent's hand is in the goal location, and zero otherwise. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible. The minimal requirement for success is to have an average score of at least 30.0 points in 100 consecutive episodes.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a floating point number between -1 and 1.

The agent runs on Python 3.6 + PyTorch. The paper that describes the algorithm is ["DDPG-network"](https://arxiv.org/abs/1509.02971). All the rest of implementation details can be found at: [report.md](https://github.com/drormeir/ReacherRL/blob/master/Report.md)

The original git repo of this project is at:
https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_navigation

# Installation
To set up a python environment to run the code in this repository, please follow the instructions below:
1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__: 
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__: 
    ```bash
    conda create --name drlnd python=3.6 
    conda activate drlnd
    ```
2. Install pytorch using conda:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
3. Clone this git repo
```bash
git clone git@github.com:drormeir/ReacherRL.git
cd ReacherRL
pip install .
```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


# Usage
The Jupyter notebook `Continuous_Control.ipynb` imports all necessary dependencies and the python files of this project.

# Report
A detailed report describing the learning algorithm, along with ideas for future work is at [report.md](https://github.com/drormeir/ReacherRL/blob/master/Report.md)
