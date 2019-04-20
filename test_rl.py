import gym
import networks

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, MaxBoltzmannQPolicy, EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################

# PARAMS
np.random.seed(77)

# AGENT PARAMS
episodes = 100
memory_limit = 50000
window_length = 1
gamma = 0.7
nb_steps = 10000


# MEMORY PARAMS
batch_size = 1024
train_interval = 10

# POLICY PARAMS
eps = 0.30

####################################################################

env = gym.make('LunarLander-v2')

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

####################################################################

model = networks.standard(input_shape, nb_actions)
#model = networks.cnn(window_length, input_shape[0], nb_actions, n_sizes=[9, 7, 3, 2], n_filters=16)
#model = networks.lstm_network(window_length, input_shape[0], nb_actions)

####################################################################

memory = SequentialMemory(limit=memory_limit, window_length=window_length)

####################################################################

policy = EpsGreedyQPolicy(eps=eps)
policy = LinearAnnealedPolicy(policy, attr='eps', value_max=eps, 
							  value_min=0, value_test = 0, nb_steps=nb_steps)
test_policy = GreedyQPolicy()

####################################################################

dqn = DQNAgent(model=model, 
			   nb_actions=nb_actions,
			   memory=memory,
			   nb_steps_warmup=window_length+batch_size,
               target_model_update=0.02,
               policy=policy, test_policy = test_policy,
               batch_size=batch_size,
               train_interval=train_interval,
               gamma = gamma)

dqn.compile(Adam(lr=0.00025), metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format('lunar'))

####################################################################

nb_episodes = 10

history = dqn.test(env, nb_episodes=nb_episodes)