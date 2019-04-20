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
np.random.seed(192)

# AGENT PARAMS
gamma = 0.99
nb_steps = 5000

# TRAIN PARAMS
lr = 0.0000025
target_model_update = 0.02
action_repetition = 1

# MEMORY PARAMS
memory_limit = 500
window_length = 1
batch_size = 32
train_interval = 1

# POLICY PARAMS
eps = 0.05

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
               target_model_update=target_model_update,
               policy=policy, test_policy = test_policy,
               batch_size=batch_size,
               train_interval=train_interval,
               gamma = gamma)

dqn.compile(Adam(lr=lr), metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format('lunar'))

####################################################################

history = dqn.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, action_repetition=action_repetition)

####################################################################

nb_episodes = 5

history = dqn.test(env, nb_episodes=nb_episodes)

dqn.save_weights('dqn_{}_weights.h5f'.format('lunar'), overwrite=True)