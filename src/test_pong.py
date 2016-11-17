import gym
import numpy as np
from agent.agent_policy import AgentPolicy
from network.network import Network
from PIL import Image
import time

# init environment
env = gym.make('Pong-v0')
# init agent
# shape = env.observation_space.shape
# shape = (None, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])

shape = (None, 1, 110, 84)
#shape = (None, 3, 210, 160)

network = Network(shape, 3)
print 'Completed network'
agent = AgentPolicy(env, network)
print 'Completed policy'


# train agent on the environment
agent.learn(
    epochs=5,
    learning_rate=0.0001,
    discount_factor=0.99,
    states_per_batch=20000,
    time_limit=5000,
    #early_stop=5
)
