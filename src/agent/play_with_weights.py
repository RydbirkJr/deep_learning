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

#shape = (None, 110, 84, 1)
shape = (None, 1, 110, 84)

cropping = (30, 10, 6, 6)


network = Network(resolution=env.observation_space.shape, number_of_outputs=3, cropping=cropping)
print 'Completed network'
agent = AgentPolicy(env, network)
print 'Completed policy'


# train agent on the environment
agent.learn(
    epochs=500,
    learning_rate=0.00025,
    discount_factor=0.99,
    states_per_batch=15000,
    time_limit=10000,
    #early_stop=5
)
