import gym
import numpy as np
from src.agent.agent_policy import AgentPolicy
from network.network import Network
from PIL import Image
import time

# init environment
env = gym.make('Pong-v0')
# init agent
# shape = env.observation_space.shape
# shape = (None, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])

#shape = (None, 110, 84, 1)
#shape = (None, 1, 110, 84)

cropping = (30, 10, 6, 6)


network = Network(resolution=env.observation_space.shape, number_of_outputs=3, cropping=cropping, weights_file='agent/weights.dump')
print 'Completed network'
agent = AgentPolicy(env, network)
print 'Completed policy'

agent.get_trajectory(0,0,deterministic=True, render=True)