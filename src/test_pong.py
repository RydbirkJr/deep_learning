import gym
import numpy as np
from src.agent.agent_policy import AgentPolicy
from src.network.network import Network

# init environment
env = gym.make('Pong-v0')

# init agent
# shape = env.observation_space.shape
shape = (None, env.observation_space.shape[0], env.observation_space.shape[1], env.observation_space.shape[2])

network = Network(shape, env.action_space.n)
agent = AgentPolicy(env, network)

# train agent on the environment
agent.learn(
    epochs=100,
    learning_rate=0.01,
    discount_factor=1,
    states_per_batch=10,
    time_limit=100,
    early_stop=5
)
