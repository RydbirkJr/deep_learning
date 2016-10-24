import time

import gym

from src.agent.agent_policy import AgentPolicy
from src.network.network import Network

# init and run an example environment

#env = gym.make('Pong-v0')
#env.reset()
#for _ in range(200):
#    env.render()
#    time.sleep(0.01)
#    env.step(env.action_space.sample())
#env.render(close=True)

# init environment
env = gym.make('CartPole-v0')
# init agent
network = Network(env.observation_space.shape[0], env.action_space.n)
agent = AgentPolicy(env, network)

# train agent on the environment
agent.learn(
    epochs=100,
    learning_rate=0.001,
    discount_factor=1.0,
    states_per_batch=50000,
    time_limit=1000,
    early_stop=5
)

