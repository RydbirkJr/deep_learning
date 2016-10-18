import time
import gym
from src.agent.agent_policy import AgentPolicy
from src.network.network import Network

# init and run an example environment

# init environment
env = gym.make('CartPole-v0')
# init agent

print env.observation_space
print env.action_space

network = Network(env.observation_space.shape[0], env.action_space.n)
agent = AgentPolicy(env, network)

# train agent on the environment
agent.learn(
    epochs=100,
    learning_rate=0.001,
    discount_factor=1,
    states_per_batch=10000,
    time_limit=10000,
    early_stop=5
)
