import gym
from agent import Agent

env = gym.make('Pong-v0')

agent = Agent(env, scale=1, cropping=(30, 10, 6, 6), weights_file="Weights/best_weightsPong1.dump", replay_memory_size=1)
agent.validate(render_test=True)
