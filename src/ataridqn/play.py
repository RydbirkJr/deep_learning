import gym
from src.ataridqn.agent import Agent

env = gym.make('Breakout-v0')

agent = Agent(env, colors=False, scale=1, cropping=(30, 10, 6, 6), weights_file="weights.dump")
agent.validate(render_test=True)
