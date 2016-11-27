import gym
from agent import Agent

env = gym.make('Pong-v0')

agent = Agent(env, colors=False, scale=1, cropping=(30, 10, 6, 6), weights_file="weights.dump.backup")
agent.validate(render_test=True)
