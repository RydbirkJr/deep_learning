import gym

# init environment
from src.ataridqn.agent import Agent

env = gym.make('Breakout-v0')

# init agent
agent = Agent(env, colors=False, scale=1, cropping=(30, 10, 6, 6))
#agent = Agent(env, colors=False, scale=.5, cropping=(30, 10, 6, 6))
# train agent on the environment
agent.learn(render_training=True, render_test=False)
#agent.learn(render_training=True, render_test=True, learning_steps_per_epoch=300)
