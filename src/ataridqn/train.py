import gym
from agent import Agent

# init environment
env = gym.make('Pong-v0')

# init agent
agent = Agent(env, colors=False, scale=1, cropping=(0, 0, 0, 0))
#agent = Agent(env, colors=False, scale=.5, cropping=(30, 10, 6, 6))
# train agent on the environment
agent.learn(render_training=False, render_test=False)
#agent.learn(render_training=True, render_test=True, learning_steps_per_epoch=300)
